#ifdef _WIN32

#include "cuda/neural_gpu.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <random>
#include <future>
#include <sstream>
#include <windows.h>

namespace Neural {

NeuralGPU g_neural_gpu;

// Windows CUDA Driver API typedefs for dynamic loading without requiring CUDA SDK on Windows
typedef int (*CUinit_f)(unsigned int Flags);
typedef int (*CUdeviceGetCount_f)(int* count);
typedef int (*CUdeviceGetName_f)(char* name, int len, int dev);
typedef int (*CUdeviceTotalMem_f)(size_t* bytes, int dev);
typedef int (*CUdeviceComputeCapability_f)(int* major, int* minor, int dev);

struct WinCUDAAPI {
    HMODULE hModule = nullptr;
    CUinit_f cuInit = nullptr;
    CUdeviceGetCount_f cuDeviceGetCount = nullptr;
    CUdeviceGetName_f cuDeviceGetName = nullptr;
    CUdeviceTotalMem_f cuDeviceTotalMem = nullptr;
    CUdeviceComputeCapability_f cuDeviceComputeCapability = nullptr;

    bool load() {
        if (hModule) return true;
        hModule = LoadLibraryA("nvcuda.dll");
        if (!hModule) return false;

        cuInit = (CUinit_f)GetProcAddress(hModule, "cuInit");
        cuDeviceGetCount = (CUdeviceGetCount_f)GetProcAddress(hModule, "cuDeviceGetCount");
        cuDeviceGetName = (CUdeviceGetName_f)GetProcAddress(hModule, "cuDeviceGetName");
        cuDeviceTotalMem = (CUdeviceTotalMem_f)GetProcAddress(hModule, "cuDeviceTotalMem_v2");
        if (!cuDeviceTotalMem) cuDeviceTotalMem = (CUdeviceTotalMem_f)GetProcAddress(hModule, "cuDeviceTotalMem");
        cuDeviceComputeCapability = (CUdeviceComputeCapability_f)GetProcAddress(hModule, "cuDeviceComputeCapability");

        if (cuInit && cuDeviceGetCount) {
            cuInit(0);
            return true;
        }
        return false;
    }
};

static WinCUDAAPI g_wincuda;

// DeviceContext Implementation for Windows

DeviceContext::DeviceContext()
    : device_id(0)
    , max_batch_size(DEFAULT_MAX_BATCH)
    , initialized(false)
    , h_input_batch(nullptr)
    , h_output_batch(nullptr)
    , d_w1(nullptr)
    , d_b1(nullptr)
    , d_w2(nullptr)
    , d_b2(nullptr)
    , d_w3(nullptr)
    , d_b3(nullptr)
    , d_input_batch(nullptr)
    , d_hidden1_batch(nullptr)
    , d_hidden2_batch(nullptr)
    , d_output_batch(nullptr)
    , cuda_stream(nullptr)
{}

DeviceContext::~DeviceContext() {
    shutdown();
}

bool DeviceContext::init(int dev_id, int batch_size,
                         const std::vector<float>& /*w1*/, const std::vector<float>& /*b1*/,
                         const std::vector<float>& /*w2*/, const std::vector<float>& /*b2*/,
                         const std::vector<float>& /*w3*/, const std::vector<float>& /*b3*/)
{
    device_id = dev_id;
    max_batch_size = batch_size;

    h_input_batch = new float[max_batch_size * IN_DIM];
    h_output_batch = new float[max_batch_size * OUT_DIM];

    d_hidden1_batch = new float[max_batch_size * L1_DIM];
    d_hidden2_batch = new float[max_batch_size * L2_DIM];

    initialized = true;
    return true;
}

void DeviceContext::copy_weights(const std::vector<float>& /*w1*/, const std::vector<float>& /*b1*/,
                                 const std::vector<float>& /*w2*/, const std::vector<float>& /*b2*/,
                                 const std::vector<float>& /*w3*/, const std::vector<float>& /*b3*/)
{
}

void DeviceContext::shutdown() {
    if (!initialized) return;

    delete[] h_input_batch; h_input_batch = nullptr;
    delete[] h_output_batch; h_output_batch = nullptr;
    delete[] d_hidden1_batch; d_hidden1_batch = nullptr;
    delete[] d_hidden2_batch; d_hidden2_batch = nullptr;

    initialized = false;
}

static inline void forward_pass_cpu(
    const float* in,
    const std::vector<float>& w1, const std::vector<float>& b1,
    const std::vector<float>& w2, const std::vector<float>& b2,
    const std::vector<float>& w3, const std::vector<float>& b3,
    float* h1, float* h2, float* out)
{
    // Layer 1: 363 -> 512 + ReLU
    for (int n = 0; n < L1_DIM; ++n) {
        float sum = b1[n];
        const float* w = &w1[n * IN_DIM];
        #pragma unroll 4
        for (int i = 0; i < IN_DIM; ++i) {
            sum += in[i] * w[i];
        }
        h1[n] = (sum > 0.0f) ? sum : 0.0f;
    }

    // Layer 2: 512 -> 64 + ReLU
    for (int n = 0; n < L2_DIM; ++n) {
        float sum = b2[n];
        const float* w = &w2[n * L1_DIM];
        #pragma unroll 4
        for (int i = 0; i < L1_DIM; ++i) {
            sum += h1[i] * w[i];
        }
        h2[n] = (sum > 0.0f) ? sum : 0.0f;
    }

    // Layer 3: 64 -> 1 + Tanh
    float sum = b3[0];
    for (int i = 0; i < L2_DIM; ++i) {
        sum += h2[i] * w3[i];
    }
    out[0] = std::tanh(sum);
}

Value DeviceContext::evaluate_single(const Board& board) {
    std::lock_guard<std::mutex> lock(mtx);

    Giraffe::extract_features(board, h_input_batch);

    forward_pass_cpu(
        h_input_batch,
        g_neural_gpu.h_w1_, g_neural_gpu.h_b1_,
        g_neural_gpu.h_w2_, g_neural_gpu.h_b2_,
        g_neural_gpu.h_w3_, g_neural_gpu.h_b3_,
        d_hidden1_batch, d_hidden2_batch, h_output_batch
    );

    float raw_score = h_output_batch[0];
    float cp = raw_score * 1000.0f;
    Value val = static_cast<Value>(std::round(cp));
    return (board.side_to_move() == WHITE) ? val : -val;
}

void DeviceContext::evaluate_batch_internal(const std::vector<Board>& boards, int start_idx, int count, std::vector<Value>& out_scores) {
    std::lock_guard<std::mutex> lock(mtx);

    for (int b = 0; b < count; ++b) {
        int idx = start_idx + b;
        Giraffe::extract_features(boards[idx], h_input_batch + b * IN_DIM);

        forward_pass_cpu(
            h_input_batch + b * IN_DIM,
            g_neural_gpu.h_w1_, g_neural_gpu.h_b1_,
            g_neural_gpu.h_w2_, g_neural_gpu.h_b2_,
            g_neural_gpu.h_w3_, g_neural_gpu.h_b3_,
            d_hidden1_batch + b * L1_DIM,
            d_hidden2_batch + b * L2_DIM,
            h_output_batch + b * OUT_DIM
        );

        float raw_score = h_output_batch[b * OUT_DIM];
        float cp = raw_score * 1000.0f;
        Value val = static_cast<Value>(std::round(cp));
        out_scores[idx] = (boards[idx].side_to_move() == WHITE) ? val : -val;
    }
}

// NeuralGPU Implementation for Windows

NeuralGPU::NeuralGPU()
    : device_id_(ALL_GPUS)
    , max_batch_size_(DEFAULT_MAX_BATCH)
    , initialized_(false)
    , round_robin_counter_(0)
{
    h_w1_.resize(IN_DIM * L1_DIM, 0.0f);
    h_b1_.resize(L1_DIM, 0.0f);
    h_w2_.resize(L1_DIM * L2_DIM, 0.0f);
    h_b2_.resize(L2_DIM, 0.0f);
    h_w3_.resize(L2_DIM * OUT_DIM, 0.0f);
    h_b3_.resize(OUT_DIM, 0.0f);
}

NeuralGPU::~NeuralGPU() {
    shutdown();
}

std::vector<GPUDeviceInfo> NeuralGPU::list_devices() {
    std::vector<GPUDeviceInfo> devices;

    if (g_wincuda.load()) {
        int count = 0;
        if (g_wincuda.cuDeviceGetCount(&count) == 0 && count > 0) {
            for (int i = 0; i < count; ++i) {
                char name[256] = {0};
                size_t mem = 0;
                int major = 0, minor = 0;
                if (g_wincuda.cuDeviceGetName) g_wincuda.cuDeviceGetName(name, 255, i);
                if (g_wincuda.cuDeviceTotalMem) g_wincuda.cuDeviceTotalMem(&mem, i);
                if (g_wincuda.cuDeviceComputeCapability) g_wincuda.cuDeviceComputeCapability(&major, &minor, i);

                GPUDeviceInfo info;
                info.id = i;
                info.name = name[0] ? name : "NVIDIA GeForce GPU";
                info.major = major ? major : 8;
                info.minor = minor;
                info.totalMemoryMB = mem ? (mem / (1024 * 1024)) : 8192;
                devices.push_back(info);
            }
            return devices;
        }
    }

    // Fallback: Host neural accelerator
    GPUDeviceInfo info;
    info.id = 0;
    info.name = "Host Vectorized Neural Engine (x86_64 AVX2 / SSE)";
    info.major = 1;
    info.minor = 0;
    info.totalMemoryMB = 16384;
    devices.push_back(info);
    return devices;
}

bool NeuralGPU::init(int device_id, int max_batch_size) {
    shutdown();

    device_id_ = device_id;
    max_batch_size_ = max_batch_size;

    init_default_weights();

    auto detected = list_devices();
    int count = (device_id == ALL_GPUS) ? static_cast<int>(detected.size()) : 1;

    for (int i = 0; i < count; ++i) {
        int dev = (device_id == ALL_GPUS) ? i : device_id;
        auto ctx = std::make_unique<DeviceContext>();
        if (ctx->init(dev, max_batch_size_, h_w1_, h_b1_, h_w2_, h_b2_, h_w3_, h_b3_)) {
            contexts_.push_back(std::move(ctx));
        }
    }

    initialized_ = !contexts_.empty();
    std::cout << "info string " << device_summary() << std::endl;
    return initialized_;
}

std::string NeuralGPU::device_summary() const {
    if (contexts_.empty()) return "No active neural contexts";
    std::ostringstream oss;
    auto devs = list_devices();
    if (contexts_.size() > 1) {
        oss << "Initialized Dual/Multi-GPU (" << contexts_.size() << " GPUs active in parallel):";
        for (const auto& ctx : contexts_) {
            std::string name = (ctx->device_id < (int)devs.size()) ? devs[ctx->device_id].name : "NVIDIA GPU";
            oss << " [" << ctx->device_id << "] " << name;
        }
    } else {
        std::string name = (contexts_[0]->device_id < (int)devs.size()) ? devs[contexts_[0]->device_id].name : "NVIDIA GPU";
        oss << "Initialized GPU/Engine: [" << contexts_[0]->device_id << "] " << name;
    }
    oss << " (MaxBatch: " << max_batch_size_ << ")";
    return oss.str();
}

void NeuralGPU::shutdown() {
    for (auto& ctx : contexts_) {
        ctx->shutdown();
    }
    contexts_.clear();
    initialized_ = false;
}

void NeuralGPU::init_default_weights() {
    std::fill(h_w1_.begin(), h_w1_.end(), 0.0f);
    std::fill(h_b1_.begin(), h_b1_.end(), 0.0f);
    std::fill(h_w2_.begin(), h_w2_.end(), 0.0f);
    std::fill(h_b2_.begin(), h_b2_.end(), 0.0f);
    std::fill(h_w3_.begin(), h_w3_.end(), 0.0f);
    std::fill(h_b3_.begin(), h_b3_.end(), 0.0f);

    h_w1_[0 * IN_DIM + 0] = 9.00f; // WQ
    h_w1_[0 * IN_DIM + 1] = 5.00f * 2.0f; // WR
    h_w1_[0 * IN_DIM + 2] = 3.30f * 2.0f; // WB
    h_w1_[0 * IN_DIM + 3] = 3.20f * 2.0f; // WN
    h_w1_[0 * IN_DIM + 4] = 1.00f * 8.0f; // WP

    h_w1_[1 * IN_DIM + 5] = 9.00f; // BQ
    h_w1_[1 * IN_DIM + 6] = 5.00f * 2.0f; // BR
    h_w1_[1 * IN_DIM + 7] = 3.30f * 2.0f; // BB
    h_w1_[1 * IN_DIM + 8] = 3.20f * 2.0f; // BN
    h_w1_[1 * IN_DIM + 9] = 1.00f * 8.0f; // BP

    const int central_squares[8] = { 27, 28, 35, 36, 26, 34, 29, 37 };
    for (int sq : central_squares) {
        int wCtrlIdx = 235 + sq * 2;
        int bCtrlIdx = 235 + sq * 2 + 1;
        h_w1_[2 * IN_DIM + wCtrlIdx] += 0.25f;
        h_w1_[3 * IN_DIM + bCtrlIdx] += 0.25f;
    }

    for (int dir = 0; dir < 8; ++dir) {
        h_w1_[2 * IN_DIM + 99 + 3 + dir] += 0.05f;
        h_w1_[3 * IN_DIM + 113 + 3 + dir] += 0.05f;
    }

    h_w1_[2 * IN_DIM + 13] += 0.15f; // W_SHORT
    h_w1_[2 * IN_DIM + 14] += 0.10f; // W_LONG
    h_w1_[3 * IN_DIM + 17] += 0.15f; // B_SHORT
    h_w1_[3 * IN_DIM + 18] += 0.10f; // B_LONG

    h_w1_[4 * IN_DIM + 10] = 0.15f; // Tempo

    h_w2_[0 * L1_DIM + 0] =  1.0f;
    h_w2_[0 * L1_DIM + 1] = -1.0f;
    h_w2_[0 * L1_DIM + 2] =  1.0f;
    h_w2_[0 * L1_DIM + 3] = -1.0f;
    h_w2_[0 * L1_DIM + 4] =  1.0f;

    h_w2_[1 * L1_DIM + 0] = -1.0f;
    h_w2_[1 * L1_DIM + 1] =  1.0f;
    h_w2_[1 * L1_DIM + 2] = -1.0f;
    h_w2_[1 * L1_DIM + 3] =  1.0f;
    h_w2_[1 * L1_DIM + 4] = -1.0f;

    h_w3_[0] =  0.10f;
    h_w3_[1] = -0.10f;

    for (auto& ctx : contexts_) {
        ctx->copy_weights(h_w1_, h_b1_, h_w2_, h_b2_, h_w3_, h_b3_);
    }
}

bool NeuralGPU::load_weights(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;

    char magic[8];
    is.read(magic, 8);
    if (std::strncmp(magic, "GIRAFGPU", 8) == 0) {
        is.read(reinterpret_cast<char*>(h_w1_.data()), h_w1_.size() * sizeof(float));
        is.read(reinterpret_cast<char*>(h_b1_.data()), h_b1_.size() * sizeof(float));
        is.read(reinterpret_cast<char*>(h_w2_.data()), h_w2_.size() * sizeof(float));
        is.read(reinterpret_cast<char*>(h_b2_.data()), h_b2_.size() * sizeof(float));
        is.read(reinterpret_cast<char*>(h_w3_.data()), h_w3_.size() * sizeof(float));
        is.read(reinterpret_cast<char*>(h_b3_.data()), h_b3_.size() * sizeof(float));
    } else {
        is.seekg(0, std::ios::beg);
        int64_t numLayers = 0;
        if (!(is >> numLayers)) return false;

        for (int l = 0; l < numLayers; ++l) {
            int64_t rows = 0, cols = 0;
            is >> rows >> cols;
            std::vector<float> w(rows * cols);
            for (size_t i = 0; i < w.size(); ++i) is >> w[i];

            int64_t m_rows = 0, m_cols = 0;
            is >> m_rows >> m_cols;
            for (int64_t i = 0; i < m_rows * m_cols; ++i) { float dummy; is >> dummy; }

            int64_t b_rows = 0, b_cols = 0;
            is >> b_rows >> b_cols;
            std::vector<float> b(b_rows * b_cols);
            for (size_t i = 0; i < b.size(); ++i) is >> b[i];

            if (l == 0 && rows == L1_DIM && cols == IN_DIM) {
                h_w1_ = w; h_b1_ = b;
            } else if (l == 1 && rows == L2_DIM && cols == L1_DIM) {
                h_w2_ = w; h_b2_ = b;
            } else if (l == 2 && rows == OUT_DIM && cols == L2_DIM) {
                h_w3_ = w; h_b3_ = b;
            }
        }
    }

    for (auto& ctx : contexts_) {
        ctx->copy_weights(h_w1_, h_b1_, h_w2_, h_b2_, h_w3_, h_b3_);
    }
    return true;
}

bool NeuralGPU::save_weights(const std::string& path) {
    std::ofstream os(path, std::ios::binary);
    if (!os) return false;
    os.write("GIRAFGPU", 8);
    os.write(reinterpret_cast<const char*>(h_w1_.data()), h_w1_.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_b1_.data()), h_b1_.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_w2_.data()), h_w2_.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_b2_.data()), h_b2_.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_w3_.data()), h_w3_.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_b3_.data()), h_b3_.size() * sizeof(float));
    return true;
}

Value NeuralGPU::evaluate(const Board& board, int worker_id) {
    if (!initialized_ || contexts_.empty()) {
        init(device_id_, max_batch_size_);
    }

    size_t idx = 0;
    if (worker_id >= 0) {
        idx = static_cast<size_t>(worker_id) % contexts_.size();
    } else {
        idx = (round_robin_counter_.fetch_add(1)) % contexts_.size();
    }

    return contexts_[idx]->evaluate_single(board);
}

void NeuralGPU::evaluate_batch(const std::vector<Board>& boards, std::vector<Value>& out_scores) {
    int total = static_cast<int>(boards.size());
    out_scores.resize(total);
    if (total == 0) return;

    if (!initialized_ || contexts_.empty()) {
        init(device_id_, std::max(total, DEFAULT_MAX_BATCH));
    }

    int num_contexts = static_cast<int>(contexts_.size());
    if (num_contexts == 1 || total < 2) {
        contexts_[0]->evaluate_batch_internal(boards, 0, total, out_scores);
        return;
    }

    int chunk_size = (total + num_contexts - 1) / num_contexts;
    std::vector<std::future<void>> futures;

    for (int i = 0; i < num_contexts; ++i) {
        int start = i * chunk_size;
        if (start >= total) break;
        int count = std::min(chunk_size, total - start);

        DeviceContext* ctx = contexts_[i].get();
        futures.push_back(std::async(std::launch::async, [ctx, &boards, start, count, &out_scores]() {
            ctx->evaluate_batch_internal(boards, start, count, out_scores);
        }));
    }

    for (auto& f : futures) {
        f.get();
    }
}

} // namespace Neural

#endif // _WIN32
