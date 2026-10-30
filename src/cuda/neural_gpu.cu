#include "neural_gpu.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <random>
#include <future>
#include <sstream>

namespace Neural {

NeuralGPU g_neural_gpu;

// CUDA Kernels

__global__ void forward_l1_kernel(
    const float* __restrict__ d_in,
    const float* __restrict__ d_w1,
    const float* __restrict__ d_b1,
    float* __restrict__ d_h1,
    int batch_size)
{
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (neuron < L1_DIM && batch_idx < batch_size) {
        const float* in = d_in + batch_idx * IN_DIM;
        const float* w = d_w1 + neuron * IN_DIM;

        float sum = d_b1[neuron];
        #pragma unroll 4
        for (int i = 0; i < IN_DIM; ++i) {
            sum += in[i] * w[i];
        }

        // ReLU activation
        d_h1[batch_idx * L1_DIM + neuron] = (sum > 0.0f) ? sum : 0.0f;
    }
}

__global__ void forward_l2_kernel(
    const float* __restrict__ d_h1,
    const float* __restrict__ d_w2,
    const float* __restrict__ d_b2,
    float* __restrict__ d_h2,
    int batch_size)
{
    int neuron = threadIdx.x;
    int batch_idx = blockIdx.x;

    if (neuron < L2_DIM && batch_idx < batch_size) {
        const float* h1 = d_h1 + batch_idx * L1_DIM;
        const float* w = d_w2 + neuron * L1_DIM;

        float sum = d_b2[neuron];
        #pragma unroll 4
        for (int i = 0; i < L1_DIM; ++i) {
            sum += h1[i] * w[i];
        }

        // ReLU activation
        d_h2[batch_idx * L2_DIM + neuron] = (sum > 0.0f) ? sum : 0.0f;
    }
}

__global__ void forward_l3_kernel(
    const float* __restrict__ d_h2,
    const float* __restrict__ d_w3,
    const float* __restrict__ d_b3,
    float* __restrict__ d_out,
    int batch_size)
{
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;

    __shared__ float s_val[64];

    if (batch_idx < batch_size) {
        float prod = (tid < L2_DIM) ? (d_h2[batch_idx * L2_DIM + tid] * d_w3[tid]) : 0.0f;
        s_val[tid] = prod;
        __syncthreads();

        for (int s = 32; s > 0; s >>= 1) {
            if (tid < s) {
                s_val[tid] += s_val[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float sum = s_val[0] + d_b3[0];
            d_out[batch_idx] = tanhf(sum);
        }
    }
}

__global__ void forward_policy_kernel(
    const float* __restrict__ d_h2,
    const float* __restrict__ d_w_pol,
    const float* __restrict__ d_b_pol,
    float* __restrict__ d_policy,
    int batch_size)
{
    int pol_idx = blockIdx.x * blockDim.x + threadIdx.x; // 0 .. 4095
    int batch_idx = blockIdx.y;

    if (pol_idx < POLICY_DIM && batch_idx < batch_size) {
        const float* h2 = d_h2 + batch_idx * L2_DIM;
        const float* w = d_w_pol + pol_idx * L2_DIM;

        float sum = d_b_pol[pol_idx];
        #pragma unroll 4
        for (int i = 0; i < L2_DIM; ++i) {
            sum += h2[i] * w[i];
        }

        d_policy[batch_idx * POLICY_DIM + pol_idx] = sum;
    }
}

// DeviceContext Implementation

DeviceContext::DeviceContext()
    : device_id(0)
    , max_batch_size(DEFAULT_MAX_BATCH)
    , initialized(false)
    , h_input_batch(nullptr)
    , h_output_batch(nullptr)
    , h_policy_batch(nullptr)
    , d_w1(nullptr)
    , d_b1(nullptr)
    , d_w2(nullptr)
    , d_b2(nullptr)
    , d_w3(nullptr)
    , d_b3(nullptr)
    , d_w_pol(nullptr)
    , d_b_pol(nullptr)
    , d_input_batch(nullptr)
    , d_hidden1_batch(nullptr)
    , d_hidden2_batch(nullptr)
    , d_output_batch(nullptr)
    , d_policy_batch(nullptr)
    , cuda_stream(nullptr)
{}

DeviceContext::~DeviceContext() {
    shutdown();
}

bool DeviceContext::init(int dev_id, int batch_size,
                         const std::vector<float>& w1, const std::vector<float>& b1,
                         const std::vector<float>& w2, const std::vector<float>& b2,
                         const std::vector<float>& w3, const std::vector<float>& b3,
                         const std::vector<float>& w_pol, const std::vector<float>& b_pol)
{
    device_id = dev_id;
    max_batch_size = batch_size;

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "DeviceContext Error setting device " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cuda_stream = static_cast<void*>(stream);

    // Host pinned memory
    cudaMallocHost(&h_input_batch, max_batch_size * IN_DIM * sizeof(float));
    cudaMallocHost(&h_output_batch, max_batch_size * OUT_DIM * sizeof(float));
    cudaMallocHost(&h_policy_batch, max_batch_size * POLICY_DIM * sizeof(float));

    // Device weights
    cudaMalloc(&d_w1, IN_DIM * L1_DIM * sizeof(float));
    cudaMalloc(&d_b1, L1_DIM * sizeof(float));
    cudaMalloc(&d_w2, L1_DIM * L2_DIM * sizeof(float));
    cudaMalloc(&d_b2, L2_DIM * sizeof(float));
    cudaMalloc(&d_w3, L2_DIM * OUT_DIM * sizeof(float));
    cudaMalloc(&d_b3, OUT_DIM * sizeof(float));
    cudaMalloc(&d_w_pol, POLICY_DIM * L2_DIM * sizeof(float));
    cudaMalloc(&d_b_pol, POLICY_DIM * sizeof(float));

    // Device batch buffers
    cudaMalloc(&d_input_batch, max_batch_size * IN_DIM * sizeof(float));
    cudaMalloc(&d_hidden1_batch, max_batch_size * L1_DIM * sizeof(float));
    cudaMalloc(&d_hidden2_batch, max_batch_size * L2_DIM * sizeof(float));
    cudaMalloc(&d_output_batch, max_batch_size * OUT_DIM * sizeof(float));
    cudaMalloc(&d_policy_batch, max_batch_size * POLICY_DIM * sizeof(float));

    copy_weights(w1, b1, w2, b2, w3, b3, w_pol, b_pol);

    initialized = true;
    return true;
}

void DeviceContext::copy_weights(const std::vector<float>& w1, const std::vector<float>& b1,
                                 const std::vector<float>& w2, const std::vector<float>& b2,
                                 const std::vector<float>& w3, const std::vector<float>& b3,
                                 const std::vector<float>& w_pol, const std::vector<float>& b_pol)
{
    cudaSetDevice(device_id);
    cudaMemcpy(d_w1, w1.data(), IN_DIM * L1_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1.data(), L1_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, w2.data(), L1_DIM * L2_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2.data(), L2_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, w3.data(), L2_DIM * OUT_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, b3.data(), OUT_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_pol, w_pol.data(), POLICY_DIM * L2_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_pol, b_pol.data(), POLICY_DIM * sizeof(float), cudaMemcpyHostToDevice);
}

void DeviceContext::shutdown() {
    if (!initialized) return;

    cudaSetDevice(device_id);

    if (cuda_stream) {
        cudaStreamDestroy(static_cast<cudaStream_t>(cuda_stream));
        cuda_stream = nullptr;
    }

    if (h_input_batch) { cudaFreeHost(h_input_batch); h_input_batch = nullptr; }
    if (h_output_batch) { cudaFreeHost(h_output_batch); h_output_batch = nullptr; }
    if (h_policy_batch) { cudaFreeHost(h_policy_batch); h_policy_batch = nullptr; }

    if (d_w1) { cudaFree(d_w1); d_w1 = nullptr; }
    if (d_b1) { cudaFree(d_b1); d_b1 = nullptr; }
    if (d_w2) { cudaFree(d_w2); d_w2 = nullptr; }
    if (d_b2) { cudaFree(d_b2); d_b2 = nullptr; }
    if (d_w3) { cudaFree(d_w3); d_w3 = nullptr; }
    if (d_b3) { cudaFree(d_b3); d_b3 = nullptr; }
    if (d_w_pol) { cudaFree(d_w_pol); d_w_pol = nullptr; }
    if (d_b_pol) { cudaFree(d_b_pol); d_b_pol = nullptr; }

    if (d_input_batch) { cudaFree(d_input_batch); d_input_batch = nullptr; }
    if (d_hidden1_batch) { cudaFree(d_hidden1_batch); d_hidden1_batch = nullptr; }
    if (d_hidden2_batch) { cudaFree(d_hidden2_batch); d_hidden2_batch = nullptr; }
    if (d_output_batch) { cudaFree(d_output_batch); d_output_batch = nullptr; }
    if (d_policy_batch) { cudaFree(d_policy_batch); d_policy_batch = nullptr; }

    initialized = false;
}

Value DeviceContext::evaluate_single(const Board& board) {
    std::lock_guard<std::mutex> lock(mtx);
    cudaSetDevice(device_id);

    Giraffe::extract_features(board, h_input_batch);

    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
    cudaMemcpyAsync(d_input_batch, h_input_batch, IN_DIM * sizeof(float), cudaMemcpyHostToDevice, stream);

    dim3 grid_l1((L1_DIM + 63) / 64, 1);
    forward_l1_kernel<<<grid_l1, 64, 0, stream>>>(d_input_batch, d_w1, d_b1, d_hidden1_batch, 1);
    forward_l2_kernel<<<1, 64, 0, stream>>>(d_hidden1_batch, d_w2, d_b2, d_hidden2_batch, 1);
    forward_l3_kernel<<<1, 64, 0, stream>>>(d_hidden2_batch, d_w3, d_b3, d_output_batch, 1);

    cudaMemcpyAsync(h_output_batch, d_output_batch, OUT_DIM * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float raw_score = h_output_batch[0];
    float cp = raw_score * 1000.0f;
    Value val = static_cast<Value>(std::round(cp));
    return (board.side_to_move() == WHITE) ? val : -val;
}

Value DeviceContext::evaluate_dual(const Board& board, float* out_policy_4096) {
    std::lock_guard<std::mutex> lock(mtx);
    cudaSetDevice(device_id);

    Giraffe::extract_features(board, h_input_batch);

    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
    cudaMemcpyAsync(d_input_batch, h_input_batch, IN_DIM * sizeof(float), cudaMemcpyHostToDevice, stream);

    dim3 grid_l1((L1_DIM + 63) / 64, 1);
    forward_l1_kernel<<<grid_l1, 64, 0, stream>>>(d_input_batch, d_w1, d_b1, d_hidden1_batch, 1);
    forward_l2_kernel<<<1, 64, 0, stream>>>(d_hidden1_batch, d_w2, d_b2, d_hidden2_batch, 1);
    forward_l3_kernel<<<1, 64, 0, stream>>>(d_hidden2_batch, d_w3, d_b3, d_output_batch, 1);

    dim3 grid_pol(POLICY_DIM / 64, 1);
    forward_policy_kernel<<<grid_pol, 64, 0, stream>>>(d_hidden2_batch, d_w_pol, d_b_pol, d_policy_batch, 1);

    cudaMemcpyAsync(h_output_batch, d_output_batch, OUT_DIM * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (out_policy_4096) {
        cudaMemcpyAsync(h_policy_batch, d_policy_batch, POLICY_DIM * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    if (out_policy_4096) {
        std::memcpy(out_policy_4096, h_policy_batch, POLICY_DIM * sizeof(float));
    }

    float raw_score = h_output_batch[0];
    float cp = raw_score * 1000.0f;
    Value val = static_cast<Value>(std::round(cp));
    return (board.side_to_move() == WHITE) ? val : -val;
}

void DeviceContext::evaluate_batch_internal(const std::vector<Board>& boards, int start_idx, int count, std::vector<Value>& out_scores) {
    std::lock_guard<std::mutex> lock(mtx);
    cudaSetDevice(device_id);
    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

    for (int offset = 0; offset < count; offset += max_batch_size) {
        int current_batch = std::min(max_batch_size, count - offset);

        for (int b = 0; b < current_batch; ++b) {
            Giraffe::extract_features(boards[start_idx + offset + b], h_input_batch + b * IN_DIM);
        }

        cudaMemcpyAsync(d_input_batch, h_input_batch, current_batch * IN_DIM * sizeof(float), cudaMemcpyHostToDevice, stream);

        dim3 grid_l1((L1_DIM + 63) / 64, current_batch);
        forward_l1_kernel<<<grid_l1, 64, 0, stream>>>(d_input_batch, d_w1, d_b1, d_hidden1_batch, current_batch);
        forward_l2_kernel<<<current_batch, 64, 0, stream>>>(d_hidden1_batch, d_w2, d_b2, d_hidden2_batch, current_batch);
        forward_l3_kernel<<<current_batch, 64, 0, stream>>>(d_hidden2_batch, d_w3, d_b3, d_output_batch, current_batch);

        cudaMemcpyAsync(h_output_batch, d_output_batch, current_batch * OUT_DIM * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        for (int b = 0; b < current_batch; ++b) {
            float raw_score = h_output_batch[b];
            float cp = raw_score * 1000.0f;
            Value val = static_cast<Value>(std::round(cp));
            int dest = start_idx + offset + b;
            out_scores[dest] = (boards[dest].side_to_move() == WHITE) ? val : -val;
        }
    }
}

// NeuralGPU Implementation

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
    h_w_pol_.resize(POLICY_DIM * L2_DIM, 0.0f);
    h_b_pol_.resize(POLICY_DIM, 0.0f);
}

NeuralGPU::~NeuralGPU() {
    shutdown();
}

std::vector<GPUDeviceInfo> NeuralGPU::list_devices() {
    std::vector<GPUDeviceInfo> devices;
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        return devices;
    }

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            GPUDeviceInfo info;
            info.id = i;
            info.name = prop.name;
            info.major = prop.major;
            info.minor = prop.minor;
            info.totalMemoryMB = prop.totalGlobalMem / (1024 * 1024);
            devices.push_back(info);
        }
    }
    return devices;
}

bool NeuralGPU::init(int device_id, int max_batch_size) {
    shutdown();

    int total_gpus = 0;
    cudaGetDeviceCount(&total_gpus);
    if (total_gpus <= 0) {
        std::cerr << "NeuralGPU Error: No NVIDIA GPUs detected!" << std::endl;
        return false;
    }

    device_id_ = device_id;
    max_batch_size_ = max_batch_size;

    init_default_weights();

    std::vector<int> target_devices;
    if (device_id == ALL_GPUS) {
        // Initialize ALL available GPUs (Dual-GPU / Multi-GPU mode)
        for (int i = 0; i < total_gpus; ++i) {
            target_devices.push_back(i);
        }
    } else {
        if (device_id < 0 || device_id >= total_gpus) {
            std::cerr << "NeuralGPU Warning: Device " << device_id << " invalid. Defaulting to 0." << std::endl;
            device_id = 0;
        }
        target_devices.push_back(device_id);
    }

    for (int dev : target_devices) {
        auto ctx = std::make_unique<DeviceContext>();
        if (ctx->init(dev, max_batch_size_, h_w1_, h_b1_, h_w2_, h_b2_, h_w3_, h_b3_, h_w_pol_, h_b_pol_)) {
            contexts_.push_back(std::move(ctx));
        }
    }

    if (contexts_.empty()) {
        std::cerr << "NeuralGPU Error: Failed to initialize any GPU device!" << std::endl;
        return false;
    }

    initialized_ = true;

    std::cout << "info string " << device_summary() << std::endl;
    return true;
}

std::string NeuralGPU::device_summary() const {
    if (contexts_.empty()) return "No active GPU contexts";
    std::ostringstream oss;
    if (contexts_.size() > 1) {
        oss << "Initialized Dual/Multi-GPU (" << contexts_.size() << " GPUs active in parallel):";
        for (const auto& ctx : contexts_) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, ctx->device_id);
            oss << " [" << ctx->device_id << "] " << prop.name
                << " (" << (prop.totalGlobalMem / (1024 * 1024)) << " MB, SM "
                << prop.major << "." << prop.minor << ")";
        }
    } else {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, contexts_[0]->device_id);
        oss << "Initialized Single GPU: [" << contexts_[0]->device_id << "] " << prop.name
            << " (" << (prop.totalGlobalMem / (1024 * 1024)) << " MB, SM "
            << prop.major << "." << prop.minor << ")";
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
    std::fill(h_w_pol_.begin(), h_w_pol_.end(), 0.0f);
    std::fill(h_b_pol_.begin(), h_b_pol_.end(), 0.0f);

    // Neuron 0: White Material accumulator
    // Neuron 1: Black Material accumulator
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

    // Central square control bonus (e4, d4, e5, d5, c4, c5, f4, f5)
    const int central_squares[8] = { 27, 28, 35, 36, 26, 34, 29, 37 };
    for (int sq : central_squares) {
        int wCtrlIdx = 235 + sq * 2;
        int bCtrlIdx = 235 + sq * 2 + 1;
        h_w1_[2 * IN_DIM + wCtrlIdx] += 0.25f;
        h_w1_[3 * IN_DIM + bCtrlIdx] += 0.25f;
    }

    // Queen & Rook mobility
    for (int dir = 0; dir < 8; ++dir) {
        h_w1_[2 * IN_DIM + 99 + 3 + dir] += 0.05f;
        h_w1_[3 * IN_DIM + 113 + 3 + dir] += 0.05f;
    }

    // Castling rights bonus
    h_w1_[2 * IN_DIM + 13] += 0.15f; // W_SHORT
    h_w1_[2 * IN_DIM + 14] += 0.10f; // W_LONG
    h_w1_[3 * IN_DIM + 17] += 0.15f; // B_SHORT
    h_w1_[3 * IN_DIM + 18] += 0.10f; // B_LONG

    // Feature 10 is Side To Move: small tempo bonus (15 cp)
    h_w1_[4 * IN_DIM + 10] = 0.15f;

    // Layer 2: Signature layer
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

    // Layer 3: Output layer (Value Head)
    h_w3_[0] =  0.10f;
    h_w3_[1] = -0.10f;

    // Policy Head Priors:
    // Natural central destination squares bonus
    for (int from = 0; from < 64; ++from) {
        for (int to : central_squares) {
            h_b_pol_[(from << 6) | to] += 0.40f;
        }
    }

    auto set_prior = [&](int from, int to, float val) {
        h_b_pol_[(from << 6) | to] += val;
    };

    // White canonical opening moves
    set_prior(12, 28, 2.5f); // e2e4
    set_prior(11, 27, 2.5f); // d2d4
    set_prior(6,  21, 2.0f); // g1f3
    set_prior(1,  18, 1.8f); // b1c3
    set_prior(10, 26, 1.8f); // c2c4
    set_prior(4,  6,  2.0f); // e1g1 (O-O)
    set_prior(4,  2,  1.5f); // e1c1 (O-O-O)

    // Black canonical opening moves
    set_prior(52, 36, 2.5f); // e7e5
    set_prior(51, 35, 2.5f); // d7d5
    set_prior(62, 45, 2.0f); // g8f6
    set_prior(57, 42, 1.8f); // b8c6
    set_prior(50, 34, 1.8f); // c7c5
    set_prior(60, 62, 2.0f); // e8g8 (O-O)
    set_prior(60, 58, 1.5f); // e8c8 (O-O-O)

    // Connect shared L2 neurons to forward move vectors:
    for (int from = 0; from < 64; ++from) {
        for (int to = 0; to < 64; ++to) {
            int idx = (from << 6) | to;
            int delta_rank = (to / 8) - (from / 8);
            if (delta_rank > 0) {
                h_w_pol_[idx * L2_DIM + 0] = 0.04f * delta_rank;
            } else if (delta_rank < 0) {
                h_w_pol_[idx * L2_DIM + 1] = -0.04f * delta_rank;
            }
        }
    }

    for (auto& ctx : contexts_) {
        ctx->copy_weights(h_w1_, h_b1_, h_w2_, h_b2_, h_w3_, h_b3_, h_w_pol_, h_b_pol_);
    }
}

bool NeuralGPU::load_weights(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        std::cerr << "NeuralGPU: Cannot open weight file " << path << std::endl;
        return false;
    }

    char magic[8];
    is.read(magic, 8);
    if (std::strncmp(magic, "GIRAFDUAL", 8) == 0) {
        is.read(reinterpret_cast<char*>(h_w1_.data()), h_w1_.size() * sizeof(float));
        is.read(reinterpret_cast<char*>(h_b1_.data()), h_b1_.size() * sizeof(float));
        is.read(reinterpret_cast<char*>(h_w2_.data()), h_w2_.size() * sizeof(float));
        is.read(reinterpret_cast<char*>(h_b2_.data()), h_b2_.size() * sizeof(float));
        is.read(reinterpret_cast<char*>(h_w3_.data()), h_w3_.size() * sizeof(float));
        is.read(reinterpret_cast<char*>(h_b3_.data()), h_b3_.size() * sizeof(float));
        is.read(reinterpret_cast<char*>(h_w_pol_.data()), h_w_pol_.size() * sizeof(float));
        is.read(reinterpret_cast<char*>(h_b_pol_.data()), h_b_pol_.size() * sizeof(float));
    } else if (std::strncmp(magic, "GIRAFGPU", 8) == 0) {
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
        ctx->copy_weights(h_w1_, h_b1_, h_w2_, h_b2_, h_w3_, h_b3_, h_w_pol_, h_b_pol_);
    }
    return true;
}

bool NeuralGPU::save_weights(const std::string& path) {
    std::ofstream os(path, std::ios::binary);
    if (!os) return false;
    os.write("GIRAFDUAL", 8);
    os.write(reinterpret_cast<const char*>(h_w1_.data()), h_w1_.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_b1_.data()), h_b1_.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_w2_.data()), h_w2_.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_b2_.data()), h_b2_.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_w3_.data()), h_w3_.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_b3_.data()), h_b3_.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_w_pol_.data()), h_w_pol_.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_b_pol_.data()), h_b_pol_.size() * sizeof(float));
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

Value NeuralGPU::evaluate_dual(const Board& board, float* out_policy_4096, int worker_id) {
    if (!initialized_ || contexts_.empty()) {
        init(device_id_, max_batch_size_);
    }

    size_t idx = 0;
    if (contexts_.size() > 1) {
        if (worker_id >= 0) {
            idx = static_cast<size_t>(worker_id) % contexts_.size();
        } else {
            idx = (round_robin_counter_.fetch_add(1)) % contexts_.size();
        }
    }

    return contexts_[idx]->evaluate_dual(board, out_policy_4096);
}

void NeuralGPU::get_policy(const Board& board, std::vector<float>& out_logits, int worker_id) {
    out_logits.resize(POLICY_DIM);
    evaluate_dual(board, out_logits.data(), worker_id);
}

void NeuralGPU::get_move_probabilities(const Board& board, const std::vector<Move>& moves, std::vector<float>& out_probs, int worker_id) {
    out_probs.assign(moves.size(), 0.0f);
    if (moves.empty()) return;

    std::vector<float> policy(POLICY_DIM);
    evaluate_dual(board, policy.data(), worker_id);

    std::vector<float> legal_logits(moves.size());
    float max_logit = -1e9f;
    for (size_t i = 0; i < moves.size(); ++i) {
        int pol_idx = (static_cast<int>(moves[i].from_sq()) << 6) | static_cast<int>(moves[i].to_sq());
        float l = policy[pol_idx];
        if (moves[i].is_promotion() && moves[i].promotion_type() == QUEEN) {
            l += 0.5f;
        }
        legal_logits[i] = l;
        if (l > max_logit) max_logit = l;
    }

    float sum_exp = 0.0f;
    for (size_t i = 0; i < moves.size(); ++i) {
        float e = std::exp(legal_logits[i] - max_logit);
        out_probs[i] = e;
        sum_exp += e;
    }

    if (sum_exp > 0.0f) {
        for (size_t i = 0; i < moves.size(); ++i) {
            out_probs[i] /= sum_exp;
        }
    }
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

    // Distribute batch across all active GPUs concurrently!
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
