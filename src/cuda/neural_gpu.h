#ifndef NEURAL_GPU_H
#define NEURAL_GPU_H

#include "../types.h"
#include "../board.h"
#include "../giraffe_features.h"
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>

namespace Neural {

constexpr int IN_DIM = Giraffe::NUM_FEATURES; // 363
constexpr int L1_DIM = 512;
constexpr int L2_DIM = 64;
constexpr int OUT_DIM = 1;

constexpr int DEFAULT_MAX_BATCH = 512;
constexpr int ALL_GPUS = -1;

struct GPUDeviceInfo {
    int id;
    std::string name;
    int major;
    int minor;
    size_t totalMemoryMB;
};

struct DeviceContext {
    int device_id;
    int max_batch_size;
    bool initialized;

    // Pinned host buffers for fast PCIe transfer
    float* h_input_batch;
    float* h_output_batch;

    // Device weights
    float* d_w1;
    float* d_b1;
    float* d_w2;
    float* d_b2;
    float* d_w3;
    float* d_b3;

    // Device intermediate buffers
    float* d_input_batch;
    float* d_hidden1_batch;
    float* d_hidden2_batch;
    float* d_output_batch;

    void* cuda_stream;
    std::mutex mtx;

    DeviceContext();
    ~DeviceContext();

    bool init(int dev_id, int batch_size,
              const std::vector<float>& w1, const std::vector<float>& b1,
              const std::vector<float>& w2, const std::vector<float>& b2,
              const std::vector<float>& w3, const std::vector<float>& b3);
    void shutdown();
    void copy_weights(const std::vector<float>& w1, const std::vector<float>& b1,
                      const std::vector<float>& w2, const std::vector<float>& b2,
                      const std::vector<float>& w3, const std::vector<float>& b3);

    Value evaluate_single(const Board& board);
    void evaluate_batch_internal(const std::vector<Board>& boards, int start_idx, int count, std::vector<Value>& out_scores);
};

class NeuralGPU {
public:
    NeuralGPU();
    ~NeuralGPU();

    static std::vector<GPUDeviceInfo> list_devices();

    // device_id = -1 (ALL_GPUS) to use all detected GPUs concurrently
    bool init(int device_id = ALL_GPUS, int max_batch_size = DEFAULT_MAX_BATCH);
    void shutdown();

    bool load_weights(const std::string& path);
    bool save_weights(const std::string& path);
    void init_default_weights();

    // Returns centipawn score from the perspective of side to move
    Value evaluate(const Board& board, int worker_id = 0);

    // Parallel multi-GPU batched evaluation
    void evaluate_batch(const std::vector<Board>& boards, std::vector<Value>& out_scores);

    int current_device() const { return device_id_; }
    int max_batch_size() const { return max_batch_size_; }
    int active_gpu_count() const { return static_cast<int>(contexts_.size()); }
    bool is_initialized() const { return initialized_; }

    std::string device_summary() const;

private:
    int device_id_;
    int max_batch_size_;
    bool initialized_;

    // Master weights on host
    std::vector<float> h_w1_;
    std::vector<float> h_b1_;
    std::vector<float> h_w2_;
    std::vector<float> h_b2_;
    std::vector<float> h_w3_;
    std::vector<float> h_b3_;

    std::vector<std::unique_ptr<DeviceContext>> contexts_;
    std::atomic<uint64_t> round_robin_counter_;
};

extern NeuralGPU g_neural_gpu;

} // namespace Neural

#endif // NEURAL_GPU_H
