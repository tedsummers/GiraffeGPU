#include "uci.h"
#include "cuda/neural_gpu.h"
#include <iostream>

int main(int argc, char* argv[]) {
    // Check if arguments were passed
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--help" || arg == "-h") {
            std::cout << "GiraffeGPU - Neural Chess Engine running on NVIDIA GPUs\n";
            std::cout << "Based on the Giraffe Chess Engine by Matthew Lai\n\n";
            std::cout << "Usage:\n";
            std::cout << "  ./GiraffeGPU           Launch UCI protocol mode\n";
            std::cout << "  ./GiraffeGPU bench     Run benchmark suite\n";
            std::cout << "  ./GiraffeGPU gpu-info  List detected NVIDIA GPUs\n";
            return 0;
        } else if (arg == "gpu-info") {
            auto devices = Neural::NeuralGPU::list_devices();
            std::cout << "Detected " << devices.size() << " NVIDIA GPU(s):" << std::endl;
            for (const auto& d : devices) {
                std::cout << "  [" << d.id << "] " << d.name 
                          << " (Compute " << d.major << "." << d.minor 
                          << ", " << d.totalMemoryMB << " MB VRAM)" << std::endl;
            }
            return 0;
        } else if (arg == "bench") {
            int depth = 4;
            if (argc > 2) {
                try { depth = std::stoi(argv[2]); } catch (...) { depth = 4; }
            }
            UCI uci;
            uci.handle_bench(depth);
            return 0;
        }
    }

    std::cout << "GiraffeGPU 1.0 (NVIDIA CUDA Neural Chess Engine)" << std::endl;
    std::cout << "Based on Giraffe by Matthew Lai (Imperial College London)" << std::endl;
    std::cout << "Written by Gemini 3.8 Flash using directions from Ted Summers" << std::endl;

    UCI uci;
    uci.loop();

    return 0;
}
