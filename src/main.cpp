#include "uci.h"
#include "cuda/neural_gpu.h"
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char* argv[]) {
    // Check if arguments were passed
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--help" || arg == "-h") {
            std::cout << "GiraffeGPU - Neural Chess Engine running on NVIDIA GPUs\n";
            std::cout << "Based on the Giraffe Chess Engine by Matthew Lai\n\n";
            std::cout << "Usage:\n";
            std::cout << "  ./GiraffeGPU              Launch UCI protocol mode\n";
            std::cout << "  ./GiraffeGPU bench        Run benchmark suite\n";
            std::cout << "  ./GiraffeGPU policy [fen] Show Dual-Head Policy Network distribution\n";
            std::cout << "  ./GiraffeGPU gpu-info     List detected NVIDIA GPUs\n";
            return 0;
        } else if (arg == "policy") {
            std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
            if (argc > 2) {
                fen = argv[2];
                for (int i = 3; i < argc; ++i) {
                    fen += " ";
                    fen += argv[i];
                }
            }
            UCI uci;
            uci.handle_policy_cmd(fen);
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
        } else if (arg == "train") {
            if (argc < 5) {
                std::cerr << "Usage: ./GiraffeGPU train <samples.tsv> <in_net> <out_net> [epochs] [lr] [policy_lr] [l2_reg] [max_delta]\n";
                return 1;
            }
            std::string tsv_path = argv[2];
            std::string in_net = argv[3];
            std::string out_net = argv[4];
            int epochs = (argc > 5) ? std::stoi(argv[5]) : 25;
            float lr = (argc > 6) ? std::stof(argv[6]) : 0.0005f;
            float policy_lr = (argc > 7) ? std::stof(argv[7]) : 0.001f;
            float l2_reg = (argc > 8) ? std::stof(argv[8]) : 0.0002f;
            float max_delta = (argc > 9) ? std::stof(argv[9]) : 0.06f;

            std::ifstream ifs(tsv_path);
            if (!ifs) {
                std::cerr << "Error: Cannot open training samples file " << tsv_path << "\n";
                return 1;
            }
            std::vector<Neural::TrainingSample> samples;
            std::string line;
            while (std::getline(ifs, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                std::string fen_str, val_str, pol_str, weight_str;
                if (std::getline(iss, fen_str, '\t') &&
                    std::getline(iss, val_str, '\t') &&
                    std::getline(iss, pol_str, '\t') &&
                    std::getline(iss, weight_str, '\t')) {
                    Neural::TrainingSample s;
                    s.fen = fen_str;
                    s.target_val = std::stof(val_str);
                    s.move_pol_idx = std::stoi(pol_str);
                    s.move_weight = std::stof(weight_str);
                    samples.push_back(std::move(s));
                }
            }
            if (samples.empty()) {
                std::cerr << "Error: 0 training samples loaded from " << tsv_path << "\n";
                return 1;
            }

            if (!Neural::g_neural_gpu.init(Neural::ALL_GPUS, 512)) {
                return 1;
            }
            if (!in_net.empty() && in_net != "none") {
                Neural::g_neural_gpu.load_weights(in_net);
            }
            if (!Neural::g_neural_gpu.train_on_gpu(samples, epochs, lr, policy_lr, l2_reg, max_delta, 256)) {
                return 1;
            }
            if (!Neural::g_neural_gpu.save_weights(out_net)) {
                std::cerr << "Error: Failed to save trained weights to " << out_net << "\n";
                return 1;
            }
            std::cout << "Saved GPU-trained Dual-Head weights to " << out_net << std::endl;
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
