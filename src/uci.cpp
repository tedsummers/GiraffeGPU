#include "uci.h"
#include "search.h"
#include "tt.h"
#include "cuda/neural_gpu.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>

UCI::UCI() = default;

UCI::~UCI() {
    handle_stop();
}

void UCI::handle_stop() {
    g_search.stop();
    if (search_thread_.joinable()) {
        search_thread_.join();
    }
}

void UCI::handle_uci() {
    std::cout << "id name GiraffeGPU 1.0" << std::endl;
    std::cout << "id author Matthew Lai / Giraffe GPU Team" << std::endl;
    std::cout << "option name Hash type spin default 64 min 1 max 8192" << std::endl;
    std::cout << "option name Threads type spin default 2 min 1 max 64" << std::endl;
    std::cout << "option name Device_GPU type spin default -1 min -1 max 7" << std::endl;
    std::cout << "option name BatchSize type spin default 512 min 1 max 4096" << std::endl;
    std::cout << "option name Ponder type check default true" << std::endl;
    std::cout << "option name WeightsFile type string default eval.net" << std::endl;
    std::cout << "uciok" << std::endl;
}

void UCI::handle_isready() {
    if (!Neural::g_neural_gpu.is_initialized()) {
        Neural::g_neural_gpu.init(Neural::ALL_GPUS, 512);
    }
    std::cout << "readyok" << std::endl;
}

void UCI::handle_setoption(const std::string& line) {
    std::istringstream iss(line);
    std::string token, name, val_str;
    iss >> token; // "setoption"
    iss >> token; // "name"

    std::string opt_name;
    while (iss >> token && token != "value") {
        if (!opt_name.empty()) opt_name += " ";
        opt_name += token;
    }

    iss >> val_str;

    if (opt_name == "Hash") {
        try {
            int mb = std::stoi(val_str);
            g_tt.resize(mb);
            std::cout << "info string Hash set to " << g_tt.size_mb() << " MB" << std::endl;
        } catch (...) {}
    } else if (opt_name == "Threads") {
        try {
            int th = std::stoi(val_str);
            g_search.set_threads(th);
            std::cout << "info string Threads set to " << g_search.num_threads() << std::endl;
        } catch (...) {}
    } else if (opt_name == "Device_GPU") {
        try {
            int dev = std::stoi(val_str);
            handle_stop();
            Neural::g_neural_gpu.init(dev, Neural::g_neural_gpu.max_batch_size());
        } catch (...) {}
    } else if (opt_name == "BatchSize") {
        try {
            int bs = std::stoi(val_str);
            handle_stop();
            Neural::g_neural_gpu.init(Neural::g_neural_gpu.current_device(), bs);
        } catch (...) {}
    } else if (opt_name == "Ponder") {
        bool enable = (val_str == "true" || val_str == "1");
        g_search.set_ponder_enabled(enable);
        std::cout << "info string Ponder set to " << (enable ? "true" : "false") << std::endl;
    } else if (opt_name == "WeightsFile") {
        if (Neural::g_neural_gpu.load_weights(val_str)) {
            std::cout << "info string Loaded weights from " << val_str << std::endl;
        } else {
            std::cout << "info string Failed to load weights from " << val_str << std::endl;
        }
    }
}

void UCI::handle_ucinewgame() {
    handle_stop();
    g_tt.clear();
    board_.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void UCI::handle_position(const std::string& line) {
    handle_stop();
    std::istringstream iss(line);
    std::string token;
    iss >> token; // "position"
    iss >> token;

    if (token == "startpos") {
        board_.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        iss >> token; // maybe "moves"
    } else if (token == "fen") {
        std::string fen;
        while (iss >> token && token != "moves") {
            if (!fen.empty()) fen += " ";
            fen += token;
        }
        board_.set_fen(fen);
    }

    if (token == "moves") {
        while (iss >> token) {
            Move m = board_.parse_move_uci(token);
            if (!m.is_null()) {
                board_.make_move(m);
            }
        }
    }
}

void UCI::handle_go(const std::string& line) {
    handle_stop();

    SearchLimits limits;
    std::istringstream iss(line);
    std::string token;
    iss >> token; // "go"

    while (iss >> token) {
        if (token == "depth") {
            iss >> limits.depth;
        } else if (token == "movetime") {
            iss >> limits.movetime;
        } else if (token == "wtime") {
            iss >> limits.wtime;
        } else if (token == "btime") {
            iss >> limits.btime;
        } else if (token == "winc") {
            iss >> limits.winc;
        } else if (token == "binc") {
            iss >> limits.binc;
        } else if (token == "movestogo") {
            iss >> limits.movestogo;
        } else if (token == "nodes") {
            iss >> limits.nodes;
        } else if (token == "infinite") {
            limits.infinite = true;
        } else if (token == "ponder") {
            limits.ponder = true;
        }
    }

    if (!Neural::g_neural_gpu.is_initialized()) {
        Neural::g_neural_gpu.init(Neural::ALL_GPUS, 512);
    }

    // Spawn search thread
    search_thread_ = std::thread([this, limits]() {
        g_search.start(board_, limits);
    });
}

void UCI::handle_ponderhit() {
    g_search.ponderhit();
}

uint64_t UCI::perft(Board& b, int depth) {
    if (depth == 0) return 1ULL;

    std::vector<Move> moves;
    b.generate_legal_moves(moves);
    if (depth == 1) return moves.size();

    uint64_t total = 0;
    for (const auto& m : moves) {
        if (b.make_move(m)) {
            total += perft(b, depth - 1);
            b.unmake_move(m);
        }
    }
    return total;
}

void UCI::handle_perft(int depth) {
    auto t0 = std::chrono::steady_clock::now();
    std::vector<Move> moves;
    board_.generate_legal_moves(moves);

    uint64_t total = 0;
    std::cout << "\nPerft Depth " << depth << ":\n";
    for (const auto& m : moves) {
        if (board_.make_move(m)) {
            uint64_t sub = (depth > 1) ? perft(board_, depth - 1) : 1ULL;
            board_.unmake_move(m);
            total += sub;
            std::cout << m.to_uci() << ": " << sub << "\n";
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    int64_t ms = std::max(int64_t(1), std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
    std::cout << "\nTotal nodes: " << total << " in " << ms << " ms ("
              << (total * 1000 / ms) << " nps)\n" << std::endl;
}

void UCI::handle_eval() {
    if (!Neural::g_neural_gpu.is_initialized()) {
        Neural::g_neural_gpu.init(Neural::ALL_GPUS, 512);
    }
    Value val = Neural::g_neural_gpu.evaluate(board_);
    std::cout << "Giraffe GPU Evaluation: " << val << " cp (side to move perspective)" << std::endl;
    std::cout << "Side to move: " << (board_.side_to_move() == WHITE ? "White" : "Black") << std::endl;
}

void UCI::handle_gpu_info() {
    auto devices = Neural::NeuralGPU::list_devices();
    std::cout << "Detected " << devices.size() << " NVIDIA GPU(s):" << std::endl;
    for (const auto& d : devices) {
        std::cout << " [" << d.id << "] " << d.name
                  << " (Compute " << d.major << "." << d.minor
                  << ", VRAM: " << d.totalMemoryMB << " MB)" << std::endl;
    }
    if (Neural::g_neural_gpu.is_initialized()) {
        std::cout << "Active device: " << Neural::g_neural_gpu.current_device()
                  << " (BatchSize: " << Neural::g_neural_gpu.max_batch_size() << ")" << std::endl;
    }
}

void UCI::handle_bench(int depth) {
    if (!Neural::g_neural_gpu.is_initialized()) {
        Neural::g_neural_gpu.init(Neural::ALL_GPUS, 512);
    }

    const std::vector<std::string> bench_positions = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqk2r/pp2bppp/2n1pn2/2pp4/2PP4/2N1PN2/PP2BPPP/R1BQK2R w KQkq - 4 7",
        "r1b1k2r/ppppqppp/2n5/4P3/2B1n3/5N2/PPP2PPP/RN1Q1RK1 b kq - 1 7",
        "r2q1rk1/1pp2ppp/p1np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 9",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
    };

    std::cout << "Running Giraffe GPU benchmark at depth " << depth << " across "
              << bench_positions.size() << " positions...\n" << std::endl;

    uint64_t total_nodes = 0;
    auto t_start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < bench_positions.size(); ++i) {
        Board b(bench_positions[i]);
        SearchLimits lim;
        lim.depth = depth;
        g_search.start(b, lim);
        total_nodes += g_search.nodes_searched();
    }

    auto t_end = std::chrono::steady_clock::now();
    int64_t total_ms = std::max(int64_t(1), std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
    uint64_t nps = (total_nodes * 1000) / total_ms;

    std::cout << "\n============================================\n";
    std::cout << "Giraffe GPU Benchmark Summary:\n";
    std::cout << "Total nodes: " << total_nodes << "\n";
    std::cout << "Total time:  " << total_ms << " ms\n";
    std::cout << "Nodes/sec:   " << nps << " nps\n";
    std::cout << "============================================\n" << std::endl;
}

void UCI::loop() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd == "uci") {
            handle_uci();
        } else if (cmd == "isready") {
            handle_isready();
        } else if (cmd == "setoption") {
            handle_setoption(line);
        } else if (cmd == "ucinewgame") {
            handle_ucinewgame();
        } else if (cmd == "position") {
            handle_position(line);
        } else if (cmd == "go") {
            handle_go(line);
        } else if (cmd == "stop") {
            handle_stop();
        } else if (cmd == "ponderhit") {
            handle_ponderhit();
        } else if (cmd == "quit") {
            handle_stop();
            break;
        } else if (cmd == "d" || cmd == "print") {
            board_.print();
        } else if (cmd == "eval") {
            handle_eval();
        } else if (cmd == "gpu_info") {
            handle_gpu_info();
        } else if (cmd == "perft") {
            int d = 5;
            iss >> d;
            handle_perft(d);
        } else if (cmd == "bench") {
            int d = 6;
            iss >> d;
            handle_bench(d);
        }
    }
}
