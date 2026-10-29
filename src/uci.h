#ifndef UCI_H
#define UCI_H

#include "board.h"
#include <string>
#include <thread>

class UCI {
public:
    UCI();
    ~UCI();

    void loop();
    void handle_bench(int depth);

private:
    void handle_uci();
    void handle_isready();
    void handle_setoption(const std::string& line);
    void handle_ucinewgame();
    void handle_position(const std::string& line);
    void handle_go(const std::string& line);
    void handle_stop();

    void handle_perft(int depth);
    void handle_eval();
    void handle_gpu_info();

    uint64_t perft(Board& b, int depth);

    Board board_;
    std::thread search_thread_;
};

#endif // UCI_H
