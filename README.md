# GiraffeGPU

A high-performance **NVIDIA GPU-accelerated Chess Engine** based on Matthew Lai's landmark **Giraffe** neural network chess architecture ([arXiv:1509.01549](http://arxiv.org/abs/1509.01549), [chessprogramming.org/Giraffe](https://chessprogramming.org/Giraffe), and [ianfab/Giraffe](https://github.com/ianfab/Giraffe)).

---

## Highlights

- **Native NVIDIA CUDA Acceleration**: Executes the neural network forward pass directly on NVIDIA GPUs using optimized CUDA kernels (`forward_l1_kernel`, `forward_l2_kernel`, `forward_l3_kernel`).
- **Faithful Giraffe Feature Representation**: Computes the complete **363-dimensional** Giraffe feature vector:
  - **Global features (19)**: Material counts (WQ, WR, WB, WN, WP, BQ, BR, BB, BN, BP), side to move, king coordinates, and short/long castling rights.
  - **Pawn features (80)**: 16 pawn slots (8 White, 8 Black) assigned to nearest files, tracking presence, $(x, y)$ coordinates, and White/Black control values.
  - **Queen features (28)**: White & Black queens with presence, $(x, y)$, 8 directional ray mobility counters, safe moves count, and threat control.
  - **Rook features (44)**: 4 rooks with $(x, y)$, 4 orthogonal ray mobility counters, safe moves count, threat control, and castling availability.
  - **Bishop features (40)**: 4 bishops with $(x, y)$, 4 diagonal ray mobility counters, safe moves count, and threat control.
  - **Knight features (24)**: 4 knights with $(x, y)$, safe jump counts, and threat control.
  - **Square Attack Maps (128)**: Least Valuable Attacker (LVA) control values for all 64 squares from both White and Black perspectives.
- **Batched GPU Evaluation**: Evaluates single positions or large batches of positions (up to 4096) in parallel across CUDA warps and streaming multiprocessors.
- **Multi-GPU & Runtime Switching**: Automatically detects and supports modern NVIDIA architectures (Ampere `sm_86`, Blackwell `sm_120`, etc.), switchable dynamically via standard UCI options.
- **Universal Chess Interface (UCI) Protocol**: Compatible with all standard chess GUIs (CuteChess, Arena, Scid vs PC, Nibbler, etc.).
- **Engine Search Features**:
  - Negamax with Principal Variation Search (PVS)
  - Iterative Deepening with Aspiration Windows
  - 64-bit Zobrist Transposition Table (with dynamic sizing & replacement)
  - Quiescence Search with Static Exchange Evaluation (SEE) pruning
  - Move ordering with TT move, MVV-LVA, SEE, Killer moves (2 plies), and History heuristic
  - Null Move Pruning & Late Move Reductions (LMR)
  - Check extensions and threefold repetition detection

---

## Directory Structure

```
GiraffeGPU/
├── Makefile                # Fast build script targeting NVCC and G++
├── CMakeLists.txt          # Modern CMake build configuration
├── README.md               # Documentation
└── src/
    ├── types.h             # Core chess types, Move encoding, bitboard definitions
    ├── bitboard.h/.cpp     # Bitboard utilities and attack lookups
    ├── magic_moves.h/.cpp  # Pradu Cannan magic bitboards for sliding pieces
    ├── board.h/.cpp        # 64-bit Bitboard chess rules, FEN parser, move generation
    ├── zobrist.h/.cpp      # 64-bit Zobrist hashing keys
    ├── see.h/.cpp          # Static Exchange Evaluation (SEE)
    ├── giraffe_features.h/.cpp # Exact 363-dim Giraffe feature extraction
    ├── tt.h/.cpp           # Transposition Table implementation
    ├── search.h/.cpp       # PVS Search, Iterative Deepening, Time Management
    ├── uci.h/.cpp          # Universal Chess Interface handler & diagnostic CLI
    ├── main.cpp            # Engine entry point
    └── cuda/
        ├── neural_gpu.h    # Host neural inference and GPU device manager
        └── neural_gpu.cu   # Custom CUDA forward pass kernels & weight loader
```

---

## Building

### Requirements
- Linux (x86_64)
- NVIDIA GPU Driver & CUDA Toolkit (`nvcc` 11.0+)
- C++17 compliant compiler (`g++` 8+)

### Build with Make
```bash
cd /home/idiot/Chess/GiraffeGPU
make -j$(nproc)
```

The compiled binary will be placed at `./GiraffeGPU`.

### Build with CMake
```bash
mkdir -p build && cd build
cmake ..
cmake --build . -j$(nproc)
```

---

## Usage

### Command Line Diagnostics
```bash
# Display detected NVIDIA GPUs and VRAM
./GiraffeGPU gpu-info

# Run benchmark across standard test positions
./GiraffeGPU bench 5
```

### UCI Protocol Mode
Launch `./GiraffeGPU` directly:
```bash
./GiraffeGPU
```

#### Supported UCI Commands:
- `uci` - Returns engine identity and available configuration options.
- `isready` - Confirms CUDA initialization on the selected GPU device.
- `setoption name Device_GPU value <id>` - Selects GPU: `-1` (default) for **Both GPUs in parallel**, or `0` for GPU 0 only (RTX 3060), or `1` for GPU 1 only (RTX 5050).
- `setoption name Threads value <num>` - Sets search worker threads (default `2` to saturate both GPUs concurrently).
- `setoption name Hash value <MB>` - Sets transposition table size in MB (e.g., `256`).
- `setoption name BatchSize value <size>` - Sets maximum GPU evaluation batch size.
- `setoption name WeightsFile value <path>` - Loads custom Giraffe network weights (e.g., `eval.net`).
- `ucinewgame` - Resets transposition table and board state for a new game.
- `position startpos [moves e2e4 ...]` - Sets up the board position.
- `position fen <fen> [moves ...]` - Sets up position from FEN.
- `go [depth <d>] [movetime <ms>] [wtime <ms> btime <ms> winc <ms> binc <ms>]` - Starts search.
- `stop` - Stops search and returns best move immediately.
- `eval` - Evaluates current position on the GPU and prints centipawns.
- `perft <depth>` - Performs move generation test.
- `d` - Prints formatted ASCII board representation.
- `quit` - Exits the engine cleanly.

---

## Playing with CuteChess

GiraffeGPU can play against other engines using `cutechess-cli`:

```bash
cutechess-cli \
  -engine cmd=/home/idiot/Chess/GiraffeGPU/GiraffeGPU name=GiraffeGPU \
  -engine cmd=/path/to/other/engine name=Opponent \
  -each proto=uci tc=10+0.1 -rounds 10 -concurrency 1
```

---

## References

1. Matthew Lai, *"Giraffe: Using Deep Reinforcement Learning to Play Chess"*, MSc Thesis, Imperial College London, 2015. [arXiv:1509.01549](http://arxiv.org/abs/1509.01549)
2. Chessprogramming Wiki: [Giraffe](https://chessprogramming.org/Giraffe)
3. Giraffe GitHub Repository: [ianfab/Giraffe](https://github.com/ianfab/Giraffe)
