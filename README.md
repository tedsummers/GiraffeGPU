# GiraffeGPU

<p align="center">
  <img src="assets/Giraffe_Logo.png" alt="GiraffeGPU Logo" width="350"/>
</p>

<p align="center">
  <strong>A high-performance NVIDIA GPU-accelerated Chess Engine based on Matthew Lai's Giraffe neural architecture. The goal is try to bring it to Master level 2400-2500 Elo.</strong><br/>
  <em>(<a href="http://arxiv.org/abs/1509.01549">arXiv:1509.01549</a>, <a href="https://chessprogramming.org/Giraffe">chessprogramming.org/Giraffe</a>, and <a href="https://github.com/ianfab/Giraffe">ianfab/Giraffe</a>)</em>
</p>

> **Authorship**: Written by **Gemini 3.8 Flash** using directions from **Ted Summers**; bug-checked and debugged by **GPT 5.6 Luna** with assistance from **Ted Summers**.

---

## Highlights

- **Native NVIDIA CUDA Acceleration**: Executes the neural network forward pass directly on NVIDIA GPUs using optimized CUDA kernels (`forward_l1_kernel`, `forward_l2_kernel`, `forward_l3_kernel`, `forward_policy_kernel`).
- **Dual-Head Neural Architecture (Value + Policy)**:
  - **Shared Trunk**: 363-dimensional input $\to$ 512 hidden ($L_1$, ReLU) $\to$ 64 signature ($L_2$, ReLU).
  - **Value Head**: $64 \to 1$ scalar evaluation in centipawns ($\tanh$ activation).
  - **Policy Head**: $64 \to 4096$ candidate move logits (`POLICY_DIM = 4096`, $(from \ll 6) \mid to$ square transitions) computed in parallel with host-side numerically stable Softmax over legal moves.
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
  - **Policy-Guided Move Ordering**: Combines TT moves, MVV-LVA captures, SEE, Killer moves, Countermove heuristic, Continuation history, and policy head logits (`score += static_cast<int>(logit * 5000.0f)`).
  - **Policy-Guided LMR**: Reduces LMR depth discount for high-confidence policy moves (`logit > 1.2`), focusing tree search on natural master moves while pruning low-confidence candidate moves (`logit < -1.5`) earlier.
  - Iterative Deepening with dynamic widening Aspiration Windows (fail-low / fail-high widening)
  - **4-Slot Clustered Transposition Table**: Full 64-bit key verification with depth-, age-, and bound-aware replacement.
  - **History Gravity**: Dynamic positive bonus and negative malus decay formula (`score += bonus - (score * |bonus|)/16384`) capped at $\pm 16384$.
  - **Continuation History**: 1-ply piece-to-square continuation history (`conthist[piece][from][to]`) tracking responses to the opponent's previous move.
  - **Countermove Heuristic**: 2D lookup table (`countermove[piece][to]`) prioritizing specific refutations to the previous move on quiet cutoffs.
  - **ProbCut Pruning**: At depth $\ge 5$, probes shallow capture searches against an aggressive $\beta + 180$ threshold to prune cut subtrees early.
  - **Enhanced Quiescence Search**: Includes promotions and selective quiet checks only near the quiescence root to avoid checking-tree explosions.
  - **Logarithmic Late Move Reduction (LMR)**: Precomputed $64 \times 64$ logarithmic reduction table dynamically adjusted for cut-nodes and check evasions.
  - **Syzygy Endgame Tablebases**: Direct Fathom probing engine integration supporting 3-4-5-6 piece Syzygy WDL tables and instant root move probing.
  - Static Exchange Evaluation (SEE) pruning and move ordering
  - Null Move Pruning with verification
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
    ├── tt.h/.cpp           # Transposition Table implementation (2-slot cluster with aging)
    ├── syzygy_tb.h/.cpp    # Fathom Syzygy tablebase interface (WDL & root probing)
    ├── syzygy/             # Fathom Syzygy endgame tablebase probing engine
    │   ├── tbprobe.c/.h    # Syzygy WDL and DTZ probe routines
    │   ├── tbcore.c/.h     # Core decompression and bit operations
    │   └── tbconfig.h      # Syzygy probe configuration
    ├── search.h/.cpp       # PVS Search, ProbCut, History gravity, Continuation History, LMR
    ├── uci.h/.cpp          # Universal Chess Interface handler & diagnostic CLI
    ├── main.cpp            # Engine entry point
    └── cuda/
        ├── neural_gpu.h    # Host neural inference and GPU device manager
        └── neural_gpu.cu   # Custom CUDA forward pass kernels & weight loader
```

---

## Building and Cross-Platform Support

GiraffeGPU has a native CUDA build for Linux and a separate native CUDA build path for Windows. A portable Windows CPU fallback is also available.

---

### 1. Linux Compilation (Native Dual-GPU with CUDA)

#### Prerequisites
- **OS**: Linux x86_64 (e.g. Arch, Ubuntu, Debian, Fedora)
- **NVIDIA GPU Driver & CUDA Toolkit**: `nvcc` 11.0+ (tested on CUDA 13.4, SM 8.6 & SM 12.0)
- **Compiler**: C++17 compliant compiler (`g++` 8+ or `clang++` 7+)
- **Make** or **CMake** (3.18+)

#### Build with Make (Recommended on Linux)
```bash
git clone https://github.com/tedsummers/GiraffeGPU.git
cd GiraffeGPU
make -j$(nproc)
```
The compiled binary `./GiraffeGPU` will be generated in the project root.

#### Build with CMake on Linux
```bash
mkdir -p build && cd build
cmake ..
cmake --build . -j$(nproc)
```
The executable will be located at `./build/GiraffeGPU`.

---

### 2. Windows Compilation & Precompiled Binary

#### Portable Windows CPU Executable
A Windows 64-bit standalone executable (`GiraffeGPU.exe`) may be included in the repository:
- **Path**: `GiraffeGPU.exe`
- Built from the portable CPU fallback path and does not contain the Linux CUDA kernels.
- Works out-of-the-box in Windows Chess GUIs (Arena, Fritz, CuteChess, Bankchess, Lucas Chess).

#### Option A: Cross-Compile for Windows on Linux (via Make & Zig)
If you are on Linux and want to build the Windows `.exe` binary:
```bash
make windows
```
*(Uses Zig's built-in cross-compiler `zig c++ -target x86_64-windows-gnu` to produce `GiraffeGPU.exe`).*

This is the portable CPU fallback. For a Windows NVIDIA GPU build equivalent to
the Linux CUDA engine, use a Windows Visual Studio Developer PowerShell with
the NVIDIA CUDA Toolkit installed:

```powershell
Set-Location C:\path\to\GiraffeGPU
Set-ExecutionPolicy -Scope Process Bypass
.\build_windows_cuda.ps1
```

The script produces `GiraffeGPU_cuda.exe`, compiles `src/cuda/neural_gpu.cu`,
excludes the CPU-only `neural_win.cpp`, and requires `nvcc` and `cl.exe` on
`PATH`. Set `GIRAFFE_CUDA_ARCH` to match the GPU if needed, for example
`sm_89` or `sm_120`. This native Windows build cannot be produced by the Linux
Zig cross-compiler because it requires the Windows CUDA/MSVC toolchain.

#### Option B: Native Windows CPU Build with CMake and Visual Studio (MSVC)
On a Windows machine with Visual Studio 2019/2022 installed:
```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```
The binary will be generated at `build\Release\GiraffeGPU.exe`.

#### Option C: Native Windows CPU Build with MinGW-w64 or Clang
From a Windows Command Prompt or MSYS2 / PowerShell:
```powershell
g++ -O3 -std=c++17 -Isrc -pthread src/*.cpp -o GiraffeGPU.exe
```

---

## Usage

### Command Line Diagnostics & Benchmarks

#### Linux:
```bash
# List detected NVIDIA GPUs, SM architectures, and VRAM:
./GiraffeGPU gpu-info

# Run benchmark across standard test positions (depth 5):
./GiraffeGPU bench 5

# Evaluate position through Dual-Head Policy Network (top moves, probabilities, and logits):
./GiraffeGPU policy
./GiraffeGPU policy "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"
```

#### Windows:
```powershell
# List detected GPU / accelerator devices:
.\GiraffeGPU.exe gpu-info

# Run benchmark across standard test positions:
.\GiraffeGPU.exe bench 5

# Inspect Policy Network candidate move distribution:
.\GiraffeGPU.exe policy
```

---

### UCI Protocol Mode
Launch `./GiraffeGPU` (Linux) or `.\GiraffeGPU.exe` (Windows) directly:
```bash
./GiraffeGPU
```

#### Supported UCI Commands:
- `uci` - Returns engine identity and available configuration options.
- `isready` - Confirms CUDA initialization on the selected GPU device.
- `setoption name Device_GPU value <id>` - Selects GPU: `-1` (default) for **Both GPUs in parallel**, or `0` for GPU 0 only (RTX 3060), or `1` for GPU 1 only (RTX 5050).
- `setoption name Threads value <num>` - Sets search worker threads (default `2` to saturate both GPUs concurrently).
- `setoption name Hash value <MB>` - Sets transposition table size in MB (e.g., `256` or `1024`).
- `setoption name BatchSize value <size>` - Sets maximum GPU evaluation batch size.
- `setoption name PolicyNet value <true/false>` - Enables or disables Dual-Head Policy Network move ordering and LMR guidance (default: `true`).
- `setoption name AspirationWindow value <cp>` - Sets the initial aspiration-window width (default: `32`).
- `setoption name NullMoveReduction value <plies>` - Sets the base null-move reduction (default: `3`).
- `setoption name LMRMovesThreshold value <moves>` - Number of ordered moves searched before LMR begins (default: `3`).
- `setoption name LMRDepthThreshold value <depth>` - Minimum depth at which LMR is enabled (default: `3`).
- `setoption name ProbCutMargin value <cp>` - Extra margin used by capture ProbCut (default: `180`).
- `setoption name SyzygyPath value <path>` - Path to Syzygy 3-4-5-6 piece tablebases (WDL and DTZ files, separated by semicolons or colons).
- `setoption name Ponder value <true/false>` - Enables or disables pondering on opponent's time (default: `true`).
- `setoption name WeightsFile value <path>` - Loads custom Giraffe network weights (supports both `GIRAFDUAL` and legacy `GIRAFGPU` formats).
- `ucinewgame` - Resets transposition table and board state for a new game.
- `position startpos [moves e2e4 ...]` - Sets up the board position.
- `position fen <fen> [moves ...]` - Sets up position from FEN.
- `go [depth <d>] [movetime <ms>] [wtime <ms> btime <ms> winc <ms> binc <ms>]` - Starts normal search.
- `go ponder [wtime <ms> btime <ms> ...]` - Starts pondering on opponent's clock.
- `ponderhit` - Signals that opponent played the predicted ponder move; seamlessly transitions to normal search.
- `stop` - Stops search and returns best move immediately.
- `eval` - Evaluates current position on the GPU; reports centipawns and top policy moves with win percentages.
- `policy [fen]` - Displays ranked legal move probability distribution and raw logits from the policy head.
- `perft <depth>` - Performs move generation test.
- `d` - Prints formatted ASCII board representation.
- `quit` - Exits the engine cleanly.

---

## Testing & Playing Strength Improvement Framework

GiraffeGPU includes a complete automated testing, benchmarking, and training suite in the `testing/` directory to measure and systematically improve engine playing strength.

### 1. Tactical & Positional EPD Benchmarks
Evaluates solve rate, depth reached, nodes, and time on standardized benchmark suites (Bratko-Kopec Test, Win At Chess, sharp tactical suites):

```bash
# Run Bratko-Kopec test suite (24 benchmark positions)
make test-epd
# or:
python3 testing/run_tests.py --suite bk --movetime 1000

# Run sharp tactical problem suite
make test-tactics

# Directly compare a candidate build against a baseline build:
python3 testing/run_tests.py --suite bk --engine ./GiraffeGPU --compare ./GiraffeGPU_baseline
```

### 2. Match Gauntlet & SPRT (Sequential Probability Ratio Test)
Runs automated head-to-head matches between two engine versions or against external reference engines (Stockfish, TacTurn, etc.) with Wald's SPRT statistical stopping conditions:

```bash
# Run a 20-game match with opening book and alternating colors:
python3 testing/run_tests.py --match --engine ./GiraffeGPU --engine2 /path/to/opponent --games 20 --tc 5+0.1

# Run SPRT test (stops early as soon as Elo gain/regression is statistically proven at 95% confidence):
python3 testing/run_tests.py --sprt --engine ./GiraffeGPU_candidate --engine2 ./GiraffeGPU_base --elo0 0.0 --elo1 10.0

# Run direct A/B match isolating the Dual-Head Policy Network (PolicyNet On vs Off):
python3 testing/run_tests.py --compare-policy --games 20 --tc 5+0.1

# SPRT statistical validation for the Policy Network:
python3 testing/run_tests.py --compare-policy --sprt --elo0 0.0 --elo1 15.0 --games 100 --tc 3+0.05
```
- Real-time scoreboard with **Elo difference**, **95% confidence error margins ($\pm$)**, **Likelihood of Superiority (LOS %)**, and **Log-Likelihood Ratio (LLR)**.
- Supports custom UCI engine parameters via `--engine1-options` and `--engine2-options` (e.g. `PolicyNet=true`, `WeightsFile=candidate.net`).
- Saves full game transcripts to `testing/results/match_<timestamp>.pgn`.

### 3. Self-Play & Reinforcement Learning Data Generator
Generates self-play games with randomized opening plies (as in Matthew Lai's Giraffe paper) to produce datasets for TD-learning and neural fine-tuning:

```bash
# Generate 50 self-play games on dual GPUs:
python3 testing/run_tests.py --selfplay --games 50 --movetime 100
# or quick run:
make test-selfplay
```
- Saves position FENs, candidate moves, evaluations, search depths, and game outcomes directly to `testing/results/selfplay_dataset.jsonl`.

### 4. Dual-Head Neural Weight Trainer (`train_weights.py`)
Trains and fine-tunes both the Value Head (position evaluation) and the Policy Head (move selection logits) on cumulative self-play datasets:

```bash
# Fine-tune the value output layer on self-play data:
python3 testing/run_tests.py --train --epochs 5

# Or invoke the trainer directly with custom weight paths:
python3 testing/train_weights.py --data testing/results/selfplay_dataset.jsonl \
  --in-weights eval.net --out-weights testing/results/candidate.net \
  --epochs 5 --lr 0.00025 --max-output-delta 0.02 --policy-lr 0
```
- Uses FEN-aware, white-perspective targets blended from final game outcomes and normalized search evaluations.
- Reports an 80/20 validation split to expose overfitting before match testing.
- Updates only the value output layer by default; the policy head is frozen unless explicitly enabled.
- Limits output-layer drift per cycle and preserves compatibility with legacy `GIRAFGPU` weight files.

### 5. SPSA Automated Search Parameter Tuner
Uses Simultaneous Perturbation Stochastic Approximation to optimize search and pruning parameters (null move reduction, LMR thresholds, aspiration window margins, SEE margins):

```bash
    python3 testing/run_tests.py --spsa --iters 20 --spsa-games 8 --tc 3+0.05

The SPSA tuner runs real `cutechess-cli` matches between the plus and minus
parameter vectors and records every iteration in
`testing/results/spsa_history.jsonl`. It automatically saves the best observed
search vector to `testing/results/best_search_params.json`, writes its UCI
options to `best_search_options.txt`, and trains
`testing/results/candidate.net` from the self-play dataset. Pass
`--no-train-weights` to skip neural training. It requires `cutechess-cli` to be
installed and a CUDA-capable engine host for meaningful strength results.
```

### 6. 5-Day Autonomous Self-Improvement System (`autonomous_5day.py`)

GiraffeGPU includes a fully automated, 120-hour (5-day) self-improvement daemon that runs 24/7 across both NVIDIA GPUs (RTX 3060 and RTX 5050), progressively training network weights and optimizing search heuristics without requiring human intervention.

#### The 4-Stage Improvement Cycle

```mermaid
flowchart TD
    A["Dual-GPU Self-Play\n(Generate ~15,000 games/day with opening randomization)"] --> B["Data Extraction & Replay Buffer\n(Extract 363-dim features + TD(λ) game outcomes)"]
    B --> C["Neural Network Retraining\n(Blended FEN-aware value targets)"]
    C --> D["Automated SPRT Gauntlet\n(Candidate vs Baseline: 100+ games at 3+0.05)"]
    D -- "PASS (Elo gain confirmed at 95% CI)" --> E["Promote Candidate to Master\n(Update eval.net & recompile binaries)"]
    D -- "FAIL (No statistical gain)" --> F["Revert & Perturb Parameters\n(SPSA Search Tuning)"]
    E --> A
    F --> A
```

#### 5-Day Progression Schedule

| Day | Focus Area | What the Daemon Executes | Expected Elo Gain |
| :--- | :--- | :--- | :--- |
| **Day 1** | **Search Parameter Optimization (SPSA)** | Automated tuning of search heuristics using `tune_spsa.py`: Late Move Reductions (LMR), Null-Move $R$ margins, Aspiration Window size, and Killer/History table weights. | **+40 to +70 Elo** |
| **Day 2** | **Tactical Distillation & Opening Tuning** | Generate games against reference engines to train the 363-dim network to eliminate opening blunders and tactical oversights. | **+60 to +100 Elo** |
| **Day 3** | **Large-Scale Dual-GPU Self-Play** | Continuous self-play using both GPUs (~25,000 games) starting from 6–10 ply randomized positions from `testing/books/openings.epd`. | **+50 to +80 Elo** |
| **Day 4** | **FEN-Aware Value Training** | Train the value output layer using blended white-perspective search and game-result targets, with a held-out validation split. | **Measured by SPRT** |
| **Day 5** | **Validation & Packaging** | Run the SPRT gauntlet and EPD benchmarks. Linux is rebuilt locally; the native Windows CUDA binary is built on Windows with `build_windows_cuda.ps1`. | **Measured by SPRT** |
| **Total** | | | **No guaranteed Elo gain** |

#### Commands to Run and Monitor

```bash
# 1. Launch in background as a daemon (starts 120-hour continuous run):
make autonomous-daemon
# or:
python3 testing/autonomous_5day.py --daemon

# 2. Check live status anytime (elapsed time, current day, games, promoted networks):
make autonomous-status
# or:
python3 testing/autonomous_5day.py --status

# 3. Follow real-time log output:
tail -f testing/results/autonomous.log

# 4. Stop the daemon gracefully anytime (finishes current cycle cleanly):
python3 testing/autonomous_5day.py --stop
```

#### Fault-Tolerance & State Persistence
- **Resumable**: State is continuously persisted to `testing/results/autonomous_state.json`. If the machine restarts or the process is stopped, running `--daemon` or `--start` will automatically resume exactly where it left off.
- **Auto-Promotion**: Whenever a candidate network wins an SPRT match at 95% confidence, it is deployed to `eval.net`. The autonomous Linux host can rebuild the portable Windows CPU fallback; the native Windows CUDA executable must be rebuilt on Windows with `build_windows_cuda.ps1`.

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

## Authors & Credits

- **Engine Implementation**: Written by **Gemini 3.8 Flash** using directions from **Ted Summers**; bug-checked and debugged by **GPT 5.6 Luna** with assistance from **Ted Summers**.
- **Theoretical Architecture**: Based on the research and neural architecture designed by Matthew Lai (*Giraffe: Using Deep Reinforcement Learning to Play Chess*, Imperial College London, 2015).

---

## References

1. Matthew Lai, *"Giraffe: Using Deep Reinforcement Learning to Play Chess"*, MSc Thesis, Imperial College London, 2015. [arXiv:1509.01549](http://arxiv.org/abs/1509.01549)
2. Chessprogramming Wiki: [Giraffe](https://chessprogramming.org/Giraffe)
3. Giraffe GitHub Repository: [ianfab/Giraffe](https://github.com/ianfab/Giraffe)

---

## License

This project is licensed under the [MIT License](LICENSE).
