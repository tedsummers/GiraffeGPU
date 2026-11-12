CUDA_PATH ?= $(shell if [ -d /opt/cuda ]; then echo /opt/cuda; elif [ -d /usr/local/cuda ]; then echo /usr/local/cuda; else echo /opt/cuda; fi)
NVCC ?= $(CUDA_PATH)/bin/nvcc
CXX ?= g++

TARGET = GiraffeGPU

SRC_DIR = src
OBJ_DIR = obj

CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
C_SRCS   = $(SRC_DIR)/syzygy/tbprobe.c
CU_SRCS  = $(wildcard $(SRC_DIR)/cuda/*.cu)

CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
C_OBJS   = $(patsubst $(SRC_DIR)/syzygy/%.c, $(OBJ_DIR)/syzygy/%.o, $(C_SRCS))
CU_OBJS  = $(patsubst $(SRC_DIR)/cuda/%.cu, $(OBJ_DIR)/cuda/%.o, $(CU_SRCS))

CFLAGS   = -O3 -march=native -MMD -MP -Wall -Wextra -pthread -I$(SRC_DIR)
CXXFLAGS = -O3 -march=native -MMD -MP -std=c++17 -Wall -Wextra -pthread -I$(SRC_DIR) -I$(CUDA_PATH)/include
NVCCFLAGS = -O3 -std=c++17 -I$(SRC_DIR) -I$(CUDA_PATH)/include \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_120,code=sm_120 \
            -gencode arch=compute_86,code=compute_86 \
            --compiler-options "-O3 -march=native -Wall"

LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lpthread

all: $(TARGET)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)
	mkdir -p $(OBJ_DIR)/cuda
	mkdir -p $(OBJ_DIR)/syzygy

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/syzygy/%.o: $(SRC_DIR)/syzygy/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/cuda/%.o: $(SRC_DIR)/cuda/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(CPP_OBJS) $(C_OBJS) $(CU_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "============================================="
	@echo "Build successful: ./$(TARGET)"
	@echo "============================================="

clean:
	rm -rf $(OBJ_DIR) $(TARGET) $(WIN_TARGET)

ZIG ?= $(shell which zig 2>/dev/null || echo /home/linuxbrew/.linuxbrew/bin/zig)
WIN_TARGET = GiraffeGPU.exe

windows: $(OBJ_DIR)
	@echo "Building Windows 64-bit standalone executable ($(WIN_TARGET))..."
	$(ZIG) cc -target x86_64-windows-gnu -O3 -march=x86_64_v3 -I$(SRC_DIR) -pthread -c $(C_SRCS) -o $(OBJ_DIR)/tbprobe_win.o
	$(ZIG) c++ -target x86_64-windows-gnu -O3 -march=x86_64_v3 -std=c++17 -I$(SRC_DIR) -pthread $(CPP_SRCS) $(OBJ_DIR)/tbprobe_win.o -o $(WIN_TARGET)
	@echo "============================================="
	@echo "Build successful: ./$(WIN_TARGET)"
	@echo "============================================="

# Native Windows GPU build. Run build_windows_cuda.ps1 from a Windows
# Developer PowerShell with CUDA nvcc and MSVC available. The existing
# `windows` target remains the portable CPU fallback cross-build.
windows-cuda:
	powershell -ExecutionPolicy Bypass -File .\\build_windows_cuda.ps1

test-epd:
	python3 testing/run_tests.py --suite bk --movetime 1000

test-tactics:
	python3 testing/run_tests.py --suite tactics --movetime 500

test-selfplay:
	python3 testing/run_tests.py --selfplay --games 5 --movetime 50

train:
	python3 testing/train_weights.py --data testing/results/selfplay_dataset.jsonl --in-weights eval.net --out-weights eval.net --epochs 10 --lr 0.0005 --policy-lr 0.001 --max-output-delta 0.05

autonomous-5day:
	python3 testing/autonomous_5day.py --start

autonomous-daemon:
	python3 testing/autonomous_5day.py --daemon

autonomous-status:
	python3 testing/autonomous_5day.py --status

phase2-5day:
	python3 testing/autonomous_phase2.py --start

phase2-daemon:
	python3 testing/autonomous_phase2.py --daemon

phase2-status:
	python3 testing/autonomous_phase2.py --status

view-match:
	python3 testing/view_match.py

watch-match:
	python3 testing/view_match.py --watch

-include $(CPP_OBJS:.o=.d) $(C_OBJS:.o=.d)

.PHONY: all clean windows windows-cuda test-epd test-tactics test-selfplay train autonomous-5day autonomous-daemon autonomous-status phase2-5day phase2-daemon phase2-status view-match watch-match
