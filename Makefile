NVCC ?= /opt/cuda/bin/nvcc
CXX ?= g++

TARGET = GiraffeGPU

SRC_DIR = src
OBJ_DIR = obj

CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS  = $(wildcard $(SRC_DIR)/cuda/*.cu)

CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
CU_OBJS  = $(patsubst $(SRC_DIR)/cuda/%.cu, $(OBJ_DIR)/cuda/%.o, $(CU_SRCS))

CXXFLAGS = -O3 -march=native -std=c++17 -Wall -Wextra -pthread -I$(SRC_DIR) -I/opt/cuda/include
NVCCFLAGS = -O3 -std=c++17 -I$(SRC_DIR) -I/opt/cuda/include \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_120,code=sm_120 \
            -gencode arch=compute_86,code=compute_86 \
            --compiler-options "-O3 -march=native -Wall"

LDFLAGS = -L/opt/cuda/lib64 -lcudart -lpthread

all: $(TARGET)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)
	mkdir -p $(OBJ_DIR)/cuda

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/cuda/%.o: $(SRC_DIR)/cuda/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "============================================="
	@echo "Build successful: ./$(TARGET)"
	@echo "============================================="

clean:
	rm -rf $(OBJ_DIR) $(TARGET) $(WIN_TARGET)

ZIG ?= $(shell which zig 2>/dev/null || echo /home/linuxbrew/.linuxbrew/bin/zig)
WIN_TARGET = GiraffeGPU.exe

windows:
	@echo "Building Windows 64-bit standalone executable ($(WIN_TARGET))..."
	$(ZIG) c++ -target x86_64-windows-gnu -O3 -std=c++17 -I$(SRC_DIR) -pthread $(CPP_SRCS) -o $(WIN_TARGET)
	@echo "============================================="
	@echo "Build successful: ./$(WIN_TARGET)"
	@echo "============================================="

test-epd:
	python3 testing/run_tests.py --suite bk --movetime 1000

test-tactics:
	python3 testing/run_tests.py --suite tactics --movetime 500

test-selfplay:
	python3 testing/run_tests.py --selfplay --games 5 --movetime 50

.PHONY: all clean windows test-epd test-tactics test-selfplay
