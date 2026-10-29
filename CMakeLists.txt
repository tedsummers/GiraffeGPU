cmake_minimum_required(VERSION 3.18)
project(GiraffeGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86 120)

find_package(CUDAToolkit REQUIRED)

include_directories(src ${CUDAToolkit_INCLUDE_DIRS})

file(GLOB SOURCES
    src/*.cpp
    src/cuda/*.cu
)

add_executable(GiraffeGPU ${SOURCES})

target_link_libraries(GiraffeGPU PRIVATE
    CUDA::cudart
    pthread
)

target_compile_options(GiraffeGPU PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
