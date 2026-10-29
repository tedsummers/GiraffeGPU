cmake_minimum_required(VERSION 3.18)
project(GiraffeGPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

file(GLOB CPP_SOURCES src/*.cpp)

if(WIN32)
    # Native Windows Build
    add_executable(GiraffeGPU ${CPP_SOURCES})
    target_include_directories(GiraffeGPU PRIVATE src)
    target_link_libraries(GiraffeGPU PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(GiraffeGPU PRIVATE /O2 /W3)
    else()
        target_compile_options(GiraffeGPU PRIVATE -O3 -Wall -Wextra)
    endif()
else()
    # Linux build with native NVIDIA CUDA support
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 86 120)

    find_package(CUDAToolkit REQUIRED)
    file(GLOB CU_SOURCES src/cuda/*.cu)

    add_executable(GiraffeGPU ${CPP_SOURCES} ${CU_SOURCES})
    target_include_directories(GiraffeGPU PRIVATE src ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(GiraffeGPU PRIVATE CUDA::cudart Threads::Threads)
    target_compile_options(GiraffeGPU PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
endif()
