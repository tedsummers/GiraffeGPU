cmake_minimum_required(VERSION 3.18)
project(GiraffeGPU LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

option(GIRAFFE_ENABLE_CUDA "Build with native NVIDIA CUDA support if available" ON)

find_package(Threads REQUIRED)

file(GLOB CPP_SOURCES src/*.cpp)
set(C_SOURCES src/syzygy/tbprobe.c)

include(CheckLanguage)
if(GIRAFFE_ENABLE_CUDA)
    check_language(CUDA)
endif()

if(WIN32 AND NOT CMAKE_CUDA_COMPILER)
    # Portable Windows CPU Build (uses src/neural_win.cpp)
    add_executable(GiraffeGPU ${CPP_SOURCES} ${C_SOURCES})
    target_include_directories(GiraffeGPU PRIVATE src)
    target_link_libraries(GiraffeGPU PRIVATE Threads::Threads)
    target_compile_definitions(GiraffeGPU PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    if(MSVC)
        target_compile_options(GiraffeGPU PRIVATE /O2 /W3 /EHsc /utf-8)
    else()
        target_compile_options(GiraffeGPU PRIVATE -O3 -Wall -Wextra)
    endif()
else()
    # Native NVIDIA CUDA Build (Linux & Windows with CUDA Toolkit)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86 120)
    endif()

    find_package(CUDAToolkit REQUIRED)
    file(GLOB CU_SOURCES src/cuda/*.cu)
    list(FILTER CPP_SOURCES EXCLUDE REGEX "neural_win\\.cpp$")

    add_executable(GiraffeGPU ${CPP_SOURCES} ${C_SOURCES} ${CU_SOURCES})
    target_include_directories(GiraffeGPU PRIVATE src ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(GiraffeGPU PRIVATE CUDA::cudart Threads::Threads)
    if(WIN32)
        target_compile_definitions(GiraffeGPU PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()
    if(MSVC)
        target_compile_options(GiraffeGPU PRIVATE
            $<$<COMPILE_LANGUAGE:C>:/O2 /W3>
            $<$<COMPILE_LANGUAGE:CXX>:/O2 /W3 /EHsc /utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=/O2,/EHsc,/utf-8>
        )
    else()
        target_compile_options(GiraffeGPU PRIVATE
            $<$<COMPILE_LANGUAGE:C>:-O3 -march=native -Wall -Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )
    endif()
endif()
