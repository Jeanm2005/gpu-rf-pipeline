cmake_minimum_required(VERSION 3.24)

# Compile device code for the GPU in this machine unless the caller picks architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(gpu_rf_pipeline LANGUAGES CXX CUDA)

# Timing is only meaningful in Release, so make it the default.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

add_executable(rfgpu
  src/main.cpp
  src/io.cpp
  src/bench.cpp
  src/spectrum.cu
  src/fir.cu
  src/xcorr.cu
)
target_link_libraries(rfgpu PRIVATE CUDA::cudart CUDA::cufft)
target_compile_options(rfgpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
