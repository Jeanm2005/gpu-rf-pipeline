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

# Every operation, its kernels, file I/O, and the timing harness. Linked by rfgpu and by the
# programs sigflowc generates with --emit=cuda (see compiler/README.md), which call the
# device-pointer entry points in src/device_ops.hpp and src/pipeline_runtime.hpp.
add_library(rfgpu_ops STATIC
  src/io.cpp
  src/bench.cpp
  src/pipeline_runtime.cpp
  src/spectrum.cu
  src/magnitude.cu
  src/fir.cu
  src/fused.cu
  src/xcorr.cu
)
target_include_directories(rfgpu_ops PUBLIC src)
target_link_libraries(rfgpu_ops PUBLIC CUDA::cudart CUDA::cufft)
target_compile_options(rfgpu_ops PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)

add_executable(rfgpu src/main.cpp)
target_link_libraries(rfgpu PRIVATE rfgpu_ops)
target_compile_options(rfgpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
