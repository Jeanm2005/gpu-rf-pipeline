cmake_minimum_required(VERSION 3.24)
project(sigflowc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Every compiler pass, linked by both the sigflowc executable and the unit tests below.
add_library(sigflow_core STATIC
  src/ast.cpp
  src/backend_cuda.cpp
  src/backend_script.cpp
  src/dce.cpp
  src/diagnostics.cpp
  src/driver.cpp
  src/fuse.cpp
  src/ir_builder.cpp
  src/json.cpp
  src/lexer.cpp
  src/liveness.cpp
  src/parser.cpp
  src/schedule.cpp
  src/sema.cpp
  src/source.cpp
)
target_include_directories(sigflow_core PUBLIC include)

add_executable(sigflowc src/main.cpp)
target_link_libraries(sigflowc PRIVATE sigflow_core)

# libFuzzer target over the whole pipeline (fuzz/fuzz_sigflow.cpp). Needs Clang; the core
# library is instrumented too, so coverage guides the fuzzer through every pass.
option(SIGFLOWC_FUZZ "Build the fuzz_sigflow libFuzzer target (Clang only)" OFF)
if(SIGFLOWC_FUZZ)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "SIGFLOWC_FUZZ needs Clang (-DCMAKE_CXX_COMPILER=clang++)")
  endif()
  target_compile_options(sigflow_core PUBLIC -fsanitize=fuzzer-no-link,address,undefined
                                             -fno-sanitize-recover=undefined)
  target_link_options(sigflow_core PUBLIC -fsanitize=address,undefined)
  add_executable(fuzz_sigflow fuzz/fuzz_sigflow.cpp)
  target_link_libraries(fuzz_sigflow PRIVATE sigflow_core)
  target_link_options(fuzz_sigflow PRIVATE -fsanitize=fuzzer)
endif()

option(SIGFLOWC_BUILD_TESTS "Build sigflowc unit and golden tests" ON)

if(SIGFLOWC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
