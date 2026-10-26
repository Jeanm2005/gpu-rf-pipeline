cmake_minimum_required(VERSION 3.24)
project(sigflowc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# No sigflowc target yet: src/main.cpp and the compiler passes land once the token, AST, and IR
# data structures are designed (see ../docs/ROADMAP.md, phase A). This file wires up the test
# infrastructure first so that work can start test-first.
option(SIGFLOWC_BUILD_TESTS "Build sigflowc unit and golden tests" ON)

if(SIGFLOWC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
