cmake_minimum_required(VERSION 3.24)
project(sigflowc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Grows to hold every compiler pass (diagnostics and lexer now; parser, sema, IR builder,
# scheduler, ... later), linked by both the sigflowc executable (once src/main.cpp exists) and
# the unit tests below.
add_library(sigflow_core STATIC
  src/diagnostics.cpp
  src/lexer.cpp
  src/source.cpp
)
target_include_directories(sigflow_core PUBLIC include)

# No sigflowc executable yet: src/main.cpp lands once the token, AST, and IR data structures are
# designed (see ../docs/ROADMAP.md, phase A). This file wires up the test infrastructure first
# so that work can start test-first.
option(SIGFLOWC_BUILD_TESTS "Build sigflowc unit and golden tests" ON)

if(SIGFLOWC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
