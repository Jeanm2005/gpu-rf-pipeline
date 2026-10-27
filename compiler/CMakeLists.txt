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
  src/dce.cpp
  src/diagnostics.cpp
  src/driver.cpp
  src/ir_builder.cpp
  src/lexer.cpp
  src/parser.cpp
  src/schedule.cpp
  src/sema.cpp
  src/source.cpp
)
target_include_directories(sigflow_core PUBLIC include)

add_executable(sigflowc src/main.cpp)
target_link_libraries(sigflowc PRIVATE sigflow_core)

option(SIGFLOWC_BUILD_TESTS "Build sigflowc unit and golden tests" ON)

if(SIGFLOWC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
