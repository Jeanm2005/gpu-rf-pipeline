include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
# Match the parent project's runtime on Windows/MSVC; a no-op everywhere else.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(sigflowc_unit_tests
  unit/dce_test.cpp
  unit/diagnostics_test.cpp
  unit/ir_builder_test.cpp
  unit/lexer_test.cpp
  unit/parser_test.cpp
  unit/schedule_test.cpp
  unit/sema_test.cpp
)
target_link_libraries(sigflowc_unit_tests PRIVATE GTest::gtest_main sigflow_core)
gtest_discover_tests(sigflowc_unit_tests)

# Golden tests: one per tests/golden/*.sf, comparing sigflowc's output and diagnostics byte for
# byte against tests/golden/<name>.expected. See cmake/SigflowGoldenTest.cmake and
# tests/golden/README.md.
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/SigflowGoldenTest.cmake)
add_sigflowc_golden_tests(${CMAKE_CURRENT_SOURCE_DIR}/golden)
