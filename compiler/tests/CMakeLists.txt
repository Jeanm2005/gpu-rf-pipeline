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
  unit/diagnostics_test.cpp
)
target_link_libraries(sigflowc_unit_tests PRIVATE GTest::gtest_main sigflow_core)
gtest_discover_tests(sigflowc_unit_tests)

# Golden tests: compare a command's stdout, byte for byte, against tests/golden/<name>.expected.
# See cmake/SigflowGoldenTest.cmake and tests/golden/README.md.
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/SigflowGoldenTest.cmake)

# Self-test for the mechanism above, run with `cmake -E echo` standing in for sigflowc, which
# does not exist yet (phase A, see ../../docs/ROADMAP.md). Delete once the first real golden
# test (e.g. an empty-program or lexer case) is registered.
add_golden_test(harness_smoke
  COMMAND ${CMAKE_COMMAND} -E echo "hello"
  EXPECTED ${CMAKE_CURRENT_SOURCE_DIR}/golden/harness_smoke.expected
)
