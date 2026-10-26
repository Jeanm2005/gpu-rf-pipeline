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
  unit/placeholder_test.cpp
)
target_link_libraries(sigflowc_unit_tests PRIVATE GTest::gtest_main)
gtest_discover_tests(sigflowc_unit_tests)
