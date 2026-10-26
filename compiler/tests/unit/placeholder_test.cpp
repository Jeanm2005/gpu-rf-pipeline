#include <gtest/gtest.h>

// Proves the GoogleTest build wiring works before there is any real compiler source to test.
// Delete this once a real unit test (e.g. lexer_test.cpp) exists.
TEST(Harness, GoogleTestIsWired)
{
    EXPECT_TRUE(true);
}
