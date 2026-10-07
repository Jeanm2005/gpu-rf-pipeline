#pragma once

// File I/O for the two raw formats the pipeline uses. Both are little-endian float32 with no
// header, read and written directly, so this code assumes a little-endian host (x86-64 and
// the ARM cores in Jetson boards both are). Every function throws std::runtime_error with the
// file path on failure.

#include "types.hpp"

#include <string>
#include <vector>

// .cf32: interleaved [I0, Q0, I1, Q1, ...]. Fails if the file holds an odd number of floats.
std::vector<cf32> read_cf32(const std::string& path);
void write_cf32(const std::string& path, const std::vector<cf32>& samples);

// .f32: plain float32 array (filter taps).
std::vector<float> read_f32(const std::string& path);
