#include "io.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

// Reads a whole file as an array of T. `what` names the element in error messages.
template <class T>
std::vector<T> read_array(const std::string& path, const char* what)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    const std::streamoff bytes = file.tellg();
    if (bytes < 0) {
        throw std::runtime_error("cannot read the size of " + path);
    }
    if (static_cast<std::uintmax_t>(bytes) % sizeof(T) != 0) {
        throw std::runtime_error(path + ": size " + std::to_string(bytes)
                                 + " bytes is not a whole number of " + what + " ("
                                 + std::to_string(sizeof(T)) + " bytes each)");
    }
    std::vector<T> data(static_cast<std::size_t>(bytes) / sizeof(T));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), bytes);
    if (!file) {
        throw std::runtime_error("failed while reading " + path);
    }
    return data;
}

}  // namespace

std::vector<cf32> read_cf32(const std::string& path)
{
    return read_array<cf32>(path, "complex float32 samples");
}

std::vector<float> read_f32(const std::string& path)
{
    return read_array<float>(path, "float32 values");
}

void write_cf32(const std::string& path, const std::vector<cf32>& samples)
{
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            throw std::runtime_error("cannot create directory " + parent.string() + ": "
                                     + ec.message());
        }
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("cannot open " + path + " for writing");
    }
    file.write(reinterpret_cast<const char*>(samples.data()),
               static_cast<std::streamsize>(samples.size() * sizeof(cf32)));
    file.close();
    if (!file) {
        throw std::runtime_error("failed while writing " + path);
    }
}
