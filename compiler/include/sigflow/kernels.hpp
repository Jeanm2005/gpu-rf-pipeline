#pragma once

// Which rfgpu kernel variant each backend uses per operation (--fir-impl, --xcorr-impl). The
// names are rfgpu's --impl names; naive is the default, as in rfgpu.

#include <string>
#include <vector>

namespace sigflow {

struct KernelChoice {
    std::string fir = "naive";    // naive | tiled | tiled_const
    std::string xcorr = "naive";  // naive | tiled | fft
};

inline const std::vector<std::string>& fir_impl_names()
{
    static const std::vector<std::string> names = {"naive", "tiled", "tiled_const"};
    return names;
}

inline const std::vector<std::string>& xcorr_impl_names()
{
    static const std::vector<std::string> names = {"naive", "tiled", "fft"};
    return names;
}

}  // namespace sigflow
