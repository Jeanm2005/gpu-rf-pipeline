#include "pipeline_runtime.hpp"

#include "check.hpp"
#include "io.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace sigflow {

namespace {

std::size_t gpu_memory_in_use()
{
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return total_bytes - free_bytes;
}

int int_flag(int argc, char** argv, const char* name, int fallback, int minimum)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            const int value = std::atoi(argv[i + 1]);
            if (value < minimum) {
                throw std::runtime_error(std::string(name) + " must be at least "
                                         + std::to_string(minimum));
            }
            return value;
        }
    }
    return fallback;
}

void make_parent_directory(const std::string& path)
{
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

std::string json_counts(const std::map<std::string, int>& counts, int& total)
{
    std::string out = "{";
    total = 0;
    for (const auto& [kind, n] : counts) {
        out += (out.size() > 1 ? ", \"" : "\"") + kind + "\": " + std::to_string(n);
        total += n;
    }
    return out + "}";
}

}  // namespace

Runtime::Runtime(const char* program, bool fused, bool reuse, int argc, char** argv)
    : program_(program), fused_(fused), reuse_(reuse)
{
    config_.runs = int_flag(argc, argv, "--bench", 1, 1);
    config_.warmup = int_flag(argc, argv, "--warmup", 0, 0);
    CUDA_CHECK(cudaFree(nullptr));  // create the context now, so it is not counted below
    used_before_ = gpu_memory_in_use();
}

std::vector<cf32> Runtime::read_capture(const std::string& path) { return ::read_cf32(path); }

void Runtime::need(int slot, std::size_t bytes)
{
    const std::size_t index = static_cast<std::size_t>(slot);
    if (slot_bytes_.size() <= index) {
        slot_bytes_.resize(index + 1, 0);
    }
    slot_bytes_[index] = std::max(slot_bytes_[index], bytes);
}

void Runtime::allocate_slots()
{
    for (std::size_t bytes : slot_bytes_) {
        slots_.push_back(std::make_unique<DeviceBuffer<unsigned char>>(bytes));
    }
}

const float* Runtime::taps(const std::string& path, int& count)
{
    auto it = taps_.find(path);
    if (it == taps_.end()) {
        const std::vector<float> host = ::read_f32(path);
        if (host.empty()) {
            throw std::runtime_error(path + ": no taps");
        }
        auto buffer = std::make_unique<DeviceBuffer<float>>(host.size());
        buffer->upload(host.data(), host.size());
        taps_bytes_ += buffer->bytes();
        it = taps_.emplace(path, std::move(buffer)).first;
    }
    count = static_cast<int>(it->second->size());
    return it->second->get();
}

cufftHandle Runtime::fft_plan(int nfft, std::size_t num_frames)
{
    plans_.push_back(std::make_unique<FftPlan>(nfft, static_cast<int>(num_frames)));
    std::size_t work = 0;
    CUFFT_CHECK(cufftGetSize(plans_.back()->get(), &work));
    fft_work_bytes_ += work;
    return plans_.back()->get();
}

const XcorrFft& Runtime::xcorr_fft(std::size_t num_a, std::size_t num_b)
{
    xcorr_ffts_.push_back(
        std::make_unique<XcorrFft>(static_cast<int>(num_a), static_cast<int>(num_b)));
    fft_work_bytes_ += xcorr_ffts_.back()->bytes();
    return *xcorr_ffts_.back();
}

void Runtime::run(const std::function<void()>& upload, const std::function<void()>& compute,
                  const std::function<void()>& download)
{
    // Every allocation is done by now: setup ends here.
    CUDA_CHECK(cudaDeviceSynchronize());
    used_after_ = gpu_memory_in_use();

    Bench bench(config_);
    bench.run(upload,
              [&] {
                  counting_ = !counted_;
                  compute();
                  counting_ = false;
                  counted_ = true;
              },
              download);
    result_ = bench.result();
}

void Runtime::write_cf32(const std::string& path, const std::vector<cf32>& data)
{
    make_parent_directory(path);
    ::write_cf32(path, data);
}

void Runtime::write_f32(const std::string& path, const std::vector<float>& data)
{
    make_parent_directory(path);
    ::write_f32(path, data);
}

void Runtime::report() const
{
    std::size_t slots_total = 0;
    for (std::size_t bytes : slot_bytes_) {
        slots_total += bytes;
    }
    int launches = 0;
    const std::string counts = json_counts(counts_, launches);
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const BenchResult& r = result_;
    std::printf(
        "{\"program\": \"%s\", \"fused\": %s, \"reuse\": %s, \"gpu\": \"%s\", "
        "\"launches\": %d, \"launches_by_kind\": %s, "
        "\"slots\": %zu, \"slot_bytes\": %zu, \"taps_bytes\": %zu, \"fft_work_bytes\": %zu, "
        "\"allocated_bytes\": %zu, \"device_used_delta_bytes\": %zu, "
        "\"runs\": %d, \"warmup\": %d, "
        "\"h2d_ms\": %.6f, \"h2d_std_ms\": %.6f, \"kernel_ms\": %.6f, \"kernel_std_ms\": %.6f, "
        "\"d2h_ms\": %.6f, \"d2h_std_ms\": %.6f}\n",
        program_.c_str(), fused_ ? "true" : "false", reuse_ ? "true" : "false", prop.name,
        launches, counts.c_str(), slot_bytes_.size(), slots_total, taps_bytes_,
        fft_work_bytes_, slots_total + taps_bytes_ + fft_work_bytes_,
        used_after_ > used_before_ ? used_after_ - used_before_ : 0, config_.runs,
        config_.warmup, r.h2d.mean_ms, r.h2d.std_ms, r.kernel.mean_ms, r.kernel.std_ms,
        r.d2h.mean_ms, r.d2h.std_ms);
}

}  // namespace sigflow
