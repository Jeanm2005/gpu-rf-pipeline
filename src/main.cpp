// rfgpu: command-line front end for the GPU operations.
//
//   rfgpu spectrum --in x.cf32 --out spec.cf32 [--nfft 1024]
//   rfgpu fir      --in x.cf32 --taps h.f32 --out y.cf32 [--impl naive]
//   rfgpu xcorr    --a a.cf32 --b b.cf32 --out r.cf32 [--impl naive]
//   rfgpu info
//
// Exit status: 0 on success, 1 on a runtime error (file, CUDA, cuFFT), 2 on a usage error.

#include "bench.hpp"
#include "check.hpp"
#include "io.hpp"
#include "ops.hpp"

#include <climits>
#include <cstdio>
#include <exception>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

template <class Fn>
std::string impl_names(const std::vector<Impl<Fn>>& impls, const char* separator)
{
    std::string names;
    for (const Impl<Fn>& impl : impls) {
        if (!names.empty()) {
            names += separator;
        }
        names += impl.name;
    }
    return names;
}

template <class Fn>
void print_impls(const char* op, const std::vector<Impl<Fn>>& impls)
{
    for (std::size_t i = 0; i < impls.size(); ++i) {
        std::fprintf(stderr, "  %-6s --impl %-12s %s%s\n", op, impls[i].name, impls[i].summary,
                     i == 0 ? " (default)" : "");
    }
}

void print_usage()
{
    std::fprintf(stderr,
        "usage:\n"
        "  rfgpu spectrum --in X.cf32 --out OUT.cf32 [--nfft N]\n"
        "  rfgpu fir      --in X.cf32 --taps H.f32 --out OUT.cf32 [--impl NAME]\n"
        "  rfgpu xcorr    --a A.cf32 --b B.cf32 --out OUT.cf32 [--impl NAME]\n"
        "  rfgpu info     print the GPU, CUDA versions, and implementations as JSON\n"
        "\n"
        "benchmark options (spectrum, fir, xcorr):\n"
        "  --bench N      time N runs with CUDA events and print the result as JSON\n"
        "  --warmup W     untimed runs before measuring (default 3, needs --bench)\n"
        "\n"
        "implementations:\n");
    print_impls("fir", fir_impls());
    print_impls("xcorr", xcorr_impls());
}

// Parsed "--flag value" pairs for one subcommand.
class Args {
public:
    Args(int argc, char** argv, const std::set<std::string>& allowed)
    {
        for (int i = 2; i < argc; i += 2) {
            const std::string flag = argv[i];
            if (allowed.count(flag) == 0) {
                throw UsageError("unknown option '" + flag + "' for " + argv[1]);
            }
            if (i + 1 >= argc) {
                throw UsageError("option " + flag + " needs a value");
            }
            if (!values_.emplace(flag, argv[i + 1]).second) {
                throw UsageError("option " + flag + " given twice");
            }
        }
    }

    bool has(const std::string& flag) const { return values_.count(flag) != 0; }

    const std::string& required(const std::string& flag) const
    {
        const auto it = values_.find(flag);
        if (it == values_.end()) {
            throw UsageError("missing required option " + flag);
        }
        return it->second;
    }

    std::string optional(const std::string& flag, const std::string& fallback) const
    {
        const auto it = values_.find(flag);
        return it == values_.end() ? fallback : it->second;
    }

    // Integer option, at least `min`.
    int integer(const std::string& flag, int fallback, int min) const
    {
        const auto it = values_.find(flag);
        if (it == values_.end()) {
            return fallback;
        }
        int value = 0;
        std::size_t used = 0;
        try {
            value = std::stoi(it->second, &used);
        } catch (const std::exception&) {
            used = 0;
        }
        if (used == 0 || used != it->second.size() || value < min) {
            throw UsageError("option " + flag + " needs an integer >= " + std::to_string(min)
                             + ", got '" + it->second + "'");
        }
        return value;
    }

private:
    std::map<std::string, std::string> values_;
};

template <class Fn>
const Impl<Fn>& find_impl(const std::vector<Impl<Fn>>& impls, const Args& args, const char* op)
{
    const std::string name = args.optional("--impl", impls.front().name);
    for (const Impl<Fn>& impl : impls) {
        if (name == impl.name) {
            return impl;
        }
    }
    throw UsageError("unknown --impl '" + name + "' for " + op + " (available: "
                     + impl_names(impls, ", ") + ")");
}

BenchConfig bench_config(const Args& args)
{
    BenchConfig config;
    if (args.has("--bench")) {
        config.runs = args.integer("--bench", 1, 1);
        config.warmup = args.integer("--warmup", 3, 0);
        if (config.runs < 10) {
            std::fprintf(stderr, "warning: --bench %d is below the 10 runs the measurement "
                                 "rules ask for\n", config.runs);
        }
    } else if (args.has("--warmup")) {
        throw UsageError("--warmup needs --bench");
    }
    return config;
}

// Kernels index with int, so every length has to fit in one.
void require_int_range(std::size_t count, const std::string& what)
{
    if (count > static_cast<std::size_t>(INT_MAX)) {
        throw std::runtime_error(what + " has " + std::to_string(count)
                                 + " elements, more than the supported " + std::to_string(INT_MAX));
    }
}

std::string gpu_name()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.name;
}

// Writes the result and reports: one line for a normal run, one JSON object for --bench.
void finish(const Args& args, const char* op, const char* impl, std::size_t num_in,
            const std::vector<cf32>& out, const Bench& bench)
{
    const std::string& path = args.required("--out");
    write_cf32(path, out);
    if (!args.has("--bench")) {
        std::printf("%s[%s]: %zu samples in, %zu out -> %s\n", op, impl, num_in, out.size(),
                    path.c_str());
        return;
    }
    const BenchResult& r = bench.result();
    std::printf("{\"op\": \"%s\", \"impl\": \"%s\", \"gpu\": \"%s\", \"num_in\": %zu, "
                "\"num_out\": %zu, \"runs\": %d, \"warmup\": %d, "
                "\"h2d_ms\": {\"mean\": %.6f, \"std\": %.6f}, "
                "\"kernel_ms\": {\"mean\": %.6f, \"std\": %.6f}, "
                "\"d2h_ms\": {\"mean\": %.6f, \"std\": %.6f}}\n",
                op, impl, gpu_name().c_str(), num_in, out.size(), bench.config().runs,
                bench.config().warmup, r.h2d.mean_ms, r.h2d.std_ms, r.kernel.mean_ms,
                r.kernel.std_ms, r.d2h.mean_ms, r.d2h.std_ms);
}

int run_spectrum(int argc, char** argv)
{
    const Args args(argc, argv, {"--in", "--out", "--nfft", "--bench", "--warmup"});
    args.required("--out");
    const int nfft = args.integer("--nfft", 1024, 1);
    Bench bench(bench_config(args));

    const std::vector<cf32> x = read_cf32(args.required("--in"));
    // The output is the input padded up to a whole number of frames.
    require_int_range(x.size() + static_cast<std::size_t>(nfft), "padded spectrum input");
    finish(args, "spectrum", "cufft", x.size(), spectrum_cufft(x, nfft, bench), bench);
    return 0;
}

int run_fir(int argc, char** argv)
{
    const Args args(argc, argv, {"--in", "--taps", "--out", "--impl", "--bench", "--warmup"});
    args.required("--out");
    const Impl<FirFn>& impl = find_impl(fir_impls(), args, "fir");
    Bench bench(bench_config(args));

    const std::vector<cf32> x = read_cf32(args.required("--in"));
    const std::vector<float> taps = read_f32(args.required("--taps"));
    if (taps.empty()) {
        throw std::runtime_error(args.required("--taps") + ": no taps");
    }
    require_int_range(x.size(), "fir input");
    require_int_range(taps.size(), "tap array");
    // An empty input has nothing to launch a kernel for.
    const std::vector<cf32> y = x.empty() ? std::vector<cf32>() : impl.fn(x, taps, bench);
    finish(args, "fir", impl.name, x.size(), y, bench);
    return 0;
}

int run_xcorr(int argc, char** argv)
{
    const Args args(argc, argv, {"--a", "--b", "--out", "--impl", "--bench", "--warmup"});
    args.required("--out");
    const Impl<XcorrFn>& impl = find_impl(xcorr_impls(), args, "xcorr");
    Bench bench(bench_config(args));

    const std::vector<cf32> a = read_cf32(args.required("--a"));
    const std::vector<cf32> b = read_cf32(args.required("--b"));
    if (a.empty() || b.empty()) {
        throw std::runtime_error("xcorr needs at least one sample in each input");
    }
    require_int_range(a.size() + b.size(), "xcorr output");
    finish(args, "xcorr", impl.name, a.size() + b.size(), impl.fn(a, b, bench), bench);
    return 0;
}

// CUDA packs versions as 1000 * major + 10 * minor.
std::string version_string(int packed)
{
    return std::to_string(packed / 1000) + "." + std::to_string((packed % 1000) / 10);
}

int run_info(int argc, char** argv)
{
    const Args args(argc, argv, {});
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int runtime = 0;
    int driver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));

    // "cuda_driver" is the newest CUDA version the installed driver supports, not the
    // driver's own version number (nvidia-smi prints that one).
    std::printf("{\"gpu\": \"%s\", \"compute_capability\": \"%d.%d\", \"memory_mb\": %zu, "
                "\"cuda_runtime\": \"%s\", \"cuda_driver\": \"%s\", "
                "\"impls\": {\"spectrum\": [\"cufft\"], \"fir\": [\"%s\"], \"xcorr\": [\"%s\"]}}\n",
                prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024),
                version_string(runtime).c_str(), version_string(driver).c_str(),
                impl_names(fir_impls(), "\", \"").c_str(),
                impl_names(xcorr_impls(), "\", \"").c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const std::string op = argc > 1 ? argv[1] : "";
        if (op == "spectrum") return run_spectrum(argc, argv);
        if (op == "fir") return run_fir(argc, argv);
        if (op == "xcorr") return run_xcorr(argc, argv);
        if (op == "info") return run_info(argc, argv);
        if (op == "--help" || op == "-h" || op == "help") {
            print_usage();
            return 0;
        }
        throw UsageError(op.empty() ? "missing subcommand" : "unknown subcommand '" + op + "'");
    } catch (const UsageError& e) {
        std::fprintf(stderr, "rfgpu: %s\n\n", e.what());
        print_usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "rfgpu: %s\n", e.what());
        return 1;
    }
}
