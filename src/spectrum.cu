// Framed FFT spectrum with cuFFT. No custom kernel here: cuFFT is the library baseline.

#include "check.hpp"
#include "device_buffer.hpp"
#include "ops.hpp"

static_assert(sizeof(cufftComplex) == sizeof(cf32), "cf32 must match cufftComplex");

namespace {

// A cuFFT plan, destroyed when it goes out of scope.
class FftPlan {
public:
    // `batch` one-dimensional complex-to-complex transforms of `nfft` points each, stored
    // back to back in one buffer.
    FftPlan(int nfft, int batch)
    {
        // The two nullptr "embed" arguments select the simple layout: frame f occupies
        // elements [f * nfft, (f + 1) * nfft). The stride and distance values are then
        // ignored by cuFFT, but are filled in with what that layout means.
        CUFFT_CHECK(cufftPlanMany(&plan_, 1, &nfft, nullptr, 1, nfft, nullptr, 1, nfft,
                                  CUFFT_C2C, batch));
    }
    ~FftPlan()
    {
        const cufftResult err = cufftDestroy(plan_);
        if (err != CUFFT_SUCCESS) {
            std::fprintf(stderr, "warning: cufftDestroy failed: %s\n", cufft_error_string(err));
        }
    }
    FftPlan(const FftPlan&) = delete;
    FftPlan& operator=(const FftPlan&) = delete;

    cufftHandle get() const { return plan_; }

private:
    cufftHandle plan_{};
};

}  // namespace

std::vector<cf32> spectrum_cufft(const std::vector<cf32>& x, int nfft, Bench& bench)
{
    const std::size_t frame = static_cast<std::size_t>(nfft);
    const std::size_t num_frames = (x.size() + frame - 1) / frame;
    std::vector<cf32> out(num_frames * frame);
    if (num_frames == 0) {
        return out;
    }

    // One-time setup, outside the timed region. The transform is out of place so that the
    // zero padding of the last frame is written once and survives every benchmark pass.
    DeviceBuffer<cf32> d_in(out.size());
    DeviceBuffer<cf32> d_out(out.size());
    d_in.zero();
    FftPlan plan(nfft, static_cast<int>(num_frames));

    bench.run(
        [&] { d_in.upload(x.data(), x.size()); },
        [&] {
            CUFFT_CHECK(cufftExecC2C(plan.get(), reinterpret_cast<cufftComplex*>(d_in.get()),
                                     reinterpret_cast<cufftComplex*>(d_out.get()),
                                     CUFFT_FORWARD));
        },
        [&] { d_out.download(out.data(), out.size()); });
    return out;
}
