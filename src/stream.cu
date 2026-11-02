// Streaming FIR over a ring of pinned buffers and CUDA streams (see stream.hpp).

#include "stream.hpp"

#include "check.hpp"
#include "device_buffer.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

// Page-locked host memory. Copies from it can run asynchronously and at full bus speed; copies
// from ordinary pageable memory are staged through a driver buffer instead. With `mapped`, the
// GPU can also read and write it in place through device_ptr() (zero-copy).
template <class T>
class PinnedBuffer {
public:
    PinnedBuffer(std::size_t count, bool mapped)
    {
        void* raw = nullptr;
        CUDA_CHECK(cudaHostAlloc(&raw, count * sizeof(T),
                                 mapped ? cudaHostAllocMapped : cudaHostAllocDefault));
        ptr_ = static_cast<T*>(raw);
        if (mapped) {
            void* device = nullptr;
            CUDA_CHECK(cudaHostGetDevicePointer(&device, raw, 0));
            device_ptr_ = static_cast<T*>(device);
        }
    }
    ~PinnedBuffer() { cudaFreeHost(ptr_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    T* get() const { return ptr_; }
    T* device_ptr() const { return device_ptr_; }

private:
    T* ptr_ = nullptr;
    T* device_ptr_ = nullptr;
};

class Stream {
public:
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class Event {
public:
    Event() { CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming)); }
    ~Event() { cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

// One slot of the ring, sized for history + chunk. With copies, the kernel works on device
// buffers; with zero-copy, it reads host_in and writes host_out directly, and the device
// buffers are empty; with registered memory, the host buffers are unused (one sample each).
// `start`, `count`, and `offset` place the chunk in flight: its outputs are
// y[start, start + count), at `offset` in the kernel's output.
struct Slot {
    Slot(std::size_t len, StreamMemory memory)
        : host_in(memory == StreamMemory::Registered ? 1 : len, memory == StreamMemory::ZeroCopy),
          host_out(memory == StreamMemory::Registered ? 1 : len,
                   memory == StreamMemory::ZeroCopy),
          dev_in(memory == StreamMemory::ZeroCopy ? 0 : len),
          dev_out(memory == StreamMemory::ZeroCopy ? 0 : len)
    {
    }
    PinnedBuffer<cf32> host_in;
    PinnedBuffer<cf32> host_out;
    DeviceBuffer<cf32> dev_in;
    DeviceBuffer<cf32> dev_out;
    Stream stream;
    Event done;
    std::size_t start = 0;
    std::size_t count = 0;
    std::size_t offset = 0;
    bool busy = false;
};

}  // namespace

// Page-locks an existing host array for as long as it lives (cudaHostRegister).
class HostRegistration {
public:
    HostRegistration(const void* data, std::size_t bytes)
    {
        if (data != nullptr && bytes > 0) {
            CUDA_CHECK(cudaHostRegister(const_cast<void*>(data), bytes, cudaHostRegisterDefault));
            data_ = data;
        }
    }
    ~HostRegistration()
    {
        if (data_ != nullptr) {
            cudaHostUnregister(const_cast<void*>(data_));
        }
    }
    HostRegistration(const HostRegistration&) = delete;
    HostRegistration& operator=(const HostRegistration&) = delete;

private:
    const void* data_ = nullptr;
};

bool parse_stream_memory(const std::string& name, StreamMemory& memory)
{
    if (name == "copy") {
        memory = StreamMemory::Copy;
    } else if (name == "registered") {
        memory = StreamMemory::Registered;
    } else if (name == "zero-copy") {
        memory = StreamMemory::ZeroCopy;
    } else {
        return false;
    }
    return true;
}

std::vector<cf32> fir_stream(const std::vector<cf32>& x, const std::vector<float>& taps,
                             const StreamConfig& config, StreamTiming& timing)
{
    if (config.chunk < 1 || config.depth < 1) {
        throw std::runtime_error("stream: chunk and depth must be at least 1");
    }
    const std::size_t num_samples = x.size();
    const std::size_t halo = taps.size() - 1;
    const std::size_t chunk = static_cast<std::size_t>(config.chunk);
    const int num_taps = static_cast<int>(taps.size());
    std::vector<cf32> y(num_samples);

    // Setup, outside the clock: the ring, and the taps on the device (and in constant memory
    // for the variants that read them there, once for every stream).
    std::vector<std::unique_ptr<Slot>> slots;
    for (int s = 0; s < config.depth; ++s) {
        slots.push_back(std::make_unique<Slot>(halo + chunk, config.memory));
    }
    const bool registered = config.memory == StreamMemory::Registered;
    const HostRegistration x_pinned(registered ? x.data() : nullptr, num_samples * sizeof(cf32));
    const HostRegistration y_pinned(registered ? y.data() : nullptr, num_samples * sizeof(cf32));
    DeviceBuffer<float> d_taps(taps.size());
    d_taps.upload(taps.data(), taps.size());
    if (config.impl == FirImpl::TiledConst || config.impl == FirImpl::Register) {
        fir_load_const_taps(d_taps.get(), num_taps, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Waits for the chunk in `slot`, then copies its outputs into y.
    const auto retire = [&](Slot& slot) {
        if (!slot.busy) {
            return;
        }
        CUDA_CHECK(cudaEventSynchronize(slot.done.get()));
        if (!registered) {
            std::memcpy(y.data() + slot.start, slot.host_out.get() + slot.offset,
                        slot.count * sizeof(cf32));
        }
        slot.busy = false;
    };

    const auto begin = std::chrono::steady_clock::now();
    int chunks = 0;
    for (std::size_t start = 0; start < num_samples; start += chunk, ++chunks) {
        Slot& slot = *slots[static_cast<std::size_t>(chunks % config.depth)];
        retire(slot);

        // History + chunk: inputs [in_start, end), outputs [start, end).
        const std::size_t end = std::min(num_samples, start + chunk);
        const std::size_t in_start = start > halo ? start - halo : 0;
        const std::size_t in_len = end - in_start;
        if (!registered) {
            std::memcpy(slot.host_in.get(), x.data() + in_start, in_len * sizeof(cf32));
        }
        slot.start = start;
        slot.count = end - start;
        slot.offset = start - in_start;

        const cudaStream_t stream = slot.stream.get();
        if (config.memory == StreamMemory::ZeroCopy) {
            fir_launch(config.impl, slot.host_in.device_ptr(), d_taps.get(),
                       slot.host_out.device_ptr(), static_cast<int>(in_len), num_taps, stream);
        } else {
            const cf32* source = registered ? x.data() + in_start : slot.host_in.get();
            CUDA_CHECK(cudaMemcpyAsync(slot.dev_in.get(), source, in_len * sizeof(cf32),
                                       cudaMemcpyHostToDevice, stream));
            fir_launch(config.impl, slot.dev_in.get(), d_taps.get(), slot.dev_out.get(),
                       static_cast<int>(in_len), num_taps, stream);
            // Only the chunk's own outputs come back; those at offset 0 .. offset - 1 lacked
            // history and are dropped.
            cf32* target = registered ? y.data() + start : slot.host_out.get() + slot.offset;
            CUDA_CHECK(cudaMemcpyAsync(target, slot.dev_out.get() + slot.offset,
                                       slot.count * sizeof(cf32), cudaMemcpyDeviceToHost,
                                       stream));
        }
        CUDA_CHECK(cudaEventRecord(slot.done.get(), stream));
        slot.busy = true;
    }
    // Drain in chunk order.
    for (int i = 0; i < config.depth; ++i) {
        retire(*slots[static_cast<std::size_t>((chunks + i) % config.depth)]);
    }
    const auto finish = std::chrono::steady_clock::now();

    timing.total_ms = std::chrono::duration<double, std::milli>(finish - begin).count();
    timing.chunks = chunks;
    return y;
}
