# Roadmap

Work that is out of scope for the MVP (minimum viable product). Nothing here gets code until the
MVP in README.md is finished and measured.

1. **Streaming.** A ring buffer of pinned host memory and CUDA streams that overlap transfers
   with compute, processing continuous chunks the way live radio data arrives. Measured against
   real-time throughput (samples per second).
2. **Channelizer.** A polyphase filter bank that splits a wideband capture into sub-channels.
3. **Direction finding.** GCC-PHAT (generalized cross-correlation with phase transform) for TDOA
   (time difference of arrival) across channels, then DOA (direction of arrival) estimation for
   a simulated antenna array.
4. **Real recordings.** Public SigMF datasets in addition to synthetic data.
5. **Jetson portability.** Notes on running on Jetson: unified memory, power and clock limits.

## Deferred ideas

- Fractional-sample delays in the two-channel fixture (the MVP fixture uses an integer delay).
