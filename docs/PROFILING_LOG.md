# Profiling log

One entry per optimization iteration, including negative results. Numbers come from Nsight
Systems, Nsight Compute, or CUDA events only. Timing rules are in `AGENTS.md` ("Measurement
rules"): warm up, average at least 10 runs, exclude one-time setup, and report transfer time
separately from kernel time.

No iterations recorded yet. Iteration 0 (naive FIR and naive cross-correlation baselines) is
task M6 in [`ROADMAP.md`](ROADMAP.md).

## Entry template

```markdown
## Iteration N — <kernel> — <short name of change>
- GPU / driver / CUDA version:
- Input size and parameters:
- Kernel time (mean ± std):          Transfer time:
- Achieved memory bandwidth (% of peak):     Occupancy:
- Top bottleneck reported by the profiler:
- What I changed and why:
- Result (speedup vs. previous iteration and vs. naive):
- Screenshot: docs/screenshots/...
```
