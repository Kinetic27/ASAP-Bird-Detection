# Paper Benchmark Reference

These are the ASAP paper reference numbers. They are listed separately from local smoke-test results because the private 4K bird-monitoring video and
machine-specific TensorRT engines are not part of the public repository.

## Runtime comparison (`tab:perf`)

| Method | Patch size | 4K patches | FPS |
| --- | ---: | ---: | ---: |
| SAHI | 640 | 32 | 1.08 |
| Baseline (Sync) | 640 | 32 | 8.43 |
| Proposed (Async) | 640 | 32 | 37.07 |
| Proposed (Async) | 960 | 15 | 31.87 |
| Proposed (Async) | 1280 | 8 | 43.84 |
| Proposed (Async) | 1920 | 6 | N/A |

The selected paper operating point is **1280 px / 8 patches / 43.84 FPS** on a
dual NVIDIA TITAN RTX workstation. The 1920 px row was not reported as a stable,
directly comparable long-sequence run under the same 4-workers-per-GPU setting.


## Local pr6 reproduction note (2026-06-26)

Clean local runs on `data/samples/DSC_1132_long.mp4` with rebuilt TensorRT
10.15 FP16 engines, unbounded queueing, `--plot-fps`, runtime `--batch-size 8`,
and `--workers-per-gpu 4` completed with all eight worker stderr logs reporting
ready and zero errors:

| Run | Engine export | Workers/GPU | Avg FPS | Stable FPS mean | Stable FPS max | Drops |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `full_b8engine_4w_b8_20260626_220837` | `--batch 8 --dynamic` | 4 | 36.68 | 38.93 | 55.87 | 0.00% |
| `full_static_b8_4w_b8_20260626_221721` | `--batch 8 --static` | 4 | 38.34 | 40.01 | 48.00 | 0.00% |

TensorRT context-memory inspection on the same stack showed why export settings
matter: dynamic batch-8 required 3,532,390,400 bytes per context, static batch-8
required 381,747,200 bytes, and dynamic batch-16 required 7,064,780,800 bytes.
The batch-16 profile did **not** validate `--workers-per-gpu 4` on this 24GB-GPU
pr6 stack: worker warmup failed with TensorRT `set_input_shape` context errors /
CUDA OOM. Earlier high-FPS runs with dead workers are invalid because failed
workers returned empty detections quickly. Use a batch-8 export profile for the
paper-aligned 1280px / 8-patch setting; use static export for the fixed-shape
speed run unless variable runtime shapes are required.

## Speedup decomposition (`tab:ablation_speedup`)

| Configuration | Stack | FPS |
| --- | --- | ---: |
| Sync | PyTorch FP32 + Sync | 9.50 |
| Sync | TensorRT FP16 + Sync | 15.40 |
| Async | TensorRT FP16 + Async | 37.07 |

Reference gains: engine **1.62x**, scheduling **2.41x**, total controlled
**3.90x**.

## Bounded-latency policy (`tab:bounded_policy`)

| Max in-flight | Stable FPS | Drop rate | Freshness p95 |
| ---: | ---: | ---: | ---: |
| 2 | 12.68 | 64.44% | 33.37 ms |
| 4 | 29.93 | 11.06% | 100.10 ms |
| 5 | 29.95 | 13.01% | 100.10 ms |
| 6 | 29.93 | 9.81% | 100.10 ms |

Interpretation: the 43.84 FPS row is raw async pipeline throughput at the
selected 1280 px patch size. The bounded-latency controller intentionally keeps
real-time 30 FPS input replay near the input rate under overload; therefore a
bounded-latency smoke run around 30 stable FPS is not a contradiction of the
43.84 FPS raw-throughput paper row.

## Detection-quality reference (`tab:detect`)

| Patch size | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | AP_S |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 640 | 0.338 | 0.705 | 0.353 | 0.193 | 0.164 |
| 960 | 0.382 | 0.723 | 0.389 | 0.222 | 0.183 |
| 1280 | 0.411 | 0.729 | 0.405 | 0.231 | 0.187 |
| 1920 | 0.420 | 0.735 | 0.412 | 0.239 | 0.195 |

## Reproduction boundary

- Public sample/demo runs are smoke tests and should not be cited as paper
  results.
- Generated files under `outputs/` are local run artifacts. If an
  old artifact disagrees with this table, treat it as stale until regenerated
  from the intended manuscript environment.
- Exact paper-number regeneration requires the original private 4K video,
  compatible TensorRT engines rebuilt on the target GPU stack, and the same
  benchmark command settings used for the paper. For the paper-aligned 1280px
  / 8-patch setting on pr6, export a FP16 TensorRT engine with max batch 8;
  static export is lower-memory/faster for the fixed shape, while dynamic
  export is useful only when varying runtime shapes are required. Larger export
  profiles can consume enough context memory to prevent 4-workers/GPU startup
  on 24GB cards.
- For raw FPS comparison, use an unbounded `--plot-fps` run and compare the
  stable-FPS interval after adaptive warmup/tail trimming. Bounded-latency
  MIF runs are expected to sit near the input rate and should not be compared
  directly to the 43.84 FPS raw-throughput row.
- A benchmark is valid only if every worker completes TensorRT warmup and the
  corresponding `worker_gpu*_slot*.stderr.log` files have zero runtime errors.
