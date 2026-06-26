# ASAP Bird Detection

ASAP (Asynchronous Slicing Accelerated Pipeline) runtime code for high-resolution small-bird detection experiments.

## What this repository supports

- Public demo commands for user-supplied local image/video inputs
- Core ASAP image/video inference pipeline
- TensorRT export helper for rebuilding machine-specific engines
- Lightweight environment checks and unit tests

## Public/private asset boundary

- The paper's internal private 4K surveillance dataset is not published in this repository.
- The local `data/` directory, including `data/samples/`, is not part of the public repository contract.
- Model weights, TensorRT engines, generated outputs, paper figures, and raw experiment logs are not included.
- Paper benchmark reference numbers are documented in `PAPER_RESULTS.md`; exact paper-number regeneration is intentionally not packaged as public tooling because it depends on private assets and matching hardware/software setup.

## Quick Start

```bash
cd <repo-root>
python -m pip install -r requirements.txt
python main.py doctor
```

TensorRT export dependencies are split out because they are only needed on CUDA/TensorRT machines:

```bash
python -m pip install -r requirements-engine.txt
```

## Run a demo with your own input

```bash
python main.py demo video -i /path/to/video.mp4
python main.py demo image -i /path/to/image.jpg --save -o outputs/runtime
```

Default sample paths are local workspace conveniences only; public clones should pass explicit `-i/--input` paths.

## Advanced runtime commands

```bash
python main.py video -i /path/to/video.mp4 --model /path/to/yolo11n_1280.engine
python main.py image -i /path/to/images --model /path/to/yolo11n_1280.engine --save
python main.py export -m /path/to/yolo11n.pt --imgsz 1280 --batch 8 --static
```

For the fixed 1280px / 8-patch paper-speed path, rebuild a TensorRT FP16 engine on the target machine with `--batch 8 --static`. A dynamic batch-8 engine also works, but on pr6/TITAN RTX it used much more TensorRT context memory and was slightly slower in the full-video FPS run. Avoid exporting the 1280px profile at batch 16 for 4-workers/GPU runs on 24GB cards; that profile can fail worker warmup with TensorRT context OOM.

The runtime now warms every worker before dispatching frames; if a higher worker count or larger TensorRT export profile does not fit GPU memory, the command fails during startup instead of reporting misleading FPS from dead workers.

Runtime filters and NMS settings are available from the CLI and YAML config. For example, `--classes 14 --iou-thres 0.45 --augment` is forwarded to worker-side YOLO inference, and `--iou-thres` is also used by the final cross-patch NMS.

Use `--plot-fps` to save a benchmark plot in the output directory. The stable-FPS summary excludes the detected warmup ramp and final drain/spike samples; use a sufficiently long video because very short smoke runs cannot provide a reliable adaptive warmup boundary. Bounded-latency mode intentionally caps throughput near the input rate, so use an unbounded run when comparing against the raw paper FPS row. Always check `worker_gpu*_slot*.stderr.log` for zero errors before trusting benchmark numbers.

## YAML configuration

`configs/default.yaml` is an editable source-tree example for the legacy runtime commands:

```bash
python main.py --config configs/default.yaml
```

The config loader flattens nested sections such as `data.input`, `model.model`, and `inference.iou_thres` into the matching CLI flags. Replace the sample input/model paths with local files before running; datasets, weights, and engines are intentionally not part of the public repository contract.

## Python import boundary

The project can be imported from a source checkout when the repository root is on `PYTHONPATH`, for example `from src.core.inference import ASAP`. It is not packaged as a pip-installable library and does not promise a stable public Python API beyond the documented CLI/demo surface.

## Verification

```bash
bash scripts/verify_public_surface.sh
```

## Reference files

- `PAPER_RESULTS.md` — paper benchmark reference values and public/private boundary

## Local-only files

Do not commit datasets, model weights, TensorRT engines, generated outputs, paper figures, raw experiment logs, local environment files, credentials, or personal integration code. The repository `.gitignore` covers the usual paths and file types.

## License

This project is released under the GNU Affero General Public License v3.0 (`AGPL-3.0-only`). It uses Ultralytics YOLO tooling, which is available under AGPL-3.0 or a commercial Ultralytics Enterprise license.

## Citation

If you use this repository, cite the ASAP paper/manuscript:

- `CITATION.cff`
- Paper title: **Asynchronous Slicing Accelerated Pipeline for Real-time High-Resolution Small Bird Flock Monitoring**

Use the final publication metadata when the paper is formally published.
