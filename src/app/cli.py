from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable

import yaml

from src.app.paths import PROJECT_ROOT, sample_image_path, sample_video_path
from src.utils.config import (
    CONF_THRES,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    MIN_OVERLAP,
    PATCH_SIZE,
    TRT_BATCH_SIZE,
    TRT_DYNAMIC_SHAPES,
)


ASAP = None
_export_tensorrt = None
LEGACY_COMMANDS = {"video", "image", "export"}


class PublicRuntimeError(RuntimeError):
    """User-facing runtime setup error without a Python traceback."""


def missing_dependency_message(package: str, command_hint: str = "runtime") -> str:
    return (
        f"Missing optional {command_hint} dependency: {package}. "
        "Install the project requirements in the target environment, then retry. "
        "See README.md for setup guidance."
    )


def get_asap_class():
    """Import the heavy inference runtime only for commands that need it."""
    global ASAP
    if ASAP is None:
        try:
            from src.core.inference import ASAP as loaded_asap
        except ModuleNotFoundError as exc:
            raise PublicRuntimeError(missing_dependency_message(exc.name)) from exc

        ASAP = loaded_asap
    return ASAP


def get_export_tensorrt():
    """Import the optional Ultralytics export helper only for export commands."""
    global _export_tensorrt
    if _export_tensorrt is None:
        try:
            from src.utils.export import export_tensorrt as loaded_export_tensorrt
        except ModuleNotFoundError as exc:
            raise PublicRuntimeError(missing_dependency_message(exc.name, "export")) from exc

        _export_tensorrt = loaded_export_tensorrt
    return _export_tensorrt


def load_config(config_path: str) -> dict:
    """Load YAML config and handle recursive inheritance via __include__ or __base__."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
        return {}

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    merged_flat: dict = {}
    includes: list[str] = []

    if "__include__" in config:
        include_value = config.pop("__include__")
        if isinstance(include_value, list):
            includes.extend(include_value)
        else:
            includes.append(include_value)

    if "__base__" in config:
        includes.append(config.pop("__base__"))

    for include_path in includes:
        resolved = include_path
        if not os.path.isabs(resolved):
            resolved = os.path.join(os.path.dirname(config_path), resolved)
        merged_flat.update(load_config(resolved))

    current_flat: dict = {}
    flatten_config(config, current_flat)
    merged_flat.update(current_flat)
    return merged_flat


def flatten_config(source: dict, target: dict) -> None:
    for key, value in source.items():
        if isinstance(value, dict):
            flatten_config(value, target)
        else:
            target[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ASAP small-bird detection runtime and demo CLI. "
            "Use demo/doctor for the public path, or video/image/export for advanced runtime access."
        )
    )
    parser.add_argument("-c", "--config", help="Path to YAML config file")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    add_export_parser(subparsers)
    add_video_parser(subparsers)
    add_image_parser(subparsers)
    add_demo_parser(subparsers)
    add_doctor_parser(subparsers)
    return parser


def add_export_parser(subparsers) -> None:
    export_parser = subparsers.add_parser(
        "export", help="Export YOLO model to TensorRT"
    )
    export_parser.add_argument(
        "-m", "--model", type=str, default=DEFAULT_MODEL_NAME, help="Path to .pt model"
    )
    export_parser.add_argument("-d", "--device", type=str, default="0", help="Device ID")
    export_parser.add_argument(
        "--imgsz", type=int, default=PATCH_SIZE, help="Input image size"
    )
    export_parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=TRT_BATCH_SIZE,
        help="TensorRT engine batch/profile size",
    )
    dynamic_group = export_parser.add_mutually_exclusive_group()
    dynamic_group.add_argument(
        "--dynamic",
        dest="dynamic",
        action="store_true",
        help="Export a dynamic-shape TensorRT engine",
    )
    dynamic_group.add_argument(
        "--static",
        dest="dynamic",
        action="store_false",
        help="Export a fixed-shape TensorRT engine for lower context memory",
    )
    export_parser.set_defaults(dynamic=TRT_DYNAMIC_SHAPES)


def add_shared_runtime_arguments(parser, *, include_stride: bool) -> None:
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Path to model engine (auto-builds if missing)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Device ID (e.g. 0 or 0,1 or all). Default: all available",
    )
    parser.add_argument("-p", "--patch-size", type=int, default=1280, help="Patch size")
    parser.add_argument(
        "-mo",
        "--min-overlap",
        type=int,
        default=MIN_OVERLAP,
        help="Minimum overlap in pixels",
    )
    parser.add_argument(
        "-cf",
        "--conf",
        type=float,
        default=CONF_THRES,
        help="Confidence threshold (0.001~1.0)",
    )
    parser.add_argument(
        "-rs",
        "--resize",
        type=str,
        default=None,
        help="Resize input to W,H (e.g. 2560,1440)",
    )
    parser.add_argument(
        "-cls",
        "--classes",
        type=str,
        default=None,
        help="Comma separated class indices (e.g. 0,14)",
    )
    parser.add_argument(
        "-w",
        "--workers-per-gpu",
        type=int,
        default=2,
        help="Number of workers per GPU; increase only after TensorRT warmup fits GPU memory",
    )
    parser.add_argument(
        "-iou",
        "--iou-thres",
        type=float,
        default=0.45 if include_stride else None,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "-gc",
        "--global-context",
        action="store_true",
        help="Add resized full frame for global context",
    )
    parser.add_argument(
        "-gs",
        "--global-size",
        type=int,
        default=None,
        help="Resize global context to this size (default: same as patch size)",
    )
    parser.add_argument(
        "-aug",
        "--augment",
        action="store_true",
        help="Enable Test-Time Augmentation (TTA)",
    )

    if include_stride:
        parser.add_argument(
            "-s",
            "--stride",
            type=int,
            default=1,
            help="Frame stride (process every Nth frame)",
        )
        parser.add_argument(
            "-off", "--offset", type=int, default=0, help="Frame start offset"
        )
        parser.add_argument(
            "-sv", "--save", action="store_true", help="Save output video"
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=8,
            help="Inference batch size per worker",
        )
        parser.add_argument(
            "-plot",
            "--plot-fps",
            action="store_true",
            help="Generate and save an FPS/latency benchmark plot",
        )
        parser.add_argument(
            "-nf",
            "--num-frames",
            type=int,
            default=None,
            help="Limit processing to N frames",
        )
        parser.add_argument(
            "-sj",
            "--save-json",
            action="store_true",
            help="Save detection results to a JSON file",
        )
        parser.add_argument(
            "--bounded-latency",
            action="store_true",
            help="Enable bounded-latency mode with queue/backpressure control",
        )
        parser.add_argument(
            "--max-in-flight",
            type=int,
            default=64,
            help="Maximum in-flight frames (dispatch backlog cap)",
        )
        parser.add_argument(
            "--drop-policy",
            type=str,
            default="drop_oldest",
            choices=["drop_oldest", "drop_newest", "latest_only"],
            help="Frame drop policy when backlog cap is reached",
        )
        parser.add_argument(
            "--input-fps-cap",
            type=float,
            default=None,
            help="Optional input dispatch rate cap (FPS) for camera-rate emulation",
        )
    else:
        parser.add_argument(
            "-sv", "--save", action="store_true", help="Save output image"
        )
        parser.add_argument(
            "-sj",
            "--save-json",
            action="store_true",
            help="Save results to detections.json",
        )


def add_video_parser(subparsers) -> None:
    video_parser = subparsers.add_parser(
        "video", help="Process video or a live stream (legacy advanced runtime command)"
    )
    video_parser.add_argument(
        "-i", "--input", type=str, required=False, help="Video file or RTSP stream"
    )
    add_shared_runtime_arguments(video_parser, include_stride=True)


def add_image_parser(subparsers) -> None:
    image_parser = subparsers.add_parser(
        "image", help="Process images (legacy advanced runtime command)"
    )
    image_parser.add_argument(
        "-i", "--input", type=str, required=False, help="Input directory or image"
    )
    add_shared_runtime_arguments(image_parser, include_stride=False)


def add_demo_parser(subparsers) -> None:
    demo_parser = subparsers.add_parser(
        "demo", help="Run a paper-aligned demo using local sample defaults or explicit inputs"
    )
    demo_subparsers = demo_parser.add_subparsers(dest="demo_command")

    demo_video_parser = demo_subparsers.add_parser(
        "video", help="Run the sample video demo"
    )
    demo_video_parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=sample_video_path(),
        help="Video file or RTSP stream (default: local sample video path)",
    )
    add_shared_runtime_arguments(demo_video_parser, include_stride=True)

    demo_image_parser = demo_subparsers.add_parser(
        "image", help="Run the sample image demo"
    )
    demo_image_parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=sample_image_path(),
        help="Input directory or image (default: local sample image path)",
    )
    add_shared_runtime_arguments(demo_image_parser, include_stride=False)


def add_doctor_parser(subparsers) -> None:
    subparsers.add_parser(
        "doctor", help="Run environment and path checks for the public project surface"
    )


def resolve_cli_args(argv: list[str], defaults: dict) -> list[str]:
    config_args = build_config_args(defaults)
    command = None

    cli_command = argv[0] if argv and argv[0] in LEGACY_COMMANDS else None
    cli_args = argv[1:] if cli_command else list(argv)
    cli_args = strip_config_flags(cli_args)
    config_mode = defaults.get("mode") if defaults else None

    if cli_command:
        command = cli_command
    elif config_mode in LEGACY_COMMANDS:
        command = config_mode

    if command:
        filtered_defaults = {
            key: value for key, value in defaults.items() if key != "mode"
        }
        return [command, *build_config_args(filtered_defaults), *cli_args]

    return [*config_args, *cli_args]


def strip_config_flags(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for index, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg in {"-c", "--config"}:
            if index + 1 < len(argv):
                skip_next = True
            continue
        cleaned.append(arg)
    return cleaned


def build_config_args(defaults: dict) -> list[str]:
    config_args: list[str] = []
    for key, value in defaults.items():
        if key == "mode" or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                config_args.append(flag)
        elif isinstance(value, list):
            config_args.append(flag)
            if key == "classes":
                config_args.append(",".join(map(str, value)))
            else:
                config_args.append(str(value))
        else:
            config_args.extend([flag, str(value)])
    return config_args


def parse_resize_argument(value: str | None):
    if not value:
        return None
    try:
        width, height = map(int, value.split(","))
    except ValueError as exc:
        raise ValueError("Error: --resize must be in format W,H (e.g. 2560,1440)") from exc
    return (width, height)


def parse_classes_argument(value: str | None):
    if not value:
        return None
    return [int(item) for item in value.split(",")]


def build_runtime(args, *, include_batch_size: bool):
    global_size = args.global_size if args.global_size is not None else args.patch_size
    runtime_kwargs = dict(
        model_path=args.model,
        device=args.device,
        patch_size=args.patch_size,
        min_overlap=args.min_overlap,
        num_workers_per_gpu=args.workers_per_gpu,
        global_context=args.global_context,
        global_size=global_size,
    )
    if include_batch_size:
        runtime_kwargs["batch_size"] = args.batch_size
    return get_asap_class()(**runtime_kwargs)


def run_export(args) -> int:
    engine_path = get_export_tensorrt()(
        args.model, args.device, imgsz=args.imgsz, batch=args.batch, dynamic=args.dynamic
    )
    if engine_path:
        print(f"Success! Engine path: {engine_path}")
        return 0
    return 1


def run_video(args) -> int:
    resize_dim = parse_resize_argument(args.resize)
    classes_list = parse_classes_argument(args.classes)
    infer = build_runtime(args, include_batch_size=True)
    infer.predict_video(
        args.input,
        args.output,
        save=args.save,
        save_json=args.save_json,
        resize=resize_dim,
        classes=classes_list,
        conf_thres=args.conf,
        iou_thres=args.iou_thres,
        augment=args.augment,
        stride=args.stride,
        offset=args.offset,
        num_frames=args.num_frames,
        plot_fps=args.plot_fps,
        bounded_latency=args.bounded_latency,
        max_in_flight=args.max_in_flight,
        drop_policy=args.drop_policy,
        input_fps_cap=args.input_fps_cap,
    )
    return 0


def run_image(args) -> int:
    resize_dim = parse_resize_argument(args.resize)
    classes_list = parse_classes_argument(args.classes)
    infer = build_runtime(args, include_batch_size=False)
    results = infer.predict_image(
        args.input,
        args.output,
        save=args.save,
        resize=resize_dim,
        classes=classes_list,
        iou_thres=args.iou_thres,
        conf_thres=args.conf,
        augment=args.augment,
    )

    if args.save_json:
        json_results = {
            os.path.basename(img_path): boxes for img_path, boxes in results.items()
        }
        json_path = os.path.join(args.output, "detections.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(json_results, handle, indent=4)
        print(f"Results saved to {json_path}")
    return 0


def run_demo(args) -> int:
    if args.demo_command == "video":
        return run_video(args)
    if args.demo_command == "image":
        return run_image(args)
    raise ValueError("demo requires a subcommand: video or image")


def run_doctor(_args) -> int:
    """Run a lightweight public environment check without external scripts."""
    print("ASAP Bird Detection doctor")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    optional_paths = {
        "default model": DEFAULT_MODEL_NAME,
        "sample video": sample_video_path(),
        "sample image": sample_image_path(),
    }
    for label, path in optional_paths.items():
        status = "found" if os.path.exists(path) else "missing (optional)"
        print(f"{label}: {status} - {path}")
    print("Doctor result: OK")
    return 0


def dispatch(args) -> int:
    if args.command == "export":
        return run_export(args)
    if args.command == "video":
        return run_video(args)
    if args.command == "image":
        return run_image(args)
    if args.command == "demo":
        return run_demo(args)
    if args.command == "doctor":
        return run_doctor(args)
    raise ValueError(f"Unsupported command: {args.command}")


def main(argv: Iterable[str] | None = None) -> int:
    incoming = list(argv if argv is not None else sys.argv[1:])
    config_path = find_config_path(incoming)
    defaults = load_config(config_path) if config_path else {}
    if config_path and not wants_help(incoming):
        print(f"Loaded configuration from {config_path}", flush=True)

    parser = build_parser()
    final_args = resolve_cli_args(incoming, defaults)
    args = parser.parse_args(final_args)

    if not args.command:
        parser.print_help()
        return 1

    try:
        return dispatch(args)
    except PublicRuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def find_config_path(argv: list[str]) -> str | None:
    for index, arg in enumerate(argv):
        if arg in {"-c", "--config"} and index + 1 < len(argv):
            return argv[index + 1]
    return None


def wants_help(argv: Iterable[str]) -> bool:
    return any(arg in {"-h", "--help"} for arg in argv)
