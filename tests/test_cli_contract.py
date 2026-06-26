import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.app import cli


class LoadConfigContractTest(unittest.TestCase):
    def test_load_config_merges_includes_and_flattens_nested_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            base = tmp_path / "base.yaml"
            child = tmp_path / "child.yaml"

            base.write_text(
                "mode: video\nruntime:\n  patch_size: 960\n  conf: 0.3\n",
                encoding="utf-8",
            )
            child.write_text(
                "__base__: base.yaml\nruntime:\n  conf: 0.5\nnested:\n  stride: 3\n",
                encoding="utf-8",
            )

            loaded = cli.load_config(str(child))

        self.assertEqual(loaded["mode"], "video")
        self.assertEqual(loaded["patch_size"], 960)
        self.assertEqual(loaded["conf"], 0.5)
        self.assertEqual(loaded["stride"], 3)


class LegacyCliContractTest(unittest.TestCase):
    @mock.patch("src.app.cli.ASAP")
    def test_video_command_preserves_legacy_runtime_invocation(self, asap_cls):
        runtime = asap_cls.return_value
        exit_code = cli.main(
            [
                "video",
                "-i",
                "input.mp4",
                "--workers-per-gpu",
                "5",
                "--batch-size",
                "6",
                "--bounded-latency",
                "--max-in-flight",
                "11",
                "--drop-policy",
                "latest_only",
                "--input-fps-cap",
                "15",
                "--iou-thres",
                "0.61",
                "--classes",
                "14",
                "--augment",
                "--save-json",
                "--global-context",
            ]
        )

        self.assertEqual(exit_code, 0)
        asap_cls.assert_called_once_with(
            model_path=None,
            device=None,
            patch_size=1280,
            min_overlap=128,
            num_workers_per_gpu=5,
            batch_size=6,
            global_context=True,
            global_size=1280,
        )
        runtime.predict_video.assert_called_once()
        _, kwargs = runtime.predict_video.call_args
        self.assertEqual(kwargs["save_json"], True)
        self.assertEqual(kwargs["bounded_latency"], True)
        self.assertEqual(kwargs["max_in_flight"], 11)
        self.assertEqual(kwargs["drop_policy"], "latest_only")
        self.assertEqual(kwargs["input_fps_cap"], 15.0)
        self.assertEqual(kwargs["iou_thres"], 0.61)
        self.assertEqual(kwargs["classes"], [14])
        self.assertEqual(kwargs["augment"], True)

    @mock.patch("src.app.cli.ASAP")
    def test_config_mode_can_drive_legacy_command_without_cli_subcommand(self, asap_cls):
        runtime = asap_cls.return_value
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = os.path.join(tmp_dir, "out")
            config_path = os.path.join(tmp_dir, "config.yaml")
            Path(config_path).write_text(
                "\n".join(
                    [
                        "mode: image",
                        "input: sample.jpg",
                        f"output: {output_dir}",
                        "save_json: true",
                        "global_context: true",
                    ]
                ),
                encoding="utf-8",
            )

            exit_code = cli.main(["--config", config_path])
            json_path = Path(output_dir) / "detections.json"
            self.assertTrue(json_path.exists())

        self.assertEqual(exit_code, 0)
        asap_cls.assert_called_once_with(
            model_path=None,
            device=None,
            patch_size=1280,
            min_overlap=128,
            num_workers_per_gpu=2,
            global_context=True,
            global_size=1280,
        )
        runtime.predict_image.assert_called_once()


class PublicCliContractTest(unittest.TestCase):
    @mock.patch("src.app.cli.sample_video_path", return_value="demo.mp4")
    @mock.patch("src.app.cli.ASAP")
    def test_demo_video_uses_sample_input_defaults(self, asap_cls, _sample_video_path):
        runtime = asap_cls.return_value

        exit_code = cli.main(["demo", "video"])

        self.assertEqual(exit_code, 0)
        runtime.predict_video.assert_called_once()
        args, _kwargs = runtime.predict_video.call_args
        self.assertEqual(args[0], "demo.mp4")

    @mock.patch("src.app.cli.get_asap_class")
    def test_missing_runtime_dependency_returns_public_error(self, get_asap_class):
        get_asap_class.side_effect = cli.PublicRuntimeError("missing dependency")

        with mock.patch("sys.stderr") as stderr:
            exit_code = cli.main(["demo", "video"])

        self.assertEqual(exit_code, 1)
        stderr.write.assert_any_call("Error: missing dependency")

    def test_legacy_paper_command_is_not_public_surface(self):
        legacy_command = "repro" + "duce"
        with (
            self.assertRaises(SystemExit) as ctx,
            mock.patch("sys.stderr"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            cli.main([legacy_command, "tables"])
        self.assertEqual(ctx.exception.code, 2)

    def test_doctor_runs_internal_public_check(self):
        exit_code = cli.main(["doctor"])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
