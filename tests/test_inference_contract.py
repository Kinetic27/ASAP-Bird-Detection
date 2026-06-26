import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.core import inference
from src.core.inference import ASAP


class InferenceContractTest(unittest.TestCase):
    def test_start_workers_forwards_filter_and_inference_options(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime = ASAP.__new__(ASAP)
            runtime.model_path = str(Path(tmp_dir) / "model.engine")
            runtime.device_list = ["cpu"]
            runtime.patch_size = 1280
            runtime.min_overlap = 128
            runtime.batch_size = 8
            runtime.global_context = False
            runtime.global_size = 1280
            runtime.num_workers_per_gpu = 1
            runtime._worker_stderr_files = []
            runtime._worker_reader_error_counts = {}

            proc = mock.Mock()
            proc.stdout.readline.return_value = ""
            with (
                mock.patch(
                    "src.core.inference.subprocess.Popen", return_value=proc
                ) as popen,
                mock.patch("src.core.inference.threading.Thread") as thread_cls,
                mock.patch.object(runtime, "_wait_for_worker_ready", return_value=None),
            ):
                thread_cls.return_value.start.return_value = None
                runtime._start_workers(
                    classes=[14],
                    conf=0.33,
                    iou_thres=0.61,
                    augment=True,
                    shm_info={"name": "demo_shm", "shape": (1, 2160, 3840, 3)},
                )

        cmd = popen.call_args.args[0]
        self.assertIn("--classes", cmd)
        self.assertEqual(cmd[cmd.index("--classes") + 1], "14")
        self.assertIn("--iou-thres", cmd)
        self.assertEqual(cmd[cmd.index("--iou-thres") + 1], "0.61")
        self.assertIn("--augment", cmd)
        self.assertIn("--worker-slot", cmd)
        self.assertEqual(cmd[cmd.index("--worker-slot") + 1], "0")

    def test_gpu_workers_get_unique_engine_copies_per_slot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            engine = tmp_path / "model.engine"
            engine.write_text("engine", encoding="utf-8")

            runtime = ASAP.__new__(ASAP)
            runtime.model_path = str(engine)
            runtime.device_list = [0]
            runtime.patch_size = 1280
            runtime.min_overlap = 128
            runtime.batch_size = 8
            runtime.global_context = False
            runtime.global_size = 1280
            runtime.num_workers_per_gpu = 2
            runtime._worker_stderr_files = []
            runtime._worker_reader_error_counts = {}

            proc = mock.Mock()
            proc.stdout.readline.return_value = ""
            with (
                mock.patch(
                    "src.core.inference.subprocess.Popen", return_value=proc
                ) as popen,
                mock.patch("src.core.inference.threading.Thread") as thread_cls,
                mock.patch.object(runtime, "_wait_for_worker_ready", return_value=None),
            ):
                thread_cls.return_value.start.return_value = None
                runtime._start_workers(
                    shm_info={"name": "demo_shm", "shape": (1, 2160, 3840, 3)}
                )

        commands = [call.args[0] for call in popen.call_args_list]
        model_paths = [cmd[cmd.index("--model") + 1] for cmd in commands]
        self.assertEqual(len(model_paths), 2)
        self.assertEqual(len(set(model_paths)), 2)
        self.assertTrue(all("_worker_gpu0_slot" in path for path in model_paths))


    def test_worker_ready_wait_fails_on_model_error_log(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            stderr_path = Path(tmp_dir) / "worker.stderr.log"
            stderr_path.write_text("[Worker 0:2] Model Error: boom\\n", encoding="utf-8")
            runtime = ASAP.__new__(ASAP)
            runtime.workers = []
            runtime._worker_stderr_files = []
            runtime.model_path = str(stderr_path)
            proc = mock.Mock()
            proc.poll.return_value = None

            with self.assertRaisesRegex(RuntimeError, "Worker failed during model warmup"):
                runtime._wait_for_worker_ready(proc, str(stderr_path), timeout_s=0.1)


    def test_worker_runner_does_not_emit_empty_results_on_exception(self):
        worker_source = Path("src/core/worker_runner.py").read_text(encoding="utf-8")
        self.assertNotIn('stdout.write(f"{frame_idx}|[]', worker_source)
        self.assertIn("sys.exit(1)", worker_source)

    def test_runtime_fails_if_worker_exits_during_inference(self):
        runtime = ASAP.__new__(ASAP)
        runtime.model_path = "/tmp/model.engine"
        runtime._worker_stderr_files = []
        proc = mock.Mock()
        proc.poll.return_value = 1
        proc.returncode = 1
        runtime.workers = [proc]

        with self.assertRaisesRegex(RuntimeError, "benchmark results are invalid"):
            runtime._raise_if_worker_exited()

    def test_default_engine_export_uses_configured_trt_max_batch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_pt = tmp_path / "yolo11n.pt"
            model_pt.write_text("fake-model", encoding="utf-8")
            built_engine = tmp_path / "yolo11n.engine"
            built_engine.write_text("engine", encoding="utf-8")

            with (
                mock.patch.object(inference, "BASE_DIR", str(tmp_path)),
                mock.patch.object(inference, "DEFAULT_MODEL_NAME", str(model_pt)),
                mock.patch(
                    "src.core.inference.export_tensorrt",
                    return_value=str(built_engine),
                ) as export,
                mock.patch(
                    "src.core.inference.torch.cuda.is_available", return_value=False
                ),
            ):
                runtime = ASAP(
                    model_path=None, device="0", patch_size=1280, batch_size=8
                )

        export.assert_called_once()
        self.assertEqual(export.call_args.kwargs["batch"], 8)
        self.assertTrue(runtime.model_path.endswith("models/yolo11n_1280.engine"))


if __name__ == "__main__":
    unittest.main()
