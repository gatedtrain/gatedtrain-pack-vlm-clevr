"""One name, one place: values the pack shares with gatedtrain, with its converter or with its container scripts, and
the settings that belong to the owner's host."""

from __future__ import annotations

import ast
import io
import os
import re
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path, PurePosixPath
from unittest import mock

from gatedtrain.ainative import hostsetup
from gatedtrain.ainative.batch import DEFAULT_TIME_LIMIT_S
from gatedtrain.families.nemo_rl import NEMO_RL_RUN_OPTIONS

from gatedtrain_pack_vlm_clevr import adapter, data, job, scenario, stages

PACKAGE = Path(stages.__file__).resolve().parent


def _literal(path: Path, name: str):
    """The value of the module-level assignment `name` in `path`, read without importing it (the container scripts
    import torch, NeMo RL or vLLM)."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


class SharedWithGatedtrain(unittest.TestCase):
    def test_the_generator_time_limit_is_gatedtrains_default(self) -> None:
        self.assertEqual(scenario.TIME_LIMIT_S, DEFAULT_TIME_LIMIT_S)

    def test_the_eval_python_is_in_the_eval_venv_gatedtrain_builds(self) -> None:
        venv = PurePosixPath(hostsetup.DEFAULT_EVAL_VENV)
        self.assertEqual(scenario.DEFAULT_EVAL_PYTHON, str(venv / "bin" / "python"))

    def test_the_run_options_are_gatedtrains_minus_memlock(self) -> None:
        """A rootless daemon cannot raise the locked-memory limit: gatedtrain's options without that one pair."""
        opts = list(NEMO_RL_RUN_OPTIONS)
        i = opts.index("memlock=-1")
        self.assertEqual(opts[i - 1], "--ulimit")
        self.assertEqual(adapter.RUN_OPTIONS, tuple(opts[:i - 1] + opts[i + 1:]))


class SharedWithTheConverter(unittest.TestCase):
    def test_one_definition_each(self) -> None:
        self.assertIs(stages.MAX_PIXELS, data.MAX_PIXELS)
        self.assertIs(stages.PNG_MAGIC, data.PNG_MAGIC)
        self.assertIs(stages.JPEG_MAGIC, data.JPEG_MAGIC)

    def test_the_generator_script_agrees(self) -> None:
        """generate_vllm.py runs in the keeper's container and cannot import the pack."""
        script = PACKAGE / "generate_vllm.py"
        self.assertEqual(_literal(script, "MAX_PIXELS"), data.MAX_PIXELS)
        magic = _literal(script, "MAGIC")
        self.assertEqual((magic["png"], magic["jpeg"]), (data.PNG_MAGIC, data.JPEG_MAGIC))


class SharedWithTheTrainingScripts(unittest.TestCase):
    def test_the_profiles(self) -> None:
        profiles = _literal(PACKAGE / "runtime" / "run_vlm_nemo.py", "PROFILES")
        self.assertEqual(stages.PROFILE_STEPS, {k: v["steps"] for k, v in profiles.items()})
        self.assertEqual({v["prompts"] for v in profiles.values()}, {stages.PROMPTS_PER_STEP})

    def test_the_greedy_sampler_of_the_cut(self) -> None:
        self.assertEqual(_literal(PACKAGE / "runtime" / "ptc_vlm_hooks.py", "DEPLOYMENT_SAMPLER"),
                         stages.GREEDY_SAMPLER)

    def test_the_gpu_estimate_has_one_home(self) -> None:
        self.assertEqual(adapter.TRAINER_ROWS[adapter.NAME]["gpu_h_estimate"], job.PROFILES["full"]["gpu_hours_est"])

    def test_the_sequence_limit_covers_the_answer(self) -> None:
        self.assertGreaterEqual(_literal(PACKAGE / "runtime" / "run_vlm_nemo.py", "MAX_SEQ"),
                                _literal(PACKAGE / "generate_vllm.py", "MAX_MODEL_LEN"))


class GpuArchitecture(unittest.TestCase):
    """The architecture lists name a GPU of the owner's host: a stage parameter, left out unless set."""

    def test_the_job_env_has_no_architecture_by_default(self) -> None:
        self.assertNotIn("TORCH_CUDA_ARCH_LIST", job.ENV)
        self.assertNotIn("FLASHINFER_CUDA_ARCH_LIST", job.ENV)

    def test_format(self) -> None:
        for good in ("8.9", "9.0", "9.0a", "12.0a", "8.9+PTX", "7.5 8.0"):
            self.assertTrue(job.valid_cuda_arch(good), good)
        for bad in ("", "89", "8.9;", "8.0;8.6", "a.b", "8.9\n", "8.9; rm", "8.9,9.0", "8.9  9.0", "8.9 " * 40, None, 8.9):
            self.assertFalse(job.valid_cuda_arch(bad), bad)

    def test_the_stage_params_name_it(self) -> None:
        for cls in (stages.Smoke, stages.Train):
            self.assertIn("cuda_arch", cls.params)
            self.assertIn("image", cls.params)

    def test_the_pipeline_file_documents_it(self) -> None:
        self.assertIn("cuda_arch", (PACKAGE / "pipeline.yaml").read_text())

    def test_every_env_name_the_job_may_set_is_registered(self) -> None:
        names = set(adapter.TRAINER_ROWS[adapter.NAME]["env_names"])
        self.assertTrue({"TORCH_CUDA_ARCH_LIST", "FLASHINFER_CUDA_ARCH_LIST"} <= names)
        self.assertTrue(set(job.ENV) <= names)


class OwnerChannels(unittest.TestCase):
    """The owner's settings come from the host file and the keeper's options; a variable is a named, deprecated
    workaround that says so."""

    def host_file(self, text: str) -> dict[str, str]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "host.toml"
        path.write_text(text, encoding="utf-8")
        return {"GATEDTRAIN_HOST_CONFIG": str(path)}

    def runtime(self, env: dict[str, str]) -> str:
        from gatedtrain.ext.rowkit import runtime_dir
        with mock.patch.dict(os.environ, env), redirect_stderr(io.StringIO()):
            os.environ.pop(adapter.RUNTIME_ENV, None) if adapter.RUNTIME_ENV not in env else None
            return runtime_dir(adapter.RUNTIME_ENV, adapter.__file__, parents=0, pack=adapter.PACK)

    def test_the_runtime_directory_is_a_host_file_setting(self) -> None:
        self.assertEqual(adapter.PACK, "vlm-clevr")
        env = self.host_file('[packs.vlm-clevr]\nruntime_dir = "/owner/runtime"\n')
        self.assertEqual(self.runtime(env), "/owner/runtime")
        self.assertTrue(self.runtime(self.host_file("")).endswith("runtime"))

    def test_the_runtime_variable_is_deprecated_and_says_so(self) -> None:
        err = io.StringIO()
        from gatedtrain.ext import rowkit
        rowkit._WARNED.discard(adapter.RUNTIME_ENV)                 # the notice is printed once per process
        with mock.patch.dict(os.environ, {adapter.RUNTIME_ENV: "/e"}), redirect_stderr(err):
            from gatedtrain.ext.rowkit import runtime_dir
            self.assertEqual(runtime_dir(adapter.RUNTIME_ENV, adapter.__file__, parents=0, pack=adapter.PACK), "/e")
        self.assertIn("deprecated", err.getvalue())
        self.assertIn("[packs.vlm-clevr]", err.getvalue())

    def test_the_eval_python_variable_is_deprecated_and_says_so(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(scenario.eval_python({scenario.EVAL_PYTHON_ENV: "/e/bin/python"}), "/e/bin/python")
            self.assertEqual(scenario.eval_python({}), scenario.DEFAULT_EVAL_PYTHON)
        self.assertEqual(err.getvalue().count("deprecated"), 1)
        self.assertIn("--eval-python", err.getvalue())


if __name__ == "__main__":
    unittest.main()
