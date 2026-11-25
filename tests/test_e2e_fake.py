"""The pack's pipeline end to end on the CPU through gatedtrain's command line, with fake container runtimes and a
same-user keeper (the pattern of gatedtrain's `tests/test_an_backend_e2e.py`).

Needs a gatedtrain source checkout (its test harness is not in the wheel): `GATEDTRAIN_SRC` names it, and the test
skips when it is unset. Every command is a subprocess of `fake_cli.py --json ...` (this pack's real entry points,
gatedtrain's runtime switches), every decision comes from a file and stderr is empty on every call.

Nothing touches Docker, a GPU or the host's Docker groups:

- the backend's containers run on `FakeRuntime` (`GATEDTRAIN_TEST_BACKEND_RUNTIME`); the smoke and full training jobs
  are played by `_pack_fakes.play_training` (samples, memory and checkpoint files as `train_grpo.py` writes them);
- the keeper is a same-user subprocess (gatedtrain's `tests/_keeper.py`, started through `fake_cli.py`) whose
  inference containers run on its own `FakeRuntime` (`GATEDTRAIN_TEST_KEEPER_RUNTIME`) and are played by
  `_pack_fakes.play_inference`: the candidate answers every item right, the base answers 0;
- the keeper's eval tree is a temporary directory holding a stand-in interpreter file, which the batch generator
  names through `GATEDTRAIN_PACK_TEXT_GSM8K_EVAL_PYTHON`;
- the switches make the Docker CLI runtime and the `nvidia-smi` probe raise, and every subprocess finds a `docker`
  and an `nvidia-smi` on its `PATH` that only leave a mark: the test fails when a mark exists.

No sealed input appears in any output, project file, job log, job record, backend output or fake daemon state.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

from gatedtrain.ext import launch
from gatedtrain.testing.fake_runtime import FakeRuntime


from gatedtrain_pack_vlm_clevr import adapter, scenario
from _images import TIERB_IMAGE
from _pack_fakes import BASE_FILES, CUT_FILES, png, write_base

TESTS = Path(__file__).resolve().parent
WRAPPER = TESTS / "fake_cli.py"
SRC_ENV = "GATEDTRAIN_SRC"
EXIT_OK, EXIT_NOT_PASS = 0, 1
MARGIN = 0.05
KEEPER_IMAGE = "sha256:" + "1" * 64            # the keeper's inference image (held by its fake daemon)
BEFORE_CONFIRM = ("intake_seal", "profile_data", "choose_task_and_base")
TO_TRAIN = ("preregister", "build_train_set", "write_recipe", "smoke")
N_ROWS = 1700
TRIPWIRE = "#!/bin/sh\necho \"$0 $*\" >> '{mark}'\nexit 1\n"


# Variables of the outer environment that would change what the subprocesses run: gatedtrain's ext host overrides and
# the switches and pack variables this test sets itself where it means to.
_OWN_ENV = ("GATEDTRAIN_TEST_BACKEND_RUNTIME", "GATEDTRAIN_TEST_KEEPER_RUNTIME", "GATEDTRAIN_TEST_HOST_VIEW",
            "GATEDTRAIN_PACK_VLM_CLEVR_RUNTIME", scenario.EVAL_PYTHON_ENV, "GATEDTRAIN_PACK_TEST_LABELS",
            "GATEDTRAIN_PACK_TEST_SAMPLER_LOG")


def base_env() -> dict[str, str]:
    """The environment every subprocess starts from, the CLI's and the keeper's alike: this process's without the
    `launch.ENV_PREFIX` variables and the variables in `_OWN_ENV`."""
    return {k: v for k, v in os.environ.items() if not k.startswith(launch.ENV_PREFIX) and k not in _OWN_ENV}


def gatedtrain_src() -> Path | None:
    value = os.environ.get(SRC_ENV)
    return Path(value) if value else None


def load_keeper_harness(src: Path) -> types.ModuleType:
    """gatedtrain's `tests/_keeper.py`, loaded by path with a stand-in for its `_support` import (whose import would
    register gatedtrain's synthetic adapter and set a host config in this process), started through `fake_cli.py`
    instead of gatedtrain's `run_cli.py` (which injects gatedtrain's synthetic pack)."""
    support = types.ModuleType("_support")
    support.FIXTURES = src / "tests" / "fixtures"
    support.cli_env = base_env
    saved = sys.modules.get("_support")
    sys.modules["_support"] = support
    try:
        spec = importlib.util.spec_from_file_location("gatedtrain_test_keeper", src / "tests" / "_keeper.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is None:
            sys.modules.pop("_support", None)
        else:
            sys.modules["_support"] = saved
    mod.RUN_CLI = WRAPPER
    return mod


def uri(i: int) -> str:
    return "data:image/png;base64," + base64.b64encode(png(i)).decode()


def write_data(path: Path) -> None:
    """CLEVR-like intake rows, each question and image distinct and recognisable (the sealed-input scan looks for
    every question and image).
    Every tenth label is 0, which the base's fake answer ("The answer is 0") gets right: the paired differences then
    vary, since a zero-width interval is never PASS (gatedtrain `gates`, design 05 section 3.6)."""
    lines = []
    for i in range(N_ROWS):
        label = 0 if i % 10 == 0 else 1 + i % 9
        text = f"How many cubes are left of the red sphere in scene {i}?"
        lines.append(json.dumps({"input": {"question": f"Q{i:04d}-e2e: {text}", "image": uri(i)}, "label": str(label)}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@unittest.skipUnless(sys.platform == "linux", "the keeper is Linux only (Unix socket, SO_PEERCRED)")
class PackEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        src = gatedtrain_src()
        if src is None:
            if os.environ.get("GATEDTRAIN_REQUIRE_CONTAINER_USER"):         # CI: never go green without the e2e
                self.fail(f"{SRC_ENV} is unset but the end to end is required")
            self.skipTest(f"{SRC_ENV} is unset: the end to end needs gatedtrain's test harness from a source checkout")
        if not (src / "tests" / "fixtures" / "ainative" / "run_cli.py").is_file():
            self.fail(f"{SRC_ENV} does not name a gatedtrain source checkout with tests/fixtures/ainative/run_cli.py")
        if os.getuid() == 0 or os.getgid() in launch.docker_socket_gids():
            why = "the backend runs no container as root or in a group that owns a Docker socket"
            if os.environ.get("GATEDTRAIN_REQUIRE_CONTAINER_USER"):
                self.fail(f"GATEDTRAIN_REQUIRE_CONTAINER_USER is set, but {why}")
            self.skipTest(why)
        keeper_mod = load_keeper_harness(src)

        # Beside this file, not under /tmp: the job's identity mounts (runtime venv, python tree) may not lie under
        # its /tmp tmpfs.
        self.tmp = Path(os.path.realpath(tempfile.mkdtemp(prefix="tmp-job-e2e-", dir=TESTS)))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        os.chmod(self.tmp, 0o755)
        self.work, self.out_root, self.hub = self.tmp / "work", self.tmp / "out", self.tmp / "hub"
        self.eval_tree = self.tmp / "eval-tree"
        for d in (self.work, self.out_root, self.hub, self.eval_tree / "venv" / "bin"):
            d.mkdir(parents=True, exist_ok=True)
        for d in (self.work, self.out_root):
            d.chmod(0o755)
        write_base(self.hub)
        self.eval_python = self.eval_tree / "venv" / "bin" / "python"
        self.eval_python.write_text("fake eval interpreter")
        self.host_file = self.tmp / "host.toml"
        self.host_file.write_text("rootless = true\n")
        self.fake = self.tmp / "fake-backend"
        self.fake.mkdir()
        self.keeper_fake = self.tmp / "fake-keeper"
        self.keeper_fake.mkdir()
        self.mark = self.tmp / "tripwire.log"
        tripwire = self.tmp / "bin"
        tripwire.mkdir()
        for name in ("docker", "nvidia-smi"):
            (tripwire / name).write_text(TRIPWIRE.format(mark=self.mark))
            (tripwire / name).chmod(0o755)
        self.data = self.tmp / "data.jsonl"
        write_data(self.data)
        self.sampler_log = self.tmp / "sampler-log.jsonl"

        path = f"{tripwire}{os.pathsep}{os.environ.get('PATH', os.defpath)}"
        common = {SRC_ENV: str(src), "PATH": path}
        env = base_env()
        env.update(common)
        env.update({launch.ENV_HOST_WORK: str(self.work), launch.ENV_OUTPUT_ROOT: str(self.out_root),
                    launch.ENV_HOST_CONFIG: str(self.host_file),
                    "GATEDTRAIN_TEST_BACKEND_RUNTIME": f"fake:{self.fake}"})
        env["XDG_CONFIG_HOME"] = str(self.tmp / "config")      # the owner's image config is a file of the test
        self.env = env
        self.outputs: list[str] = []
        self._n = 0

        raw = yaml.safe_load(scenario.SCENARIO.pipeline.read_text())
        refs = {s["name"]: s for s in raw["stages"]}
        refs["choose_task_and_base"]["params"]["base_model_dir"] = str(self.hub)
        for name in ("smoke", "train"):
            refs[name]["params"].update(image=TIERB_IMAGE)
        self.pipeline = self.tmp / "pipeline.yaml"
        self.pipeline.write_text(yaml.safe_dump(raw, sort_keys=False))

        sock_dir = self.tmp / "dockerd"
        sock_dir.mkdir(mode=0o700)
        os.chmod(sock_dir, 0o700)
        (sock_dir / "docker.sock").write_bytes(b"")
        self.kp = keeper_mod.KeeperProcess(
            env={**common, "GATEDTRAIN_TEST_KEEPER_RUNTIME": f"fake:{KEEPER_IMAGE}:{self.keeper_fake}",
                 "GATEDTRAIN_PACK_TEST_LABELS": str(self.data),
                 "GATEDTRAIN_PACK_TEST_SAMPLER_LOG": str(self.sampler_log)},
            args=["--inference-image", KEEPER_IMAGE, "--docker-host", f"unix://{sock_dir / 'docker.sock'}",
                  "--eval-tree", str(self.eval_tree), "--eval-python", str(self.eval_python)])
        self.kp.__enter__()
        self.addCleanup(self.kp.__exit__, None, None, None)
        self.addCleanup(self.assert_no_tripwire)

    def assert_no_tripwire(self) -> None:
        self.assertFalse(self.mark.exists(), self.mark.read_text() if self.mark.exists() else "")

    # the command line ---------------------------------------------------------------------------------------------
    def cli(self, *args: str, expect: int | None = EXIT_OK) -> dict:
        proc = subprocess.run([sys.executable, str(WRAPPER), "--json", *args], cwd=self.tmp, env=self.env,
                              capture_output=True, text=True, timeout=300)
        self.outputs += [proc.stdout, proc.stderr]
        where = f"{args!r}\nstdout: {proc.stdout[-3000:]}\nstderr: {proc.stderr[-3000:]}"
        self.assertEqual(proc.stderr, "", where)
        self.assertEqual(proc.stdout.count("\n"), 1, where)
        doc = json.loads(proc.stdout)
        self.assertEqual((doc["schema"], doc["exit"]), ("gatedtrain.cli_result", proc.returncode), where)
        if expect is not None:
            self.assertEqual(proc.returncode, expect, where)
        return doc

    def decision(self, doc: dict) -> str:
        self._n += 1
        path = self.tmp / f"decision-{self._n}.json"
        path.write_text(json.dumps(doc))
        return str(path)

    def stage(self, root: Path, ref: str, decision: dict | None = None) -> dict:
        extra = ["--decision", self.decision(decision)] if decision else []
        return self.cli("project", "run", str(root), ref, *extra)

    @staticmethod
    def read(root: Path, rel: str) -> dict:
        return json.loads((root / rel).read_text())

    def assert_no_sealed_input(self, root: Path) -> None:
        """No input outside the train part in any output, project file (job logs and `jobs/` records included), the
        backend's output root, the host work directory, the fake daemons' state or the keeper's log."""
        rows = [json.loads(line)["input"] for line in self.data.read_text().split("\n") if line]
        every = {x for r in rows for x in (r["question"], r["image"])}
        train = {x for r in self.read(root, "stages/intake_seal/train_part.json")["rows"]
                 for x in (r["input"]["question"], r["input"]["image"])}
        sealed = every - train
        self.assertGreater(len(sealed), 100)
        pattern = re.compile("|".join(re.escape(s) for s in sorted(sealed, key=len, reverse=True)))
        seen = {f"output {i}": t for i, t in enumerate(self.outputs)}
        seen["keeper log"] = self.kp.output()
        files = [p for d in (root, self.out_root, self.fake, self.keeper_fake, self.work) for p in d.rglob("*")]
        self.assertTrue(any(p.name == "attempt-1.log" for p in files))
        for path in files:
            if path.is_file():
                seen[str(path.relative_to(self.tmp))] = path.read_text(encoding="utf-8", errors="replace")
        for where, text in seen.items():
            hit = pattern.search(text)
            self.assertIsNone(hit, f"{where} holds the sealed input {hit and hit.group(0)!r}")

    # the end to end -----------------------------------------------------------------------------------------------
    def test_e2e_fake(self) -> None:
        root = self.tmp / "proj"
        self.cli("project", "create", str(root), "--goal", "clevr answers", "--data", str(self.data),
                 "--pipeline", str(self.pipeline), "--gpu-hours", "4", "--keeper", str(self.kp.socket))
        for ref in BEFORE_CONFIRM:
            self.stage(root, ref, {"accept_inconclusive": True} if ref == "intake_seal" else None)
        metric = {"metric_confirmed": True, "scorer": "vqa_answer", "margin": MARGIN}
        self.cli("keeper", "confirm", str(root), "--decision", self.decision(metric), "--yes")
        for ref in TO_TRAIN:
            self.stage(root, ref, metric if ref == "preregister" else None)
        for ref in ("train", "evaluate_sealed", "gate", "report"):
            self.stage(root, ref)

        status = self.cli("project", "status", str(root))
        self.assertEqual({k: v["status"] for k, v in status["data"]["stages"].items()},
                         {k: "done" for k in status["data"]["stages"]})
        self.assertEqual(len(status["data"]["stages"]), 11)
        self.assertEqual(status["data"]["gate"], "PASS")
        self.assertEqual(self.read(root, "stages/gate/gate_result.json")["status"], "PASS")

        reg = self.read(root, "stages/preregister/registration.json")
        self.assertEqual((reg["scorer"], reg["direction"], reg["margin"]), ("vqa_answer", "higher_is_better", MARGIN))
        base = self.read(root, "stages/choose_task_and_base/base.json")
        model = self.read(root, "stages/train/model.json")
        self.assertEqual((base["generator"], base["label"]), ("vlm_clevr_vllm", "base"))
        self.assertEqual(sorted(Path(f["path"]).name for f in base["files"]),
                         sorted(n for n in BASE_FILES if n != "LICENSE"))
        self.assertEqual((model["generator"], model["label"], model["checkpoint"]),
                         ("vlm_clevr_vllm", "candidate", 60))
        self.assertEqual(sorted(Path(f["path"]).name for f in model["files"]), sorted(CUT_FILES))

        report = self.read(root, "stages/report/report.json")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["compute"]["keeper"]["containers"], 2)
        self.assertEqual(report["compute"]["backend"]["jobs"], 2)

        # The keeper passed the deployment's sampler and seeds to each of its two containers.
        dep = self.read(root, "stages/choose_task_and_base/deployment.json")
        self.assertEqual(dep["eval_seeds"], [11])
        logged = [json.loads(x) for x in self.sampler_log.read_text(encoding="utf-8").split("\n") if x]
        self.assertEqual(len(logged), 2)
        for doc in logged:                  # the registered seed and the keeper's control seed (AN-26), in any order
            self.assertEqual(doc["sampler"], dep["sampler"])
            self.assertEqual(len(doc["seeds"]), 2)
            self.assertIn(11, doc["seeds"])
        self.assertEqual(dep["sampler"], {"name": "vllm", "decoding": "greedy", "temperature": 0, "top_p": 1.0,
                                          "max_new_tokens": 512})

        keeper_runs = FakeRuntime(state_dir=self.keeper_fake).runs
        self.assertEqual(len(keeper_runs), 2)
        for spec in keeper_runs:
            self.assertEqual(spec.argv[:3], (str(self.eval_python), "-I", "/gen/generate_vllm.py"))
            self.assertEqual((spec.network, spec.read_only_root), ("none", True))
            self.assertTrue(spec.gpu_options)
            self.assertIn(str(self.eval_tree), {m.container for m in spec.mounts})
        backend_runs = FakeRuntime(state_dir=self.fake).runs
        self.assertEqual([r.argv[r.argv.index("--profile") + 1] for r in backend_runs], ["smoke", "full"])
        for spec in backend_runs:
            self.assertEqual((spec.image, spec.network), (TIERB_IMAGE, "none"))
            self.assertTrue(spec.gpu_options)

        doc = self.cli("verify", str(root), "--keeper", str(self.kp.socket), expect=EXIT_NOT_PASS)
        self.assertIs(doc["data"]["same_user"], True)
        self.assertEqual(doc["problems"], [])
        self.assert_no_sealed_input(root)


class KeeperEnvTest(unittest.TestCase):
    def test_keeper_starts_from_the_cli_base_env(self) -> None:
        """The keeper harness's `cli_env` stand-in is `base_env`: no ext host override or pack variable of the outer
        environment reaches the keeper."""
        src = gatedtrain_src()
        if src is None:
            self.skipTest(f"{SRC_ENV} is unset")
        mod = load_keeper_harness(src)
        outer = {launch.ENV_PREFIX + "OUTPUT_ROOT": "/elsewhere", "GATEDTRAIN_PACK_VLM_CLEVR_RUNTIME": "/elsewhere",
                 "GATEDTRAIN_TEST_KEEPER_RUNTIME": "fake:x:y", "PACK_TEST_KEEP": "1"}
        with mock.patch.dict(os.environ, outer):
            got = mod.cli_env()
            self.assertEqual(got, base_env())
        self.assertEqual(got.get("PACK_TEST_KEEP"), "1")
        for name in outer:
            if name != "PACK_TEST_KEEP":
                self.assertNotIn(name, got)


class FakeCliGuardTest(unittest.TestCase):
    def test_refuses_without_runtime_switch(self) -> None:
        """Without a runtime switch nothing would keep the real Docker CLI away: the wrapper does not start."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("GATEDTRAIN_TEST_BACKEND_RUNTIME", "GATEDTRAIN_TEST_KEEPER_RUNTIME")}
        proc = subprocess.run([sys.executable, str(WRAPPER), "--json", "project", "status", str(TESTS / "absent")],
                              env=env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertIn("set a runtime switch", proc.stderr)


if __name__ == "__main__":
    unittest.main()
