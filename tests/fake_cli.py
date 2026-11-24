"""Test-only entry point: gatedtrain's command line with this pack as installed (its real entry points) and
gatedtrain's own test runtime switches, for the CPU end to end (`test_e2e_fake.py`).

The switches come from gatedtrain's test harness, `tests/fixtures/ainative/run_cli.py` of the gatedtrain source
checkout named by `GATEDTRAIN_SRC`: `GATEDTRAIN_TEST_BACKEND_RUNTIME` (the backend's containers on `FakeRuntime`),
`GATEDTRAIN_TEST_KEEPER_RUNTIME` (the keeper's inference containers on `FakeRuntime`) and `GATEDTRAIN_TEST_HOST_VIEW`
(the keeper's rootful-access detection on a fake tree). Either runtime switch also makes the Docker CLI runtime and
the `nvidia-smi` probe raise (`run_cli._forbid_real_tools`), so no code path can start `docker` or `nvidia-smi`.
`run_cli.main` itself is not called: it injects gatedtrain's synthetic pack in place of the installed ones.

The fake runtimes run `gatedtrain.testing.fake_runtime.subprocess_handler` for every container, which would run the
container's script on the host; here it is replaced, before the switches take it, by `handler`, which plays the
script instead (`_pack_fakes`): the pack's training job (`/inputs/train_grpo.py`) or its batch generator
(`/gen/generate_vllm.py`, answering with the labels of the intake data file named by `GATEDTRAIN_PACK_TEST_LABELS`
and appending the sampler document it is given to `GATEDTRAIN_PACK_TEST_SAMPLER_LOG` when that is set).
Any other container command fails the run.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from gatedtrain.testing import fake_runtime

from _pack_fakes import labels_by_key, play_inference, play_training

LABELS_ENV = "GATEDTRAIN_PACK_TEST_LABELS"
SAMPLER_LOG_ENV = "GATEDTRAIN_PACK_TEST_SAMPLER_LOG"     # where the batch generator's play logs its sampler documents
SWITCHES = (("GATEDTRAIN_TEST_BACKEND_RUNTIME", "_fake_backend_runtime"),
            ("GATEDTRAIN_TEST_HOST_VIEW", "_fake_host_view"),
            ("GATEDTRAIN_TEST_KEEPER_RUNTIME", "_fake_keeper_runtime"))


def load_run_cli(src: str):
    path = Path(src) / "tests" / "fixtures" / "ainative" / "run_cli.py"
    spec = importlib.util.spec_from_file_location("gatedtrain_test_run_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def handler(run_spec, host_path) -> int:
    argv = list(run_spec.argv)
    if any(a.endswith("/run_vlm_nemo.py") for a in argv):
        return play_training(run_spec, host_path)
    if "/gen/generate_vllm.py" in argv:
        log = os.environ.get(SAMPLER_LOG_ENV)
        if log:
            import json
            arg = argv[argv.index("--sampler") + 1]
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps(json.loads(host_path(arg).read_text()), sort_keys=True) + "\n")
        return play_inference(run_spec, host_path, labels_by_key(Path(os.environ[LABELS_ENV])))
    print("fake_cli: unexpected container command")
    return 99


def main(argv: list[str]) -> int:
    if not (os.environ.get(SWITCHES[0][0]) or os.environ.get(SWITCHES[2][0])):
        raise SystemExit("fake_cli: set a runtime switch (GATEDTRAIN_TEST_BACKEND_RUNTIME or "
                         "GATEDTRAIN_TEST_KEEPER_RUNTIME); without one the real Docker CLI could run")
    run_cli = load_run_cli(os.environ["GATEDTRAIN_SRC"])
    fake_runtime.subprocess_handler = handler          # before a switch binds it
    for env, switch in SWITCHES:
        if os.environ.get(env):
            getattr(run_cli, switch)(os.environ[env])
    from gatedtrain import cli

    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
