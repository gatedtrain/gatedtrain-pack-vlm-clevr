"""The training job: a gatedtrain backend `Job` of the ext adapter `vlm-clevr-grpo` (adapter.py).

The job runs the pack's `runtime/run_vlm_nemo.py` (the recipe gatedtrain-recipe-nemo-rl-vlm-clevr's launcher, taking its
training file as an argument) with the NeMo RL image's own interpreter. The image is one of gatedtrain's pinned NeMo RL
images (`gatedtrain.families.nemo_rl.NEMO_RL_IMAGES`, by image ID). The runtime directory (scripts and prompt) is
mounted read-only at its own host path, as the launcher expects; the images of the training rows are mounted read-only
at their own host path too, because the rows name them by absolute path; the training file and the base model are
mounted under /inputs. The output is /out; /tmp is a tmpfs; no network.

`train_job` leaves `output_dir` empty: the caller fills in the backend's new output directory for the job's key
(`dataclasses.replace(job, output_dir=str(backend.new_output_dir(job.label, key=job.key())))`).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping

from gatedtrain.ainative.backend import Job
from gatedtrain.ainative.runtime import Mount
from gatedtrain.errors import Problem, Refused
from gatedtrain.families.nemo_rl import NEMO_RL_COMMIT, NEMO_RL_IMAGES, NEMO_RL_PYTHON

from . import adapter
from .adapter import FRAMEWORK_PACKAGE, NAME, SCRIPT

RULE = "VLM-4"
PACKAGE_DIR = Path(__file__).resolve().parent
FRAMEWORK = {"name": FRAMEWORK_PACKAGE, "version": NEMO_RL_COMMIT, "commit": NEMO_RL_COMMIT}

# The recipe's launch facts (its runtime/README.md "Launch request facts"). TORCH_CUDA_ARCH_LIST 8.9 is the recipe
# host's GPU.
ENV = {
    "FLASHINFER_CUDA_ARCH_LIST": "8.9",
    "HF_HUB_OFFLINE": "1",
    "HOME": "/tmp/home",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "RAY_memory_monitor_refresh_ms": "0",
    "TORCH_CUDA_ARCH_LIST": "8.9",
}
# run_vlm_nemo.py PROFILES: smoke 5 steps, full 60 steps, both 16 prompts x 8 generations per step.
PROFILES = {
    "smoke": {"gpu_hours_est": 0.25, "time_limit_s": 3600},
    "full": {"gpu_hours_est": 1.0, "time_limit_s": 10800},
}
MEMORY_RUNGS = ("a", "b", "c")


EXTRA_IMAGES_ENV = "GATEDTRAIN_PACK_VLM_CLEVR_EXTRA_IMAGES"
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


def accepted_images(environ: Mapping[str, str] = os.environ) -> tuple[str, ...]:
    """The image IDs a job may run in: gatedtrain's pinned NeMo RL images, then the IDs the owner lists (comma
    separated) in `GATEDTRAIN_PACK_VLM_CLEVR_EXTRA_IMAGES`, the locally built derived image of
    `containers/nemo-rl-rootless/` (its ID is host specific). A listed entry that is not an image ID raises
    ValueError; it is never silently dropped."""
    extra = [x.strip() for x in (environ.get(EXTRA_IMAGES_ENV) or "").split(",") if x.strip()]
    bad = [x for x in extra if not _IMAGE_ID.fullmatch(x)]
    if bad:
        raise ValueError(f"{EXTRA_IMAGES_ENV} holds {len(bad)} entries that are not image IDs (sha256:<64 hex>)")
    return tuple(NEMO_RL_IMAGES) + tuple(x for x in extra if x not in NEMO_RL_IMAGES)


def _refused(message: str, fix: str) -> Refused:
    return Refused([Problem(RULE, "job", message, fix)])


def train_job(*, profile: str, image: str, inputs_dir: Path, images_dir: Path, base_dir: Path, seed: int,
              memory_rung: str = "a", runtime_dir: Path | None = None) -> Job:
    """The training job of `profile` (smoke or full). `inputs_dir` holds `train.jsonl`; `images_dir` the images its rows
    name by absolute path; `base_dir` is the base model directory. `runtime_dir` defaults to the adapter's runtime
    directory and must be that directory. ValueError for another profile or memory rung; Refused (VLM-4) for an image
    that is not one of the pinned NeMo RL images or another runtime directory."""
    if profile not in PROFILES:
        raise ValueError(f"profile {profile!r} is not one of {', '.join(PROFILES)}")
    if memory_rung not in MEMORY_RUNGS:
        raise ValueError(f"memory rung {memory_rung!r} is not one of {', '.join(MEMORY_RUNGS)}")
    try:
        allowed = accepted_images()
    except ValueError as exc:
        raise _refused(str(exc), f"fix {EXTRA_IMAGES_ENV}") from None
    if image not in allowed:
        raise _refused(f"the image {image!r} is not one of the pinned NeMo RL images or an image listed in "
                       f"{EXTRA_IMAGES_ENV}", "pass one of " + ", ".join(allowed)
                       + " (the image ID the owner's Docker daemon holds)")
    runtime = adapter.RUNTIME
    if runtime_dir is not None and Path(runtime_dir).resolve() != Path(runtime).resolve():
        raise _refused(f"runtime directory {runtime_dir} is not the adapter's {runtime}, which the adapter row names "
                       "as an input root",
                       f"set {adapter.RUNTIME_ENV}={runtime_dir} in the environment of every gatedtrain process that "
                       "runs smoke or train (the adapter reads it at import)")
    inputs, images = Path(inputs_dir), Path(images_dir)
    return Job(
        label=f"vlm-grpo-{profile}",
        adapter=NAME,
        runtime={"kind": "container", "image": image},
        framework=dict(FRAMEWORK),
        argv=(NEMO_RL_PYTHON, f"{runtime}/{SCRIPT}", "--profile", profile, "--out", "/out", "--seed", str(seed),
              "--base", "/inputs/base-model", "--memory", memory_rung, "--train-file", "/inputs/train.jsonl"),
        env=dict(ENV),
        mounts=(Mount(runtime, runtime), Mount(str(images), str(images)),
                Mount(str(inputs / "train.jsonl"), "/inputs/train.jsonl"),
                Mount(str(Path(base_dir)), "/inputs/base-model")),
        output_dir="",
        output_container="/out",
        tmpfs=("/tmp",),
        network="none",
        gpu=True,
        time_limit_s=PROFILES[profile]["time_limit_s"],
        gpu_hours_est=PROFILES[profile]["gpu_hours_est"],
        training_seed=seed,
        profile=profile,
    )
