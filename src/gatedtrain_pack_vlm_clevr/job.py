"""The training job: a gatedtrain backend `Job` of the ext adapter `vlm-clevr-grpo` (adapter.py).

The job runs the pack's `runtime/run_vlm_nemo.py` (the recipe gatedtrain-recipe-nemo-rl-vlm-clevr's launcher, taking its
training file as an argument) with the NeMo RL image's own interpreter. The image is one of gatedtrain's pinned NeMo RL
images (the accepted IDs of gatedtrain's image lock for the logical names `nemo-rl/tierb`, `nemo-rl/tierb-uid1000` and
`nemo-rl/tierb-rootless`, plus the images the owner recorded with `gatedtrain image add`). The runtime directory (scripts and prompt) is
mounted read-only at its own host path, as the launcher expects; the images of the training rows are mounted read-only
at their own host path too, because the rows name them by absolute path; the training file and the base model are
mounted under /inputs. The output is /out; /tmp is a tmpfs; no network.

`train_job` leaves `output_dir` empty: the caller fills in the backend's new output directory for the job's key
(`dataclasses.replace(job, output_dir=str(backend.new_output_dir(job.label, key=job.key())))`).
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Mapping

from gatedtrain.ainative.backend import Job
from gatedtrain.ainative.runtime import Mount
from gatedtrain.errors import Problem, Refused
from gatedtrain.families import nemo_rl
from gatedtrain.families.nemo_rl import NEMO_RL_COMMIT, NEMO_RL_PYTHON

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


# The logical NeMo RL images (gatedtrain's image lock) a job may run in. The IDs are data in gatedtrain's lock and in the
# owner's own image config (`gatedtrain image add`); none is written in this pack.
LOGICAL_IMAGES = nemo_rl.LOGICAL_IMAGES

# DEPRECATED workaround, named and recorded (engineering standards rule 4). Before gatedtrain 0.7 the pack's 0.2.0
# README told the owner of a rootless host to list the ID of the locally built derived image in this variable. The
# supported way now is `gatedtrain image add nemo-rl/tierb-rootless ...`. The variable is kept for this release only so
# that the documented 0.2.0 workflow does not stop working at once; it is read, warned about at every use, and removed
# in pack release EXTRA_IMAGES_REMOVAL. Owner: the pack maintainer; plan: delete the variable, `_extra_images` and their
# tests in that release.
EXTRA_IMAGES_ENV = "GATEDTRAIN_PACK_VLM_CLEVR_EXTRA_IMAGES"
EXTRA_IMAGES_REMOVAL = "0.4.0"
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


def _extra_images(environ: Mapping[str, str]) -> list[str]:
    extra = [x.strip() for x in (environ.get(EXTRA_IMAGES_ENV) or "").split(",") if x.strip()]
    if not extra:
        return []
    bad = [x for x in extra if not _IMAGE_ID.fullmatch(x)]
    if bad:
        raise ValueError(f"{EXTRA_IMAGES_ENV} holds {len(bad)} entries that are not image IDs (sha256:<64 hex>)")
    warnings.warn(f"{EXTRA_IMAGES_ENV} is deprecated and will be removed in gatedtrain-pack-vlm-clevr "
                  f"{EXTRA_IMAGES_REMOVAL}: record the image with `gatedtrain image add nemo-rl/tierb-rootless ...` "
                  "instead (see the README)", FutureWarning, stacklevel=3)
    return extra


def accepted_images(environ: Mapping[str, str] = os.environ) -> tuple[str, ...]:
    """The image IDs a job may run in: gatedtrain's accepted images of the three logical NeMo RL names (its image
    lock, then the owner's `gatedtrain image add` entries; an untrustworthy owner config raises gatedtrain's Refused
    IMG-3, it is never ignored), then, deprecated, the IDs listed (comma separated) in
    `GATEDTRAIN_PACK_VLM_CLEVR_EXTRA_IMAGES`. A listed entry that is not an image ID raises ValueError; it is never
    silently dropped."""
    out: list[str] = []
    for logical in LOGICAL_IMAGES:
        out += [i for i in nemo_rl.accepted_images(logical) if i not in out]
    out += [i for i in _extra_images(environ) if i not in out]
    return tuple(out)


def require_accepted_image(image: str, field: str = "job", environ: Mapping[str, str] = os.environ) -> None:
    """Refuse (VLM-4) an image that is not accepted; the fix names `gatedtrain image add`."""
    try:
        allowed = accepted_images(environ)
    except ValueError as exc:
        raise Refused([Problem(RULE, field, str(exc), f"fix {EXTRA_IMAGES_ENV}")]) from None
    if image not in allowed:
        raise Refused([Problem(RULE, field, f"the image {image!r} is not an accepted NeMo RL image (gatedtrain's image "
                               "lock or the owner's image config)",
                               "pass one of " + ", ".join(allowed) + "; for an image you built, record it with "
                               "`gatedtrain image add nemo-rl/tierb-rootless --id sha256:... --recipe NAME "
                               "--recipe-commit COMMIT`")])


def _refused(message: str, fix: str) -> Refused:
    return Refused([Problem(RULE, "job", message, fix)])


def train_job(*, profile: str, image: str, inputs_dir: Path, images_dir: Path, base_dir: Path, seed: int,
              memory_rung: str = "a", runtime_dir: Path | None = None) -> Job:
    """The training job of `profile` (smoke or full). `inputs_dir` holds `train.jsonl`; `images_dir` the images its rows
    name by absolute path; `base_dir` is the base model directory. `runtime_dir` defaults to the adapter's runtime
    directory and must be that directory. ValueError for another profile or memory rung; Refused (VLM-4) for an image
    that is not an accepted NeMo RL image or another runtime directory."""
    if profile not in PROFILES:
        raise ValueError(f"profile {profile!r} is not one of {', '.join(PROFILES)}")
    if memory_rung not in MEMORY_RUNGS:
        raise ValueError(f"memory rung {memory_rung!r} is not one of {', '.join(MEMORY_RUNGS)}")
    require_accepted_image(image)
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
