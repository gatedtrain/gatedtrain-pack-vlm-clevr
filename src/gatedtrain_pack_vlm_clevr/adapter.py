"""Ext adapter `vlm-clevr-grpo`: the launch rules of the pack's NeMo RL VLM GRPO training job (`job.train_job`).

gatedtrain's local backend checks every job against the launch rules of the ext adapter it names
(`gatedtrain.ainative.backend.job_problems`). This module registers that adapter through the `gatedtrain.adapters`
entry point (or in process with `install()`): the training row, modelled on the row of the recipe
gatedtrain-recipe-nemo-rl-vlm-clevr, an evaluator row for the pack's scorer, and a small adapter class.

The row differs from the recipe's in four ways: no spec binding (the pack's stages bind the inputs), the trainer id
`rl.nemo_rl_vlm@0.1` is the recipe's (the same trainer), the input roots are this package's `runtime/` (scripts and
prompt, mounted read-only at its own host path, so the path inside the container is the host path) and the whole
`{output_root}` (the job's inputs, the images and the base model live under `{output_root}/gatedtrain-<hex>/`), and
the evaluator is the pack's. Every script runs with an interpreter of the pinned NeMo RL image; there is no host
interpreter. `GATEDTRAIN_PACK_VLM_CLEVR_RUNTIME` names another runtime directory (it must hold the same scripts).

Runs of this adapter are read by the pack's own stages; `gatedtrain ext-collect` does not read them, so the class's
run-reading methods refuse. No framework is imported here (MM-12).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, NoReturn

from gatedtrain.errors import Problem, Refused
from gatedtrain.ext.rowkit import CONTAINER_PATHS, runtime_dir
from gatedtrain.families.nemo_rl import NEMO_RL_COMMIT, NEMO_RL_PYTHON, NEMO_RL_RUN_OPTIONS

NAME = "vlm-clevr-grpo"
TRAINER_ID = "rl.nemo_rl_vlm@0.1"
EVALUATOR_ID = "vqa.vqa_answer@0.1"
FRAMEWORK_PACKAGE = "nemo_rl"
PACK = "vlm-clevr"                   # the name of this pack's `[packs.<name>]` table in the owner's host file
# The runtime directory: the host file's `[packs.vlm-clevr] runtime_dir`, else this package's `runtime/`. The variable
# below is a deprecated workaround (it prints a notice naming the file key) that goes away with gatedtrain 0.9.0.
RUNTIME_ENV = "GATEDTRAIN_PACK_VLM_CLEVR_RUNTIME"
RUNTIME = runtime_dir(RUNTIME_ENV, __file__, parents=0, pack=PACK)
OUTPUT_ROOT = "{output_root}"
SCRIPT = "run_vlm_nemo.py"
GPU_H_ESTIMATE = {"smoke": 0.25, "full": 1.0}      # GPU-hours per profile: the registry row and the jobs use these
# gatedtrain's NEMO_RL_RUN_OPTIONS minus `--ulimit memlock=-1`: a rootless Docker daemon cannot raise the locked-memory
# limit above its own hard limit, and the container then fails to start ("error setting rlimit type 8"). The
# container keeps the daemon's limit.
_MEMLOCK = NEMO_RL_RUN_OPTIONS.index("memlock=-1")
RUN_OPTIONS = NEMO_RL_RUN_OPTIONS[:_MEMLOCK - 1] + NEMO_RL_RUN_OPTIONS[_MEMLOCK + 1:]

TRAINER_ROWS: dict[str, dict[str, Any]] = {NAME: {
    "uses": (TRAINER_ID,),
    "pins": (),
    "nemo_rl_commit": NEMO_RL_COMMIT,
    "outputs": ("cuts/checkpoint-<step>/ full-weight cuts with cut.json written last", "run_args.json",
                "env-freeze.txt", "env_calls.jsonl", "nemo/logs/exp_<n>/train_data_step<N>.jsonl"),
    "components": ("*.safetensors", "model.safetensors.index.json", "config.json", "tokenizer.json",
                   "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json",
                   "video_preprocessor_config.json", "vocab.json", "merges.txt", "sampler.json"),
    "consumption": {"target": "row", "fallback": None},
    "gpu_h_estimate": GPU_H_ESTIMATE["full"],        # the full profile (job.PROFILES)
    "evaluators": (EVALUATOR_ID,),
    "framework": FRAMEWORK_PACKAGE,
    "input_roots": (RUNTIME, OUTPUT_ROOT),
    "spec_binding": False,
    "env_names": ("FLASHINFER_CUDA_ARCH_LIST", "HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE",
                  "PYTHONUNBUFFERED", "RAY_memory_monitor_refresh_ms", "TORCH_CUDA_ARCH_LIST"),
    "scan_skip": (),
    "interpreter_trees": (),
    "interpreter": NEMO_RL_PYTHON,
    "image_interpreters": {SCRIPT: NEMO_RL_PYTHON},
    "container_paths": CONTAINER_PATHS,
    "env_paths": ("HOME",),
    "run_options": {SCRIPT: RUN_OPTIONS},
}}

# The pack's scorer (answer.py `score`): 1 when the response's canonical answer equals the label's, else 0.
EVALUATOR_ROWS: dict[str, dict[str, Any]] = {EVALUATOR_ID: {
    "metric": "exact_match",
    "direction": "higher_is_better",
    "score_range": (0, 1),
    "scorer": {"name": "vqa_answer", "version": "gatedtrain-pack-vlm-clevr@0.1", "weights_sha256": None,
               "code_sha256": "sha256:" + hashlib.sha256((Path(__file__).parent / "answer.py").read_bytes()).hexdigest()},
    "tier": "T3",                                    # not the GRPO training reward (NeMo RL's VLM environment)
    "binary": True,
    "error_classes": ("missing_score",),
    "score_outcomes": ("wrong_answer", "no_answer"),
    "payload_text_field": "question",
    "item_payload": ("question", "answer"),
    "normalizer": False,
    "env_version": False,
    "framework": "gatedtrain-pack-vlm-clevr",
    "samples": "items n, 1 seed",
}}

PLUGIN = {"adapters": {NAME: "gatedtrain_pack_vlm_clevr.adapter:VlmClevrGrpoAdapter"},
          "trainer_rows": TRAINER_ROWS, "evaluator_rows": EVALUATOR_ROWS}


def install() -> None:
    """Register the plugin in this process when the entry point has not (idempotent)."""
    from gatedtrain import spec
    from gatedtrain.ext import registry

    if registry.ADAPTERS.get(NAME) != PLUGIN["adapters"][NAME]:
        registry.register(PLUGIN, "gatedtrain_pack_vlm_clevr.adapter.PLUGIN")
    spec._external_ids()


def _not_collected(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise Refused([Problem("EXT-12", "adapter", f"runs of the ext adapter {NAME} are read by the "
                           "gatedtrain-pack-vlm-clevr stages, not by gatedtrain ext-collect",
                           "run the pack's pipeline (gatedtrain's AI-native commands)")])


class VlmClevrGrpoAdapter:
    """The adapter interface (MM-12) for the pack's training job: its identity and metric; the run-reading methods
    refuse (module docstring)."""

    name = NAME
    trainer_id = TRAINER_ID
    evaluator_ids = (EVALUATOR_ID,)
    framework_name = FRAMEWORK_PACKAGE
    binds_spec = False
    checkpoint_completeness = {
        "library": "NeMo RL (the recipe's cut writer publishes cuts/checkpoint-<N>/ with cut.json last)",
        "rule": "not claimed: ext-collect does not read this adapter's runs",
        "complete_dir_pattern": r"^checkpoint-(\d+)$",
        "sources": [],
        "verified": False,
    }
    seed_derivation = ("the job's training_seed is passed to run_vlm_nemo.py --seed, NeMo RL grpo.seed "
                       "(set_seed in setup); vLLM sampling is seeded by NeMo RL")

    def metric(self, evaluator_id: str) -> dict[str, Any]:
        if evaluator_id not in self.evaluator_ids:
            raise KeyError(evaluator_id)
        row = EVALUATOR_ROWS[evaluator_id]
        return {"metric": row["metric"], "direction": row["direction"], "score_range": tuple(row["score_range"]),
                "error_classes": tuple(row["error_classes"]), "score_outcomes": tuple(row["score_outcomes"]),
                "scorer": dict(row["scorer"]), "binary": row["binary"], "tier": row["tier"]}

    identity = model_dir_problem = checkpoint_dirs = find_cuts = cut_problem = checks = _not_collected
    consumption = framework_identity = script_problems = mount_binding_problems = _not_collected
