"""The CLEVR pack's scenario (the `gatedtrain.scenarios` entry point `vlm-clevr`): its batch generator, its scorer
and its pipeline file.

- `generators` is empty: the pack has no per-item generator, so a project of this pack needs a keeper, which runs the
  batch generator `vlm_clevr_vllm` (`BATCH`, gatedtrain AN-25) in its inference container.
- `BATCH` runs `generate_vllm.py` with the python of the keeper's root-owned eval venv, mounted from the keeper's eval
  trees, on the GPU. The interpreter is `bin/python` of the eval venv `gatedtrain setup host` builds
  (`gatedtrain.ainative.hostsetup.DEFAULT_EVAL_VENV`) unless the environment variable
  `GATEDTRAIN_PACK_VLM_CLEVR_EVAL_PYTHON` names another absolute path when this module is imported. Only the
  environment of the keeper's own process counts: the keeper loads its own copy of the scenario and runs the generator
  from it, whatever the owner's shell sets. The keeper still runs the generator only when that interpreter is a
  regular file inside one of its configured eval trees (`keeper serve --eval-tree`), so the variable can move the eval
  venv, never take it out of those trees.
- `scorers` registers `vqa_answer` by reference: `answer.score`, the sha256 of `answer.py` (the file that defines
  it, which imports only the standard library), and `answer.label_ok`.
- `pipeline` is `pipeline.yaml` beside this file.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Mapping

from gatedtrain.ainative.batch import DEFAULT_TIME_LIMIT_S, BatchGenerator
from gatedtrain.ainative.hostsetup import DEFAULT_EVAL_VENV
from gatedtrain.canonical import file_sha256

from . import answer

PACKAGE_DIR = Path(__file__).resolve().parent
NAME = "vlm-clevr"
GENERATOR = "vlm_clevr_vllm"
SCORER = "vqa_answer"
EVAL_PYTHON_ENV = "GATEDTRAIN_PACK_VLM_CLEVR_EVAL_PYTHON"
# The eval venv's python where `gatedtrain setup host` builds it.
DEFAULT_EVAL_PYTHON = str(PurePosixPath(DEFAULT_EVAL_VENV) / "bin" / "python")
TIME_LIMIT_S = DEFAULT_TIME_LIMIT_S


def eval_python(environ: Mapping[str, str] = os.environ) -> str:
    """The container path of the eval venv's python: `GATEDTRAIN_PACK_VLM_CLEVR_EVAL_PYTHON` when set and not
    empty, else `DEFAULT_EVAL_PYTHON`. A value that is not an absolute path raises ValueError (the pack then fails
    to load, and gatedtrain refuses it, AN-8). The variable is a deprecated workaround, removed in pack 0.4.0: the keeper's
    own `--eval-python` option (gatedtrain 0.7.1) replaces the interpreter of an eval-tree generator and is part of the
    keeper's signed environment record. A set variable prints a notice on stderr."""
    if environ.get(EVAL_PYTHON_ENV):
        print(f"warning: {EVAL_PYTHON_ENV} is deprecated: start the keeper with `--eval-python` instead; the variable is "
              "removed in pack 0.4.0", file=sys.stderr)
    value = environ.get(EVAL_PYTHON_ENV) or DEFAULT_EVAL_PYTHON
    if not PurePosixPath(value).is_absolute():
        raise ValueError(f"{EVAL_PYTHON_ENV} must be an absolute path")
    return value


BATCH = BatchGenerator(script=PACKAGE_DIR / "generate_vllm.py", interpreter=eval_python(), gpu=True,
                       time_limit_s=TIME_LIMIT_S, eval_trees=True)

SCORERS = {SCORER: {
    "id": "vqa_answer@0.1",
    "fn": answer.score,
    "code_sha256": file_sha256(Path(answer.__file__).resolve()),
    "planning_sd": 0.5,
    "direction": "higher_is_better",
    "range": [0, 1],
    "what": "canonical answer (number or yes/no) equals the label",
    "unit": "exact match",
    "label_ok": answer.label_ok,
}}


@dataclass(frozen=True)
class Scenario:
    """What the `gatedtrain.scenarios` entry point exposes (the fields gatedtrain reads): the per-item generators by
    name, the pipeline file, the scorers by name and the batch generators by name. The mappings are read-only."""

    name: str
    generators: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    pipeline: Path | None = None
    scorers: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    batch_generators: Mapping[str, BatchGenerator] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "generators", MappingProxyType(dict(self.generators)))
        object.__setattr__(self, "batch_generators", MappingProxyType(dict(self.batch_generators)))
        object.__setattr__(self, "scorers", MappingProxyType({k: MappingProxyType(dict(v))
                                                              for k, v in self.scorers.items()}))


SCENARIO = Scenario(NAME, {}, PACKAGE_DIR / "pipeline.yaml", SCORERS, {GENERATOR: BATCH})
