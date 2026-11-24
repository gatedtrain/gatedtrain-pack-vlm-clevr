"""Stages of the CLEVR scenario pack. Every stage's registry name is prefixed `clevr_`.

`clevr_choose_task_and_base` copies the base model's files into a backend output directory; `clevr_build_train_set`
decodes the training part's inline images into files and writes the recipe's `train.jsonl`; `clevr_smoke` and
`clevr_train` run the pack's `runtime/run_vlm_nemo.py` as GPU jobs of the project's backend (`ctx.backend`, job.py
`train_job`): smoke applies the MM-38 checks to NeMo RL's environment log, train hands the last cut's deployable files
off as the candidate. Pack refusals use the rules `VLM-<n>`: VLM-1 data rows, VLM-2 base model, VLM-3 training
output, VLM-4 job parameters (as job.py), VLM-5 recipe; smoke refusals MM-38.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

from gatedtrain.ainative.project import ArtifactRef
from gatedtrain.ainative.stage import Proposal, StageContext
from gatedtrain.ainative.trust_stages import SCHEMA_DEPLOYMENT, SCHEMA_MODEL_REF, SCHEMA_TRAIN_PART
from gatedtrain.errors import Problem, Refused

from . import job as job_mod
from .answer import canonical_label, label_ok

SCHEMA_VERSION = 1
SCHEMA_PROFILE = "gatedtrain_pack_vlm_clevr.profile"
SCHEMA_TRAIN_SET = "gatedtrain_pack_vlm_clevr.train_set"
SCHEMA_RECIPE = "gatedtrain_pack_vlm_clevr.recipe"
SCHEMA_SMOKE = "gatedtrain_pack_vlm_clevr.smoke"
PERCENTILES = (("p50", 50), ("p90", 90), ("p99", 99))

GENERATOR = "vlm_clevr_vllm"            # the batch generator the scenario registers
ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
# A base model directory must hold these (the processor files vLLM's AutoProcessor reads included); every file of the
# directory with an allowed suffix is copied, the rest (LICENSE) is not.
REQUIRED_BASE = ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
                 "video_preprocessor_config.json", "chat_template.jinja")
MODEL_SUFFIXES = (".safetensors", ".json", ".txt", ".jinja")    # the keeper's MODEL_FILE_SUFFIXES
# gatedtrain's registration refuses a temperature of 0 (registration.py "sampler.temperature"), so the deployment
# samples at 0.7 over four seeds, as the text pack; the recipe's own gate was greedy with one seed.
SAMPLER = {"name": "vllm", "temperature": 0.7, "top_p": 1.0, "max_new_tokens": 512}
EVAL_SEEDS = [11, 12, 13, 14]
# run_vlm_nemo.py PROFILES (steps); `train_job` passes no --max-steps, so these are the only steps a job runs.
PROFILE_STEPS = {"smoke": 5, "full": 60}
TRAINING_SEED = 7100                    # the recipe's seed
MEMORY_RUNG = "a"
JOB_POLL_S = 30.0
MIN_SAMPLES = 8
PROMPTS_PER_STEP = 16                   # run_vlm_nemo.py PROFILES; one epoch (NeMo RL's max_num_epochs 1) of the rows
# The prompt budget: Qwen3.5 reads one image token per 32 x 32 pixels, the sequence limit is 1,024 tokens of which
# about 110 are text and up to 512 the answer, so an image may use about 400 tokens (about 400,000 pixels).
MAX_PIXELS = 400_000
MAX_LOG_BYTES = 64 * 1024 * 1024
DATA_URI = re.compile(r"data:image/(png|jpeg);base64,")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
BASE_DIR = "base-model"
DATA_DIR = "train-data"
_CHECKPOINT = re.compile(r"checkpoint-([0-9]+)")
_CHUNK = 1 << 20
_MAX_JSON = 1 << 20
_PLAIN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _refuse(rule: str, field_: str, message: str, fix: str) -> Refused:
    return Refused([Problem(rule, field_, message, fix)])


def _percentile(sorted_vals: list[int], pct: int) -> int:
    rank = -(-pct * len(sorted_vals) // 100)
    return sorted_vals[max(rank, 1) - 1]


def _meta(schema: str) -> dict:
    return {"schema": schema, "schema_version": SCHEMA_VERSION}


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def decode_image(uri: object) -> tuple[str, bytes] | None:
    """(extension, bytes) of a PNG or JPEG data URI whose bytes carry the matching magic number, else None."""
    if not isinstance(uri, str):
        return None
    m = DATA_URI.match(uri)
    if m is None:
        return None
    try:
        raw = base64.b64decode(uri[m.end():], validate=True)
    except (binascii.Error, ValueError):
        return None
    ext = m.group(1)
    if not raw.startswith(PNG_MAGIC if ext == "png" else JPEG_MAGIC):
        return None
    return ("jpg" if ext == "jpeg" else "png"), raw


def image_size(raw: bytes) -> tuple[int, int] | None:
    """(width, height) read from a PNG header or a JPEG start-of-frame marker, else None. Nothing is decoded."""
    if raw.startswith(PNG_MAGIC):
        if len(raw) < 24 or raw[12:16] != b"IHDR":
            return None
        return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    if raw.startswith(JPEG_MAGIC):
        i = 2
        while i + 9 < len(raw):
            if raw[i] != 0xFF:
                return None
            marker = raw[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            length = int.from_bytes(raw[i + 2:i + 4], "big")
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                return int.from_bytes(raw[i + 7:i + 9], "big"), int.from_bytes(raw[i + 5:i + 7], "big")
            i += 2 + length
    return None


def _size_ok(raw: bytes) -> bool:
    size = image_size(raw)
    return size is not None and size[0] > 0 and size[1] > 0 and size[0] * size[1] <= MAX_PIXELS


def checked_rows(rows: object, field_: str) -> list[dict]:
    """The training rows, each `{input: {question, image}, label}` with a canonical label and a decodable image.
    Refused (VLM-1) naming row indexes only."""
    if not (isinstance(rows, list) and rows):
        raise _refuse("VLM-1", field_, "the training rows are missing or empty", "rerun the stage that writes the "
                      "train part")
    shape = [i for i, r in enumerate(rows) if not (
        isinstance(r, dict) and isinstance(r.get("input"), dict) and isinstance(r["input"].get("question"), str)
        and isinstance(r["input"].get("image"), str) and "label" in r)]
    if shape:
        raise _refuse("VLM-1", field_, f"{len(shape)} of {len(rows)} rows are not {{input: {{question, image}}, "
                      f"label}} (first row index from 0: {shape[0]})",
                      "give the data as the pack's converter writes it (gatedtrain-pack-vlm-clevr-data)")
    bad = [i for i, r in enumerate(rows) if not label_ok(r["label"])]
    if bad:
        raise _refuse("VLM-1", field_, f"{len(bad)} of {len(rows)} labels are not an integer, number word or yes/no "
                      f"(first row index from 0: {bad[0]})", "fix the labels in the source data and intake it again")
    undecodable = [i for i, r in enumerate(rows) if decode_image(r["input"]["image"]) is None]
    if undecodable:
        raise _refuse("VLM-1", field_, f"{len(undecodable)} of {len(rows)} images are not PNG or JPEG data URIs "
                      f"(first row index from 0: {undecodable[0]})", "convert the data with "
                      "gatedtrain-pack-vlm-clevr-data and intake it again")
    oversize = [i for i, r in enumerate(rows) if not _size_ok(decode_image(r["input"]["image"])[1])]
    if oversize:
        raise _refuse("VLM-1", field_, f"{len(oversize)} of {len(rows)} images have unreadable dimensions or more than "
                      f"{MAX_PIXELS} pixels (first row index from 0: {oversize[0]})",
                      "convert the data with gatedtrain-pack-vlm-clevr-data, which scales images down")
    return rows


class ProfileData:
    name = "clevr_profile_data"
    version = "0.1"
    inputs = {"train_part": SCHEMA_TRAIN_PART}
    outputs = {"profile": SCHEMA_PROFILE}
    needs: frozenset[str] = frozenset()
    params: frozenset[str] = frozenset()
    trains = False

    def _profile(self, ctx: StageContext) -> dict:
        rec = ctx.project.read_artifact(ctx.inputs["train_part"])
        rows = checked_rows(rec.get("rows") if isinstance(rec, dict) else None, "inputs.train_part.rows")
        labels: dict[str, int] = {}
        sizes, qlens, keys, images = [], [], set(), set()
        for r in rows:
            lab = canonical_label(r["label"])
            labels[lab] = labels.get(lab, 0) + 1
            _, raw = decode_image(r["input"]["image"])
            sha = hashlib.sha256(raw).hexdigest()
            images.add(sha)
            keys.add((sha, r["input"]["question"]))
            sizes.append(len(raw))
            qlens.append(len(r["input"]["question"]))
        sizes.sort()
        qlens.sort()
        return {**_meta(SCHEMA_PROFILE), "n": len(rows), "n_distinct_images": len(images),
                "n_duplicate_keys": len(rows) - len(keys), "labels": dict(sorted(labels.items())),
                "image_bytes": {"min": sizes[0], "max": sizes[-1], **{k: _percentile(sizes, p) for k, p in PERCENTILES}},
                "question_chars": {"min": qlens[0], "max": qlens[-1],
                                   **{k: _percentile(qlens, p) for k, p in PERCENTILES}}}

    def plan(self, ctx: StageContext) -> Proposal:
        p = self._profile(ctx)
        return Proposal(self.name, f"profile {p['n']} training rows on {p['n_distinct_images']} images: "
                        f"{len(p['labels'])} distinct answers, question length p50 {p['question_chars']['p50']} chars",
                        {}, {"n": p["n"]})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        return {"profile": ctx.project.write_artifact(ctx.stage or self.name, "profile", self._profile(ctx),
                                                      SCHEMA_PROFILE)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        p = ctx.project.read_artifact(outputs["profile"])
        n = ctx.project.read_artifact(ctx.inputs["train_part"]).get("rows")
        if not isinstance(p, dict) or p.get("n") != (len(n) if isinstance(n, list) else None):
            return [Problem("VLM-1", "profile.n", "the row count differs from the train part", "rerun clevr_profile_data")]
        return []


# ---------------------------------------------------------------------------------------------------------------------
# File helpers (as the text pack: hand-off files are mode 0644, replaced atomically, never through a link)
# ---------------------------------------------------------------------------------------------------------------------

def _read(ctx: StageContext, name: str, rule: str) -> dict:
    rec = ctx.project.read_artifact(ctx.inputs[name])
    if not isinstance(rec, dict):
        raise _refuse(rule, f"inputs.{name}", "the input record is not an object", "rerun the stage that writes it")
    return rec


def _sha256_path(path: Path | str, *, follow: bool = True) -> str | None:
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | (0 if follow else os.O_NOFOLLOW))
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _same(path: Path, sha256: str) -> bool:
    try:
        st = os.lstat(path)
        return (stat.S_ISREG(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o644
                and _sha256_path(path, follow=False) == sha256)
    except OSError:
        return False


def _replace(dst: Path, write) -> str:
    tmp = dst.with_name(f".{dst.name}.tmp")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    h = hashlib.sha256()
    with os.fdopen(fd, "wb") as f:
        write(f, h)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o644)
    os.replace(tmp, dst)
    return "sha256:" + h.hexdigest()


def _write_bytes(dst: Path, data: bytes) -> None:
    """`dst` holds `data` (mode 0644); an identical file is left as it is (same inode, so a job's key is stable)."""
    if _same(dst, _sha256_bytes(data)):
        return

    def write(f, h):
        f.write(data)
        h.update(data)
    _replace(dst, write)


def _copy(src: Path, dst: Path) -> str:
    want = _sha256_path(src)
    if want is None:
        raise _refuse("VLM-2", "params.base_model_dir", f"{src.name} is not a regular file",
                      "point base_model_dir at the downloaded model directory")
    if _same(dst, want):
        return want

    def write(f, h):
        with open(src, "rb") as fi:
            for chunk in iter(lambda: fi.read(_CHUNK), b""):
                f.write(chunk)
                h.update(chunk)
    got = _replace(dst, write)
    if got != want:
        raise _refuse("VLM-2", "params.base_model_dir", f"{src.name} changed while it was copied",
                      "leave the base model directory unchanged and run the stage again")
    return got


def _backend(ctx: StageContext, stage: str) -> Any:
    if ctx.backend is None:
        raise _refuse("VLM-4", "backend", f"stage {stage!r} needs the project's job backend (ctx.backend)",
                      "run the stage through the stage machine")
    return ctx.backend


def _model_dir(ref: dict, field_: str, rule: str, fix: str) -> Path:
    """The one absolute directory holding the files a model reference lists, with hashes; the files include
    `config.json`."""
    files = ref.get("files")
    ok = isinstance(files, list) and files and all(
        isinstance(f, dict) and isinstance(f.get("path"), str) and isinstance(f.get("sha256"), str) for f in files)
    names = [Path(f["path"]).name for f in files] if ok else []
    if not ok or "config.json" not in names or len(set(names)) != len(names):
        raise _refuse(rule, f"{field_}.files", "the model reference does not list the model's files with their hashes",
                      fix)
    dirs = {os.path.dirname(f["path"]) for f in files}
    if len(dirs) != 1 or not os.path.isabs(next(iter(dirs))):
        raise _refuse(rule, f"{field_}.files", "the model files do not lie in one absolute directory", fix)
    return Path(next(iter(dirs)))


def _changed_files(ref: dict) -> list[str]:
    bad = []
    for f in ref["files"]:
        try:
            ok = _sha256_path(f["path"], follow=False) == f["sha256"]
        except OSError:
            ok = False
        if not ok:
            bad.append(Path(f["path"]).name)
    return bad


# ---------------------------------------------------------------------------------------------------------------------
# Base model and deployment
# ---------------------------------------------------------------------------------------------------------------------

class ChooseTaskAndBase:
    """Task: CLEVR answers (a number or yes/no). Base: Qwen3.5-0.8B from `params.base_model_dir` (a model directory
    downloaded by the owner at the pinned commit); every file of it with an allowed suffix is copied (mode 0644) into
    the backend output directory `base-model`, where the keeper user can read them, and listed with its hash in the
    `base` model reference."""

    name = "clevr_choose_task_and_base"
    version = "0.1"
    inputs = {"profile": SCHEMA_PROFILE}
    outputs = {"deployment": SCHEMA_DEPLOYMENT, "base": SCHEMA_MODEL_REF}
    needs: frozenset[str] = frozenset()
    params = frozenset({"base_model_dir"})
    trains = False

    def _source(self, ctx: StageContext) -> tuple[Path, list[str]]:
        d = ctx.params.get("base_model_dir")
        if not (isinstance(d, str) and os.path.isabs(d) and os.path.isdir(d)):
            raise _refuse("VLM-2", "params.base_model_dir", f"base_model_dir {d!r} is not an absolute path of a "
                          "directory", "set base_model_dir to the absolute path of the Qwen3.5-0.8B model directory "
                          "downloaded on the host")
        src = Path(d)
        names = sorted(p.name for p in src.iterdir() if p.is_file() and p.suffix in MODEL_SUFFIXES)
        missing = [n for n in REQUIRED_BASE if n not in names]
        if not any(n.endswith(".safetensors") for n in names):
            missing.append("a .safetensors file")
        if missing:
            raise _refuse("VLM-2", "params.base_model_dir", f"the base model directory lacks {', '.join(missing)}",
                          "download the complete model at the pinned commit")
        try:
            arch = json.loads((src / "config.json").read_text(encoding="utf-8")).get("architectures")
        except (OSError, ValueError, AttributeError):
            arch = None
        if not (isinstance(arch, list) and ARCHITECTURE in arch):
            raise _refuse("VLM-2", "params.base_model_dir", f"config.json does not name {ARCHITECTURE} in "
                          "`architectures`", "point base_model_dir at the Qwen3.5-0.8B model directory")
        return src, names

    def plan(self, ctx: StageContext) -> Proposal:
        n = ctx.project.read_artifact(ctx.inputs["profile"]).get("n")
        src, names = self._source(ctx)
        return Proposal(self.name, f"task: CLEVR answers ({n} training rows); base model: {ARCHITECTURE} from {src}, "
                        f"its {len(names)} files copied for the keeper",
                        {"task": "clevr", "sampler": dict(SAMPLER), "eval_seeds": list(EVAL_SEEDS)},
                        {"gpu_hours": 0.0})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        src, names = self._source(ctx)
        out = _backend(ctx, self.name).new_output_dir(BASE_DIR)
        files = [{"path": str(out / n), "sha256": _copy(src / n, out / n)} for n in names]
        stage = ctx.stage or self.name
        deployment = {**_meta(SCHEMA_DEPLOYMENT), "sampler": dict(SAMPLER), "eval_seeds": list(EVAL_SEEDS)}
        base = {**_meta(SCHEMA_MODEL_REF), "generator": GENERATOR, "label": "base", "files": files}
        return {"deployment": ctx.project.write_artifact(stage, "deployment", deployment, SCHEMA_DEPLOYMENT),
                "base": ctx.project.write_artifact(stage, "base", base, SCHEMA_MODEL_REF)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        fix = "rerun clevr_choose_task_and_base"
        base = ctx.project.read_artifact(outputs["base"])
        dep = ctx.project.read_artifact(outputs["deployment"])
        if dep.get("sampler") != SAMPLER or dep.get("eval_seeds") != EVAL_SEEDS:
            return [Problem("VLM-2", "deployment", "the deployment is not the pack's sampler and seeds", fix)]
        if base.get("generator") != GENERATOR or base.get("label") != "base":
            return [Problem("VLM-2", "base", f"the base reference is not a {GENERATOR} base model", fix)]
        try:
            _model_dir(base, "base", "VLM-2", fix)
        except Refused as exc:
            return list(exc.problems)
        bad = _changed_files(base) + [Path(f["path"]).name for f in base["files"] if os.path.exists(f["path"])
                                      and stat.S_IMODE(os.lstat(f["path"]).st_mode) != 0o644]
        if bad:
            return [Problem("VLM-2", "base.files", f"{', '.join(sorted(set(bad)))}: not the listed mode 0644 file", fix)]
        return []


# ---------------------------------------------------------------------------------------------------------------------
# Training set and recipe
# ---------------------------------------------------------------------------------------------------------------------

def train_rows(rows: list[dict], images_dir: str) -> list[dict]:
    """The recipe's training rows (`make_data.py`'s row form): `messages` is NeMo RL's generic form with the image
    named by its absolute path in `images_dir`, the label as the assistant text; `input` is the overlap key (image
    hash and question), `row_id` the position."""
    out = []
    for i, r in enumerate(rows):
        ext, raw = decode_image(r["input"]["image"])
        sha = hashlib.sha256(raw).hexdigest()
        q = r["input"]["question"]
        out.append({"row_id": f"clevr-train-{i:05d}", "image_sha256": f"sha256:{sha}", "input": f"sha256:{sha}\n{q}",
                    "messages": [{"role": "user", "content": [{"type": "image", "image": f"{images_dir}/{sha}.{ext}"},
                                                              {"type": "text", "text": q}]},
                                 {"role": "assistant", "content": [{"type": "text", "text": canonical_label(r["label"])}]}]})
    return out


def train_jsonl(rows: list[dict]) -> bytes:
    """One JSON line per row; JSON escapes every line break inside a value, so lines split on "\\n" only."""
    return "".join(json.dumps(r, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
                   for r in rows).encode("ascii")


def _package_hashes() -> dict[str, str]:
    d = job_mod.PACKAGE_DIR / "runtime"
    return {name: _sha256_path(d / name) for name in ("run_vlm_nemo.py", "ptc_vlm_hooks.py", "vlm_env_logged.py",
                                                      "prompt_clevr_cot.txt")}


class BuildTrainSet:
    """The keeper's training part as the recipe's training data in a backend output directory `train-data-<hash>`:
    `images/<sha256>.<png|jpg>` (each decoded from its row, once per distinct image) and `train.jsonl` (rows naming
    the images by absolute path). The record keeps the directory, the file's hash and the image count."""

    name = "clevr_build_train_set"
    version = "0.1"
    inputs = {"train_part": SCHEMA_TRAIN_PART}
    outputs = {"train_set": SCHEMA_TRAIN_SET}
    needs: frozenset[str] = frozenset()
    params: frozenset[str] = frozenset()
    trains = False

    def _rows(self, ctx: StageContext) -> list[dict]:
        return checked_rows(_read(ctx, "train_part", "VLM-1").get("rows"), "inputs.train_part.rows")

    def plan(self, ctx: StageContext) -> Proposal:
        rows = self._rows(ctx)
        return Proposal(self.name, f"write {len(rows)} training rows and their images as the recipe's training data",
                        {}, {"gpu_hours": 0.0, "n": len(rows)})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        rows = self._rows(ctx)
        digest = ctx.inputs["train_part"].sha256.removeprefix("sha256:")[:12]
        out = _backend(ctx, self.name).new_output_dir(f"{DATA_DIR}-{digest}")
        img_dir = out / "images"
        img_dir.mkdir(mode=0o755, exist_ok=True)
        os.chmod(img_dir, 0o755)
        n_images = 0
        for r in rows:
            ext, raw = decode_image(r["input"]["image"])
            path = img_dir / f"{hashlib.sha256(raw).hexdigest()}.{ext}"
            if not _same(path, _sha256_bytes(raw)):
                _write_bytes(path, raw)
                n_images += 1
        data = train_jsonl(train_rows(rows, str(img_dir)))
        _write_bytes(out / "train.jsonl", data)
        rec = {**_meta(SCHEMA_TRAIN_SET), "n": len(rows), "directory": str(out), "images_dir": str(img_dir),
               "n_images": len(list(img_dir.iterdir())), "train_jsonl_sha256": _sha256_bytes(data)}
        return {"train_set": ctx.project.write_artifact(ctx.stage or self.name, "train_set", rec, SCHEMA_TRAIN_SET)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        t = ctx.project.read_artifact(outputs["train_set"])
        fix = "rerun clevr_build_train_set"
        try:
            ok = (isinstance(t.get("directory"), str) and t["n"] == len(self._rows(ctx))
                  and _same(Path(t["directory"]) / "train.jsonl", t["train_jsonl_sha256"])
                  and len(list(Path(t["images_dir"]).iterdir())) == t["n_images"])
        except (OSError, KeyError, TypeError, Refused):
            ok = False
        if not ok:
            return [Problem("VLM-1", "train_set", "the training data on disk does not match the record", fix)]
        return []


class WriteRecipe:
    """The recipe's settings: its fixed profiles (`run_vlm_nemo.py` PROFILES, which `train_job` runs as they are), the
    training seed, the memory rung, and the hashes of the runtime scripts and prompt the jobs mount. Only the template
    parameters `profile_steps` (with the script's values) and `memory_rung` (a, b or c) are accepted."""

    name = "clevr_write_recipe"
    version = "0.1"
    inputs = {"train_set": SCHEMA_TRAIN_SET}
    outputs = {"recipe": SCHEMA_RECIPE}
    needs: frozenset[str] = frozenset()
    params = frozenset({"profile_steps", "memory_rung"})
    trains = False

    def _settings(self, ctx: StageContext) -> tuple[dict, str]:
        steps = ctx.params.get("profile_steps", PROFILE_STEPS)
        if not (isinstance(steps, dict) and steps == PROFILE_STEPS
                and all(type(v) is int for v in steps.values())):
            raise _refuse("VLM-5", "params.profile_steps", f"profile_steps {steps!r} is not the recipe's "
                          f"{PROFILE_STEPS}: the training job runs run_vlm_nemo.py's fixed profiles",
                          f"leave profile_steps out, or set it to {PROFILE_STEPS}")
        rung = ctx.params.get("memory_rung", MEMORY_RUNG)
        if rung not in job_mod.MEMORY_RUNGS:
            raise _refuse("VLM-5", "params.memory_rung", f"memory_rung {rung!r} is not one of "
                          f"{', '.join(job_mod.MEMORY_RUNGS)}", "use a, or b/c if the smoke run is out of memory")
        return dict(steps), rung

    def _recipe(self, ctx: StageContext) -> dict:
        steps, rung = self._settings(ctx)
        n = _read(ctx, "train_set", "VLM-1").get("n")
        need = PROMPTS_PER_STEP * max(steps.values())
        if not (type(n) is int and n >= need):
            raise _refuse("VLM-5", "inputs.train_set", f"the training set has {n} rows, but {steps['full']} full steps "
                          f"of {PROMPTS_PER_STEP} prompts each read {need} rows (NeMo RL makes one pass over the "
                          "rows, so fewer rows end the run early)", f"give at least {need} labelled rows (the sealed "
                          "part comes on top of them)")
        return {**_meta(SCHEMA_RECIPE), "profile_steps": steps, "memory_rung": rung, "seed": TRAINING_SEED,
                "gpu_hours": {p: v["gpu_hours_est"] for p, v in job_mod.PROFILES.items()},
                "train_set_sha256": ctx.inputs["train_set"].sha256, "runtime_sha256": _package_hashes()}

    def plan(self, ctx: StageContext) -> Proposal:
        r = self._recipe(ctx)
        return Proposal(self.name, f"NeMo RL VLM GRPO recipe: smoke {r['profile_steps']['smoke']} steps, full "
                        f"{r['profile_steps']['full']} steps, seed {r['seed']}, memory rung {r['memory_rung']}",
                        {"profile_steps": r["profile_steps"], "memory_rung": r["memory_rung"]},
                        {"gpu_hours": sum(r["gpu_hours"].values())})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        return {"recipe": ctx.project.write_artifact(ctx.stage or self.name, "recipe", self._recipe(ctx),
                                                     SCHEMA_RECIPE)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        r = ctx.project.read_artifact(outputs["recipe"])
        if r.get("train_set_sha256") != ctx.inputs["train_set"].sha256 or r.get("runtime_sha256") != _package_hashes():
            return [Problem("VLM-5", "recipe", "the recipe is not for this training set and this pack's runtime",
                            "rerun clevr_write_recipe")]
        return []


# ---------------------------------------------------------------------------------------------------------------------
# The training jobs
# ---------------------------------------------------------------------------------------------------------------------

class _JobStage:
    version = "0.1"
    needs = frozenset({"gpu"})
    params = frozenset({"image"})
    PROFILE = ""
    RULE = ""

    def _params(self, ctx: StageContext) -> dict:
        image = ctx.params.get("image")
        if not (isinstance(image, str) and image):
            raise _refuse("VLM-4", "params.image", f"image {image!r} is not an image reference",
                          "set image to the ID of the derived NeMo RL image the owner's Docker daemon holds")
        try:
            allowed = job_mod.accepted_images()
        except ValueError as exc:
            raise _refuse("VLM-4", "params.image", str(exc), f"fix {job_mod.EXTRA_IMAGES_ENV}") from None
        if image not in allowed:
            raise _refuse("VLM-4", "params.image", f"image {image!r} is not one of the pinned NeMo RL images or an "
                          f"image listed in {job_mod.EXTRA_IMAGES_ENV}", "set image to one of " + ", ".join(allowed))
        return {"image": image}

    def _base(self, ctx: StageContext, *, hashes: bool) -> Path:
        fix = "rerun clevr_choose_task_and_base"
        base = _read(ctx, "base", "VLM-2")
        d = _model_dir(base, "inputs.base", "VLM-2", fix)
        bad = _changed_files(base) if hashes else [Path(f["path"]).name for f in base["files"]
                                                   if not os.path.isfile(f["path"])]
        if bad:
            raise _refuse("VLM-2", "inputs.base.files", f"{', '.join(bad)}: not the base model file its reference "
                          "lists (changed or removed since clevr_choose_task_and_base)", fix)
        return d

    def _recipe(self, ctx: StageContext) -> tuple[dict, dict]:
        recipe, ts = _read(ctx, "recipe", "VLM-5"), _read(ctx, "train_set", "VLM-1")
        if recipe.get("train_set_sha256") != ctx.inputs["train_set"].sha256:
            raise _refuse("VLM-5", "inputs.recipe", "the recipe was written for another training set",
                          "rerun clevr_write_recipe")
        if recipe.get("runtime_sha256") != _package_hashes():
            raise _refuse("VLM-5", "inputs.recipe", "the recipe names other runtime scripts than this pack's",
                          "rerun clevr_write_recipe")
        return recipe, ts

    def _estimate(self) -> dict:
        return {"gpu_hours": job_mod.PROFILES[self.PROFILE]["gpu_hours_est"]}

    @staticmethod
    def _verify_images(data: Path, ts: dict) -> None:
        """The training images are the files `clevr_build_train_set` wrote: the record's directory is `images` of its
        own data directory and every file there is a regular file named by its own SHA-256."""
        fix = "rerun clevr_build_train_set"
        if Path(str(ts.get("images_dir"))) != data / "images":
            raise _refuse("VLM-1", "inputs.train_set", "the record's image directory is not the data directory's "
                          "`images`", fix)
        try:
            files = sorted((data / "images").iterdir())
            ok = len(files) == ts.get("n_images")
            for f in files:
                got = _sha256_path(f, follow=False)
                ok = ok and got is not None and got.removeprefix("sha256:") == f.stem
        except OSError:
            ok = False
        if not ok:
            raise _refuse("VLM-1", "inputs.train_set", "an image is not the file its name says (changed, removed or "
                          "added since clevr_build_train_set)", fix)

    def _run_job(self, ctx: StageContext) -> tuple[Any, str]:
        """Check the training data on disk, build the job, ask for its output directory by its key, submit (a job of
        the same key reattaches) and wait. The backend and the job id of a succeeded job."""
        p = self._params(ctx)
        base_dir = self._base(ctx, hashes=True)
        recipe, ts = self._recipe(ctx)
        data = Path(ts["directory"])
        if not _same(data / "train.jsonl", ts["train_jsonl_sha256"]):
            raise _refuse("VLM-1", "inputs.train_set", "train.jsonl is not the file its record lists",
                          "rerun clevr_build_train_set")
        self._verify_images(data, ts)
        backend = _backend(ctx, self.name)
        j = job_mod.train_job(profile=self.PROFILE, image=p["image"], inputs_dir=data, images_dir=Path(ts["images_dir"]),
                              base_dir=base_dir, seed=recipe["seed"], memory_rung=recipe["memory_rung"])
        j = dataclasses.replace(j, output_dir=str(backend.new_output_dir(j.label, key=j.key())))
        job_id = backend.submit(j)
        status = backend.wait(job_id, poll_s=JOB_POLL_S)
        if status.state != "succeeded":
            raise _refuse(self.RULE, f"{self.PROFILE}.job", f"the {self.PROFILE} training job {job_id} ended "
                          f"{status.state}" + (f" ({status.reason})" if status.reason else ""),
                          "read `gatedtrain project job logs`, fix the cause and run the stage again")
        return backend, job_id


def _jsonl(path: Path, listed: dict, name: str) -> list:
    f = listed.get(name)
    if f is None:
        raise _refuse("MM-38", f"smoke.{name}", f"the smoke job wrote no {name}", "rerun smoke with the pack's script")
    if os.lstat(path / name).st_size > MAX_LOG_BYTES:
        raise _refuse("MM-38", f"smoke.{name}", f"{name} is larger than {MAX_LOG_BYTES} bytes", "rerun smoke")
    data = (path / name).read_bytes()
    if _sha256_bytes(data) != f["sha256"]:
        raise _refuse("MM-38", f"smoke.{name}", f"{name} changed after the job ended", "rerun smoke")
    try:
        return [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]
    except (UnicodeDecodeError, ValueError):
        raise _refuse("MM-38", f"smoke.{name}", f"{name} is not JSON lines", "rerun smoke with the pack's script") \
            from None


def smoke_measures(calls: list) -> dict:
    """The MM-38 measurements from the logging environment's `env_calls.jsonl` (vlm_env_logged.py: one line per
    sample with `reward` and `response_sha256`). Raises ValueError on a malformed line."""
    reward_sum = 0.0
    digests: set[str] = set()
    for i, r in enumerate(calls):
        if not (isinstance(r, dict) and isinstance(r.get("response_sha256"), str) and _num(r.get("reward"))):
            raise ValueError(f"env_calls.jsonl line {i + 1} has no response hash and reward")
        reward_sum += r["reward"]
        digests.add(r["response_sha256"])
    n = len(calls)
    return {"n_samples": n, "n_distinct_completions": len(digests), "reward_mean": reward_sum / n if n else 0.0,
            "all_rewards_zero": n > 0 and reward_sum == 0}


def smoke_reasons(m: dict) -> list[str]:
    out = []
    if m["n_samples"] < MIN_SAMPLES:
        out.append(f"only {m['n_samples']} samples (at least {MIN_SAMPLES} are needed to judge)")
    else:
        if m["all_rewards_zero"]:
            out.append(f"all rewards are 0 across {m['n_samples']} samples")
        d = m["n_distinct_completions"]
        if d < 2:
            out.append(f"{d} distinct completion{'' if d == 1 else 's'} across {m['n_samples']} samples")
    return out


class Smoke(_JobStage):
    """The recipe's smoke profile (5 steps) as a GPU job, then MM-38: refused when every reward is 0 or fewer than 2
    distinct completions appear across at least 8 samples. NeMo RL writes no GPU memory log, so the memory bound is
    the job's own success on the host's GPU at the recipe's memory rung (an out-of-memory run ends the job)."""

    name = "clevr_smoke"
    inputs = {"base": SCHEMA_MODEL_REF, "train_set": SCHEMA_TRAIN_SET, "recipe": SCHEMA_RECIPE}
    outputs = {"smoke": SCHEMA_SMOKE}
    trains = False
    PROFILE = "smoke"
    RULE = "MM-38"

    def plan(self, ctx: StageContext) -> Proposal:
        p = self._params(ctx)
        self._base(ctx, hashes=False)
        recipe, ts = self._recipe(ctx)
        return Proposal(self.name, f"smoke-train {recipe['profile_steps']['smoke']} GRPO steps on {ts['n']} rows in "
                        f"{p['image']}; refuse all-zero rewards or one repeated completion", {}, self._estimate())

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        backend, job_id = self._run_job(ctx)
        arts = backend.artifacts(job_id)
        listed = {f["path"]: f for f in arts["files"]}
        calls = _jsonl(Path(arts["output_dir"]), listed, "env_calls.jsonl")
        try:
            m = smoke_measures(calls)
        except ValueError as exc:
            raise _refuse("MM-38", "smoke", f"the smoke output cannot be judged: {exc}",
                          "rerun smoke with the pack's run_vlm_nemo.py") from None
        reasons = smoke_reasons(m)
        if reasons:
            raise _refuse("MM-38", "smoke", f"the smoke run {job_id} is degenerate: {'; '.join(reasons)}",
                          "change the recipe, the data or the base model so the smoke run learns from varied "
                          "completions")
        del m["all_rewards_zero"]
        out = {**_meta(SCHEMA_SMOKE), "verdict": "PASS", "reasons": [], "job_id": job_id, **m,
               "memory": "no GPU memory log: the job finished at the recipe's memory rung",
               "base_sha256": ctx.inputs["base"].sha256, "train_set_sha256": ctx.inputs["train_set"].sha256,
               "recipe_sha256": ctx.inputs["recipe"].sha256}
        return {"smoke": ctx.project.write_artifact(ctx.stage or self.name, "smoke", out, SCHEMA_SMOKE)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        s = ctx.project.read_artifact(outputs["smoke"])
        try:
            again = smoke_reasons({**s, "all_rewards_zero": s.get("reward_mean") == 0})
        except (KeyError, TypeError):
            again = ["incomplete record"]
        if s.get("verdict") != "PASS" or s.get("reasons") != [] or again:
            return [Problem("MM-38", "smoke.verdict", "the smoke record is not a PASS its measurements support",
                            "rerun smoke")]
        return []


def _cut_step(name: str) -> int | None:
    m = _CHECKPOINT.fullmatch(name)
    return int(m.group(1)) if m else None


class Train(_JobStage):
    """The recipe's full profile (60 steps) as a GPU job after a smoke PASS of the same inputs; the candidate is the
    last cut's deployable files (`cuts/checkpoint-<N>/cut.json` lists them with their hashes), handed off by the
    backend (each `o+r` in `o+x` directories: the stage makes them so first, they are the owner's files)."""

    name = "clevr_train"
    inputs = {"base": SCHEMA_MODEL_REF, "train_set": SCHEMA_TRAIN_SET, "recipe": SCHEMA_RECIPE,
              "smoke": SCHEMA_SMOKE}
    outputs = {"model": SCHEMA_MODEL_REF}
    trains = True
    PROFILE = "full"
    RULE = "VLM-3"

    def _smoke_passed(self, ctx: StageContext) -> None:
        s = _read(ctx, "smoke", "MM-38")
        if s.get("verdict") != "PASS":
            raise _refuse("MM-38", "inputs.smoke", "the smoke run did not pass", "rerun clevr_smoke")
        for key in ("base", "train_set", "recipe"):
            if s.get(f"{key}_sha256") != ctx.inputs[key].sha256:
                raise _refuse("MM-38", "inputs.smoke", f"the smoke run passed with another {key} than this "
                              "training run's", "rerun clevr_smoke on these inputs")

    def plan(self, ctx: StageContext) -> Proposal:
        self._smoke_passed(ctx)
        p = self._params(ctx)
        self._base(ctx, hashes=False)
        recipe, ts = self._recipe(ctx)
        return Proposal(self.name, f"GRPO-train {recipe['profile_steps']['full']} steps on {ts['n']} rows in "
                        f"{p['image']}; the candidate is the last cut", {}, self._estimate())

    @staticmethod
    def _open_cut(top: Path, last: int) -> list[str]:
        """Make the cut's directories and files readable by other users (o+x, o+r) and return the deployable file
        names `cut.json` lists, refusing (VLM-3) a cut whose files do not match its hashes.

        The container wrote the cut, so nothing in it is trusted: every directory and file is opened without
        following links (O_NOFOLLOW) and checked on the open descriptor, a name must be a plain file name with an
        allowed suffix, and a file's hash is checked before its mode is changed (through the descriptor)."""
        where = f"train.checkpoint-{last}"
        fix = "read `gatedtrain project job logs` and run the stage again"
        modes = []
        fds = []

        def refuse(message: str) -> Refused:
            return _refuse("VLM-3", where, message, fix)

        def open_dir(path: Path) -> int:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            except OSError:
                raise refuse(f"{path.name} is not a directory (or is a link)") from None
            fds.append(fd)
            return fd

        try:
            dir_fds = [open_dir(top / "cuts"), open_dir(top / "cuts" / f"checkpoint-{last}")]
            cut_dir = top / "cuts" / f"checkpoint-{last}"
            try:
                fd = os.open(cut_dir / "cut.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as f:
                    if not stat.S_ISREG(os.fstat(fd).st_mode):
                        raise ValueError
                    cut = json.loads(f.read(_MAX_JSON).decode("utf-8"))
                names, hashes = cut["deployable"], cut["files"]
                ok = (isinstance(names, list) and names and isinstance(hashes, dict)
                      and all(isinstance(n, str) and isinstance(hashes.get(n), str) for n in names)
                      and cut.get("global_step") == last)
            except (OSError, ValueError, KeyError, TypeError):
                ok = False
            if not ok:
                raise refuse(f"cut checkpoint-{last} has no valid cut.json")
            for n in names:
                if not _PLAIN_NAME.fullmatch(n) or n in (".", ".."):
                    raise refuse("a listed name is not a plain file name")
                if Path(n).suffix not in MODEL_SUFFIXES:
                    raise refuse(f"{n} has a suffix the keeper does not accept ({', '.join(MODEL_SUFFIXES)})")
            file_fds = []
            for n in names:                     # open and hash every file first; change modes only when all pass
                try:
                    fd = os.open(cut_dir / n, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                except OSError:
                    raise refuse(f"{n} of the last cut is not a regular file (or is a link)") from None
                fds.append(fd)
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise refuse(f"{n} of the last cut is not a regular file")
                h = hashlib.sha256()
                with os.fdopen(os.dup(fd), "rb") as f:
                    for chunk in iter(lambda: f.read(_CHUNK), b""):
                        h.update(chunk)
                if "sha256:" + h.hexdigest() != hashes[n]:
                    raise refuse(f"{n} does not match the hash its cut.json lists")
                file_fds.append(fd)
            for fd in dir_fds:
                os.fchmod(fd, stat.S_IMODE(os.fstat(fd).st_mode) | 0o755)
            for fd in file_fds:
                os.fchmod(fd, stat.S_IMODE(os.fstat(fd).st_mode) | 0o444)
            return list(names)
        finally:
            for fd in fds:
                os.close(fd)

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        self._smoke_passed(ctx)
        backend, job_id = self._run_job(ctx)
        arts = backend.artifacts(job_id)
        paths = {f["path"] for f in arts["files"]}
        cuts = {s for p in paths if p.startswith("cuts/") and p.endswith("/cut.json")
                for s in [_cut_step(p.split("/")[1])] if s is not None}
        if not cuts:
            raise _refuse("VLM-3", "train.checkpoints", f"the training job {job_id} wrote no cuts/checkpoint-<N> "
                          "with a cut.json", "read `gatedtrain project job logs` and run the stage again")
        last = max(cuts)
        recipe = _read(ctx, "recipe", "VLM-5")
        if last != recipe["profile_steps"]["full"]:
            raise _refuse("VLM-3", "train.checkpoints", f"the last cut of job {job_id} is checkpoint-{last}, not the "
                          f"final step {recipe['profile_steps']['full']}", "run the stage again")
        names = self._open_cut(Path(arts["output_dir"]), last)
        files = backend.hand_off(job_id, [f"cuts/checkpoint-{last}/{n}" for n in names])
        model = {**_meta(SCHEMA_MODEL_REF), "generator": GENERATOR, "label": "candidate", "job_id": job_id,
                 "checkpoint": last, "files": files}
        return {"model": ctx.project.write_artifact(ctx.stage or self.name, "model", model, SCHEMA_MODEL_REF)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        fix = "rerun clevr_train"
        m = ctx.project.read_artifact(outputs["model"])
        if m.get("generator") != GENERATOR or m.get("label") != "candidate":
            return [Problem("VLM-3", "model", f"the model is not a {GENERATOR} candidate", fix)]
        try:
            d = _model_dir(m, "model", "VLM-3", fix)
        except Refused as exc:
            return list(exc.problems)
        if d.name != f"checkpoint-{m.get('checkpoint')}":
            return [Problem("VLM-3", "model.files", "the files are not those of the recorded checkpoint", fix)]
        bad = _changed_files(m) + [Path(f["path"]).name for f in m["files"] if os.path.exists(f["path"])
                                   and not os.lstat(f["path"]).st_mode & stat.S_IROTH]
        if bad:
            return [Problem("VLM-3", "model.files", f"{', '.join(sorted(set(bad)))}: not the listed file readable by "
                            "other users", fix)]
        return []
