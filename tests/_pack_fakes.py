"""Fake container work for the pack's tests: what the training job and the keeper's inference container write.

- `play_training` plays `run_vlm_nemo.py` in a job's container: it writes `env_calls.jsonl` (the logging environment's
  lines) and `cuts/checkpoint-<N>/` directories whose files come out 0600 (directories 0700) with a `cut.json` listing
  the deployable files and their hashes, as the recipe's cut writer does. A model file holds `<profile> step <N>
  <name>`, so the full run's safetensors file starts with `full step`.
- `play_inference` plays `generate_vllm.py`: one canonical prediction per seed and item, `<answer>LABEL</answer>` for a
  model whose safetensors file starts with `full step` (the candidate), `<answer>0</answer>` otherwise. Labels come
  from an intake data file keyed by the item's question and image.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

PNG = b"\x89PNG\r\n\x1a\n"
CANDIDATE_MARK = b"full step"
BASE_FILES = ("chat_template.jinja", "config.json", "LICENSE", "merges.txt",
              "model.safetensors-00001-of-00001.safetensors", "model.safetensors.index.json",
              "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
              "video_preprocessor_config.json", "vocab.json")
CUT_FILES = ("chat_template.jinja", "config.json", "model-00001-of-00001.safetensors", "model.safetensors.index.json",
             "preprocessor_config.json", "sampler.json", "tokenizer.json", "tokenizer_config.json",
             "video_preprocessor_config.json")


def write_base(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name in BASE_FILES:
        (d / name).write_text(f"base {name}\n")
    (d / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}))


def png(i: int) -> bytes:
    """Not a decodable picture: a PNG signature and a valid IHDR chunk (32 x 24, the size the pack's header reader
    needs), then a marker that makes every image distinct."""
    return PNG + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 32, 24) + b"\x08\x02\x00\x00\x00" + b"\x00" * 4 \
        + f"image {i:05d}-end".encode()


def play_training(run_spec, host_path, *, rewards=(1.0, 0.0, 0.0, 0.0), texts=None, chmod=False) -> int:
    argv = list(run_spec.argv)
    profile = argv[argv.index("--profile") + 1]
    out = host_path("/out")
    assert host_path("/inputs/train.jsonl").is_file()
    assert host_path("/inputs/base-model/config.json").is_file()
    steps, save = (5, 5) if profile == "smoke" else (60, 20)
    with open(out / "env_calls.jsonl", "a", encoding="utf-8") as f:
        for step in range(steps):
            for pos in range(16):
                resp = texts[pos % len(texts)] if texts else f"step {step} pos {pos}"
                f.write(json.dumps({"call": step + 1, "position": pos, "row_id": f"clevr-train-{pos:05d}",
                                    "reward": rewards[pos % len(rewards)],
                                    "response_sha256": "sha256:" + hashlib.sha256(resp.encode()).hexdigest(),
                                    "utc": "2026-11-24T00:00:00.000000Z"}) + "\n")
            if (step + 1) % save == 0:
                cut = out / "cuts" / f"checkpoint-{step + 1}"
                os.makedirs(cut, mode=0o700, exist_ok=True)
                files = {}
                for name in CUT_FILES:
                    data = f"{profile} step {step + 1} {name}\n".encode()
                    fd = os.open(cut / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    os.write(fd, data)
                    os.close(fd)
                    files[name] = "sha256:" + hashlib.sha256(data).hexdigest()
                doc = {"deployable": list(CUT_FILES), "files": files, "global_step": step + 1, "profile": profile,
                       "schema": "ptc_mm.vlm_cut"}
                (cut / "cut.json").write_text(json.dumps(doc, sort_keys=True))
                os.chmod(cut / "cut.json", 0o600)
                for d in (cut, out / "cuts"):
                    os.chmod(d, 0o700)
    print("trained", profile)
    return 0


def labels_by_key(data: Path) -> dict:
    return {(r["input"]["question"], r["input"]["image"]): r["label"] for r in
            (json.loads(line) for line in data.read_text(encoding="utf-8").split("\n") if line)}


def play_inference(run_spec, host_path, labels: dict) -> int:
    argv = list(run_spec.argv)
    arg = lambda flag: host_path(argv[argv.index(flag) + 1])          # noqa: E731
    doc = json.loads(arg("--sampler").read_text(encoding="utf-8"))
    items = [json.loads(line) for line in arg("--items").read_text(encoding="utf-8").split("\n") if line]
    model = arg("--model")
    candidate = next(model.glob("*.safetensors")).read_bytes().startswith(CANDIDATE_MARK)
    lines = []
    for seed in doc["seeds"]:
        for item in items:
            answer = labels[(item["input"]["question"], item["input"]["image"])] if candidate else "0"
            lines.append(json.dumps({"item_id": item["item_id"], "prediction": f"<answer>{answer}</answer>",
                                     "seed": seed}, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                    allow_nan=False))
    (arg("--out") / "predictions.jsonl").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return 0
