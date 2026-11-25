"""The CLEVR pack's batch generator (gatedtrain AN-25): standard library plus vLLM, transformers and Pillow, run in
the keeper's inference container as

    python -I generate_vllm.py --items ITEMS --sampler SAMPLER --model MODEL_DIR --out OUT_DIR

It is copied into the container on its own, so it imports nothing from the pack and embeds the prompt (a test pins it
to `runtime/prompt_clevr_cot.txt`). It reads the items (canonical `{"item_id", "input"}` lines, the input being
`{"question", "image"}` with the image a `data:image/png|jpeg;base64,` URI), the sampler and seeds
(`{"sampler", "seeds"}`) and the model directory, and writes `OUT_DIR/predictions.jsonl`: one canonical JSON line
`{"item_id", "prediction", "seed"}` per seed and item, in (seed, item) order, the prediction being the completion text.

The model sees exactly the training message: one user message `[{type: image}, {type: text, text:
PROMPT.format(question)}]` through the checkpoint's processor chat template with `add_generation_prompt=True` and no
template arguments (Qwen3.5 then answers without thinking). The sampler (temperature, top_p, max_new_tokens) and the seeds are the
registered deployment's; a sampler outside the deployment keys is refused with exit code 2 before the engine is loaded, and so is
an item whose image is not a PNG or JPEG data URI (the message quotes no item).

The container root is read-only with only /tmp writable and no network, so every cache is pointed below /tmp before
vLLM is imported. Once the predictions file is written, flushed, synced and closed, the process ends at once with
`os._exit(0)`: vLLM's engine teardown at interpreter exit is known to hang.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import os
import sys

PROMPT = ("Think step-by-step to solve the following problem, and answer in the following format: <think> "
          "step-by-step thought process </think> <answer> final answer </answer>\nNote that your final answer must "
          "only contain a single numerical output (e.g. 2, 12, 45) for numerical solutions, and only the strings "
          "\"yes\" or \"no\" for yes/no type questions.\n\nLet's think step-by-step: \n{}\n")
SAMPLER_KEYS = frozenset({"name", "temperature", "top_p", "max_new_tokens", "decoding"})
PREDICTIONS = "predictions.jsonl"
MAX_MODEL_LEN = 1024
MAX_PIXELS = 400_000      # the prompt budget (about 400 image tokens), as the pack's stages and converter
MAGIC = {"png": b"\x89PNG\r\n\x1a\n", "jpeg": b"\xff\xd8\xff"}
DATA_URI_PREFIXES = ("data:image/png;base64,", "data:image/jpeg;base64,")

_ENV_DEFAULTS = (
    ("VLLM_CACHE_ROOT", "/tmp/cache/vllm"),
    ("TRITON_CACHE_DIR", "/tmp/cache/triton"),
    ("TORCHINDUCTOR_CACHE_DIR", "/tmp/cache/inductor"),
    ("XDG_CACHE_HOME", "/tmp/cache"),
    ("HF_HOME", "/tmp/cache/hf"),
    ("VLLM_USE_FLASHINFER_SAMPLER", "0"),
    ("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
)


class _Refusal(Exception):
    pass


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_sampler(doc) -> tuple[dict, list]:
    """The sampler and seeds of `doc`, or _Refusal with a message that quotes no item."""
    if not isinstance(doc, dict):
        raise _Refusal("the sampler file is not an object with sampler and seeds")
    sampler, seeds = doc.get("sampler"), doc.get("seeds")
    if not isinstance(sampler, dict):
        raise _Refusal("the sampler is not an object")
    extra = sorted(set(sampler) - SAMPLER_KEYS)
    if extra:
        raise _Refusal(f"the sampler has keys outside {sorted(SAMPLER_KEYS)}: {extra}")
    for key in ("temperature", "top_p"):
        if not _number(sampler.get(key)):
            raise _Refusal(f"the sampler's {key} must be a number")
    if "decoding" in sampler and sampler["decoding"] != "greedy":
        raise _Refusal("the sampler's decoding must be 'greedy' when present")
    if sampler.get("decoding") == "greedy":
        if sampler["temperature"] != 0 or sampler["top_p"] != 1:
            raise _Refusal("a greedy sampler has temperature 0 and top_p 1")
    elif sampler["temperature"] == 0:
        raise _Refusal("temperature 0 is greedy decoding and needs the sampler's decoding to be 'greedy'")
    max_new = sampler.get("max_new_tokens")
    if isinstance(max_new, bool) or not isinstance(max_new, int) or max_new < 1:
        raise _Refusal("the sampler's max_new_tokens must be a positive integer")
    if not isinstance(seeds, list) or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise _Refusal("the seeds must be a list of integers")
    return sampler, seeds


def _decode(item_input) -> tuple[str, bytes]:
    """(question, image bytes) of an item input, or _Refusal naming the position only."""
    if not (isinstance(item_input, dict) and isinstance(item_input.get("question"), str)
            and isinstance(item_input.get("image"), str)):
        raise _Refusal("an item input is not an object with string 'question' and 'image'")
    uri = item_input["image"]
    prefix = next((p for p in DATA_URI_PREFIXES if uri.startswith(p)), None)
    if prefix is None:
        raise _Refusal("an item image is not a PNG or JPEG data URI")
    try:
        raw = base64.b64decode(uri[len(prefix):], validate=True)
    except (binascii.Error, ValueError):
        raise _Refusal("an item image is not valid base64") from None
    if not raw.startswith(MAGIC["png" if "png" in prefix else "jpeg"]):
        raise _Refusal("an item image does not carry the PNG or JPEG signature its URI names")
    return item_input["question"], raw


def _check_image(Image, raw: bytes) -> None:
    """Refuse an image Pillow cannot open as PNG or JPEG or whose pixel count is over the budget (header only)."""
    try:
        size = Image.open(io.BytesIO(raw), formats=["PNG", "JPEG"]).size
    except Exception:
        raise _Refusal("an item image cannot be opened as PNG or JPEG") from None
    if size[0] < 1 or size[1] < 1 or size[0] * size[1] > MAX_PIXELS:
        raise _Refusal(f"an item image has more than {MAX_PIXELS} pixels")


def _line(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode() + b"\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    for name in ("--items", "--sampler", "--model", "--out"):
        ap.add_argument(name, required=True)
    args = ap.parse_args(argv)
    with open(args.sampler, encoding="utf-8") as f:
        sampler_doc = json.load(f)
    from PIL import Image
    try:
        sampler, seeds = _check_sampler(sampler_doc)
        with open(args.items, encoding="utf-8") as f:
            items = [json.loads(line) for line in f.read().split("\n") if line]
        if not all(isinstance(i, dict) and "item_id" in i and "input" in i for i in items):
            raise _Refusal("an item line is not an object with item_id and input")
        decoded = [_decode(item["input"]) for item in items]
        for _, raw in decoded:
            _check_image(Image, raw)
    except _Refusal as exc:
        print(f"generate_vllm: refused: {exc}", file=sys.stderr)
        return 2
    except ValueError:
        print("generate_vllm: refused: the items file is not JSON lines", file=sys.stderr)
        return 2

    for key, value in _ENV_DEFAULTS:
        os.environ.setdefault(key, value)
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    requests = []
    for question, raw in decoded:
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": PROMPT.format(question)}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        requests.append({"prompt": prompt, "multi_modal_data": {"image": Image.open(io.BytesIO(raw)).convert("RGB")}})
    greedy = sampler.get("decoding") == "greedy"
    # Greedy decoding ignores the seed. The keeper runs it under a control seed too and compares the predictions
    # (gatedtrain AN-26), so a later pass must not differ from the first. Prefix caching would let it reuse the first
    # pass's key-value blocks, so it is off; and one request at a time (max_num_seqs 1) keeps the numerics from
    # depending on which requests share a batch: with the default batching, 54 to 56 % of the items of the 0.8B base
    # model differed between two passes of one process (30 % for the trained model), a long chain of thought turning a
    # last-bit difference into another text. vLLM's batch-invariant mode does not support Qwen3.5's attention (GDN_ATTN).
    extra = {"enable_prefix_caching": False, "max_num_seqs": 1} if greedy else {}
    llm = LLM(model=args.model, dtype="bfloat16", seed=0, max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 1}, mm_processor_cache_gb=0, **extra)
    lines = []
    for seed in seeds:
        params = SamplingParams(temperature=float(sampler["temperature"]), top_p=float(sampler["top_p"]),
                                max_tokens=sampler["max_new_tokens"], **({} if greedy else {"seed": seed}))
        outputs = llm.generate(requests, params)
        if len(outputs) != len(items):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(items)} requests")
        lines.extend(_line({"item_id": item["item_id"], "prediction": out.outputs[0].text, "seed": seed})
                     for item, out in zip(items, outputs))

    fd = os.open(os.path.join(args.out, PREDICTIONS), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(b"".join(lines))
        f.flush()
        os.fsync(f.fileno())
    return 0


if __name__ == "__main__":
    code = main()
    if code == 0:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    sys.exit(code)
