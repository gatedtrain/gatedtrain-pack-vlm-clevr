"""A stand-in for the `vllm` package, for the tests of `generate_vllm.py` only.

It records what the script passes (the environment at import, the engine arguments, the chat-template calls and the
sampling parameters) as JSON lines in the file named by `FAKEVLLM_RECORD`, and answers each request (a dict with `prompt` and `multi_modal_data`) with
`"<answer><len(prompt) % 7></answer>"`; it records each image's size.

Like vLLM's engine teardown, which is known to hang, `LLM` registers an exit handler: it records an `atexit` event and
then sleeps `FAKEVLLM_EXIT_HANG_S` seconds (default 0). `FAKEVLLM_SHORT=1` makes `generate` return one output too few.
"""

import atexit
import json
import os
import time

_WATCHED_ENV = (
    "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "XDG_CACHE_HOME", "HF_HOME",
    "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_WORKER_MULTIPROC_METHOD",
)


def _record(doc):
    path = os.environ.get("FAKEVLLM_RECORD")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(doc, sort_keys=True) + "\n")


_record({"event": "import", "env": {k: os.environ.get(k) for k in _WATCHED_ENV}})


class SamplingParams:
    def __init__(self, temperature=1.0, top_p=1.0, max_tokens=16, seed=None):
        self.temperature, self.top_p, self.max_tokens, self.seed = temperature, top_p, max_tokens, seed


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        _record({"event": "chat_template", "messages": messages, "kwargs": kwargs})
        return "<|user|>" + messages[0]["content"] + "<|assistant|>"


class _Completion:
    def __init__(self, text):
        self.text = text


class _RequestOutput:
    def __init__(self, prompt, text):
        self.prompt = prompt
        self.outputs = [_Completion(text)]


class LLM:
    def __init__(self, model, **kwargs):
        _record({"event": "llm", "model": model, "kwargs": kwargs})
        atexit.register(_teardown)

    def get_tokenizer(self):
        return _Tokenizer()

    def generate(self, prompts, sampling_params=None, **kwargs):
        sp = sampling_params
        prompts = list(prompts)
        _record({"event": "generate", "n": len(prompts), "images": [list(p["multi_modal_data"]["image"].size) for p in prompts], "kwargs": sorted(kwargs),
                 "sampling": {"temperature": sp.temperature, "top_p": sp.top_p, "max_tokens": sp.max_tokens,
                              "seed": sp.seed}})
        out = [_RequestOutput(p["prompt"], f"<answer>{len(p['prompt']) % 7}</answer>") for p in prompts]
        return out[:-1] if os.environ.get("FAKEVLLM_SHORT") == "1" else out


def _teardown():
    _record({"event": "atexit"})
    time.sleep(float(os.environ.get("FAKEVLLM_EXIT_HANG_S", "0")))
