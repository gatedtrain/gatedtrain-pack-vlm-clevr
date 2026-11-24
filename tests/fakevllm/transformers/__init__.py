"""A stand-in for `transformers` for the tests of `generate_vllm.py` only: `AutoProcessor` records its chat-template
calls in the file named by `FAKEVLLM_RECORD` and renders image parts as `<img>`."""

import json
import os


def _record(doc):
    path = os.environ.get("FAKEVLLM_RECORD")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(doc, sort_keys=True) + "\n")


class AutoProcessor:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        _record({"event": "processor", "path": path, "kwargs": kwargs})
        return cls()

    def apply_chat_template(self, messages, **kwargs):
        _record({"event": "chat_template", "messages": messages, "kwargs": kwargs})
        parts = messages[0]["content"]
        return "<|user|>" + "".join("<img>" if p["type"] == "image" else p["text"] for p in parts) + "<|assistant|>"
