"""The pack's deciding scorer `vqa_answer@0.1` (standard library only; no gatedtrain import).

The answer parser of the CLEVR recipe's `score_vqa.py`, as a function of one response and one label:

1. Drop everything up to the last `</think>` if one occurs.
2. If an `<answer>...</answer>` block occurs, the candidate is the content of the last one; otherwise the rest.
3. The canonical answer is the last token of the candidate that is an integer, a number word zero..twenty or
   yes/no (case-insensitive, ASCII digits only, word boundaries); number words become digits and leading zeros are dropped.
4. The score is 1.0 when the canonical answer equals the label's canonical form, else 0.0.

It never raises: a response that is not a string, or a label without a canonical form, scores 0.0.
"""

from __future__ import annotations

import re

NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                "sixteen seventeen eighteen nineteen twenty").split()
TOKEN_RE = re.compile(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r"|yes|no)\b", re.I | re.ASCII)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def _token(tok: str) -> str:
    t = tok.lower()
    if t in NUMBER_WORDS:
        return str(NUMBER_WORDS.index(t))
    if t.isdigit():
        return str(int(t))
    return t


def canonical(text: str) -> str | None:
    """The canonical answer of a response, or None when it holds none."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = ANSWER_RE.findall(text)
    candidate = blocks[-1] if blocks else text
    tokens = TOKEN_RE.findall(candidate)
    return _token(tokens[-1]) if tokens else None


def canonical_label(label: object) -> str | None:
    """The canonical form of a label: exactly one token and nothing else (`" 7 "` -> "7", `"Yes"` -> "yes")."""
    if not isinstance(label, str):
        return None
    tokens = TOKEN_RE.findall(label)
    if len(tokens) != 1 or TOKEN_RE.sub("", label).strip():
        return None
    return _token(tokens[0])


def label_ok(label: object) -> bool:
    return canonical_label(label) is not None


def score(prediction: object, label: object) -> float:
    if not isinstance(prediction, str):
        return 0.0
    want = canonical_label(label)
    return 1.0 if want is not None and canonical(prediction) == want else 0.0
