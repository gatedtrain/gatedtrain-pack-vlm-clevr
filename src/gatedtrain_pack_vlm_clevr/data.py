"""Convert CLEVR-style items (`{"question", "answer", "image"}` JSONL plus an image directory) to the intake format
(`{"input": {"question", "image"}, "label"}`) with the image inline as a data URI, so a sealed image never lies in a
file the owner can read. Refusals name line numbers and reasons only, never row content: the data may be sealed later.

Images are re-encoded as JPEG by default (needs Pillow, the `convert` extra): a CLEVR PNG is about 190 KB and the
framework's data file is limited to 64 MiB. Pass `jpeg_quality=None` to keep the source bytes (PNG or JPEG).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

from gatedtrain.errors import Problem, Refused

from .answer import canonical_label

MAX_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 400_000      # the prompt budget: about 400 image tokens (stages.MAX_PIXELS)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def _refused(rule_field: str, message: str, fix: str) -> Refused:
    return Refused([Problem("VLM-1", rule_field, message, fix)])


def _mime(raw: bytes) -> str | None:
    if raw.startswith(PNG_MAGIC):
        return "image/png"
    if raw.startswith(JPEG_MAGIC):
        return "image/jpeg"
    return None


def _prepare(raw: bytes, jpeg_quality: int | None) -> tuple[bytes, str, bool] | str:
    """(bytes, mime, resized) of the image to embed, or the reason it is refused. An image over `MAX_PIXELS` is scaled
    down, keeping its aspect ratio, so it fits the prompt budget."""
    mime = _mime(raw)
    if mime is None:
        return "the image is neither PNG nor JPEG"
    try:
        from PIL import Image
    except ImportError:
        if jpeg_quality is not None:
            raise _refused("jpeg_quality", "re-encoding needs Pillow, which is not installed",
                           "install the extra `convert` or pass --keep-images") from None
        return raw, mime, False
    try:
        img = Image.open(io.BytesIO(raw), formats=["PNG", "JPEG"])
        if img.size[0] * img.size[1] > 8 * MAX_PIXELS:
            return "the image is far larger than the pixel budget"
        img.load()
    except Exception:
        return "the image cannot be decoded"
    resized = img.size[0] * img.size[1] > MAX_PIXELS
    if resized:
        scale = (MAX_PIXELS / (img.size[0] * img.size[1])) ** 0.5
        img = img.convert("RGB").resize((max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale))),
                                        Image.LANCZOS)
    if jpeg_quality is None and not resized:
        return raw, mime, False
    buf = io.BytesIO()
    if jpeg_quality is None:
        img.save(buf, "PNG")
        return buf.getvalue(), "image/png", True
    img.convert("RGB").save(buf, "JPEG", quality=jpeg_quality)
    return buf.getvalue(), "image/jpeg", resized


def _uri(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def _read_image(images_dir: Path, rel: object) -> bytes | str:
    """The bytes of an image under `images_dir`: a relative path whose every component is a real directory or file
    (no link anywhere on it), at most `MAX_IMAGE_BYTES`."""
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) or ".." in Path(rel).parts:
        return "the image is not a relative path inside the image directory"
    parts = Path(rel).parts
    try:
        cur = os.open(images_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in parts[:-1]:
                nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur)
                os.close(cur)
                cur = nxt
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=cur)
        finally:
            os.close(cur)
        with os.fdopen(fd, "rb") as f:
            if os.fstat(fd).st_size > MAX_IMAGE_BYTES:
                return "the image file is larger than the limit"
            return f.read(MAX_IMAGE_BYTES + 1)
    except OSError:
        return "the image file cannot be read"


def convert(src: Path, images_dir: Path, dst: Path, *, jpeg_quality: int | None = 90,
            max_bytes: int = MAX_BYTES) -> dict[str, Any]:
    """Write `dst` and return `{"rows", "duplicates", "refused", "bytes"}`. When any row is refused nothing is
    written; `resized` counts images scaled down to the pixel budget. Refused (VLM-1): an existing destination, an unreadable or empty source, a result over `max_bytes`
    (the message gives how many rows fit)."""
    dst, images_dir = Path(dst), Path(images_dir)
    if os.path.lexists(dst):
        raise _refused("dst", "the destination file already exists", "choose a new --dst")
    try:
        text = Path(src).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _refused("src", f"cannot read the source as UTF-8 ({type(exc).__name__})",
                       "pass a UTF-8 JSONL file") from None
    lines = text.split("\n")        # never splitlines(): a question may hold U+2028
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise _refused("src", "the source file holds no row (it is empty)",
                       "pass a JSONL file, one {question, answer, image} object per line")
    seen: dict[tuple[str, str], str] = {}
    out: list[bytes] = []
    refused: list[dict[str, Any]] = []
    duplicates = 0
    resized = 0
    for no, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except ValueError:
            refused.append({"line": no, "reason": "not valid JSON"})
            continue
        if not (isinstance(row, dict) and isinstance(row.get("question"), str)
                and isinstance(row.get("answer"), str)):
            refused.append({"line": no, "reason": "not an object with string 'question' and 'answer'"})
            continue
        label = canonical_label(row["answer"])
        if label is None:
            refused.append({"line": no, "reason": "the answer is not an integer, number word or yes/no"})
            continue
        raw = _read_image(images_dir, row.get("image"))
        if isinstance(raw, str):
            refused.append({"line": no, "reason": raw})
            continue
        prepared = _prepare(raw, jpeg_quality)
        if isinstance(prepared, str):
            refused.append({"line": no, "reason": prepared})
            continue
        img_bytes, mime, was_resized = prepared
        resized += was_resized
        key = (hashlib.sha256(raw).hexdigest(), row["question"])
        if key in seen:
            if seen[key] != label:
                refused.append({"line": no, "reason": "the same image and question have a different label earlier"})
            else:
                duplicates += 1
            continue
        seen[key] = label
        out.append((json.dumps({"input": {"question": row["question"], "image": _uri(img_bytes, mime)},
                                "label": label}, ensure_ascii=False) + "\n").encode("utf-8"))
    if refused:
        return {"rows": 0, "duplicates": duplicates, "refused": refused, "bytes": 0, "resized": resized}
    total = sum(len(x) for x in out)
    if total > max_bytes:
        fit, used = 0, 0
        for x in out:
            if used + len(x) > max_bytes:
                break
            used += len(x)
            fit += 1
        raise _refused("max_bytes", f"the data would be {total} bytes, over the {max_bytes} byte limit; {fit} of "
                       f"{len(out)} rows fit",
                       "use fewer rows, or JPEG re-encoding (a lower --jpeg-quality), or smaller images")
    try:
        fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    except FileExistsError:
        raise _refused("dst", "the destination file already exists", "choose a new --dst") from None
    with os.fdopen(fd, "wb") as f:
        f.write(b"".join(out))
    os.chmod(dst, 0o644)
    return {"rows": len(out), "duplicates": duplicates, "refused": [], "bytes": total, "resized": resized}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gatedtrain-pack-vlm-clevr-data", description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, type=Path, help="items JSONL ({question, answer, image})")
    ap.add_argument("--images", required=True, type=Path, help="directory the image paths are relative to")
    ap.add_argument("--dst", required=True, type=Path, help="intake JSONL to create (must not exist)")
    ap.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality of the embedded images (default 90)")
    ap.add_argument("--keep-images", action="store_true", help="embed the source image bytes unchanged")
    args = ap.parse_args(argv)
    try:
        summary = convert(args.src, args.images, args.dst, jpeg_quality=None if args.keep_images else args.jpeg_quality)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print(json.dumps(summary))
    return 3 if summary["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
