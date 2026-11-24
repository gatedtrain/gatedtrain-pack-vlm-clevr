import base64
import io
import json
import os
import tempfile
from unittest import mock
import unittest
from pathlib import Path

from PIL import Image

from gatedtrain.errors import Refused
from gatedtrain_pack_vlm_clevr import data


def png(color, size=(32, 24)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


class ConvertTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        (self.tmp / "images").mkdir()
        self.dst = self.tmp / "out.jsonl"

    def source(self, rows, images=None):
        for name, raw in (images or {"a.png": png("red"), "b.png": png("blue")}).items():
            (self.tmp / "images" / name).write_bytes(raw)
        src = self.tmp / "items.jsonl"
        src.write_bytes("".join(json.dumps(r) + "\n" for r in rows).encode())
        return src

    def convert(self, src, **kw):
        return data.convert(src, self.tmp / "images", self.dst, **kw)

    def read(self):
        return [json.loads(x) for x in self.dst.read_text().split("\n") if x]

    def test_happy_path_png_kept(self):
        src = self.source([{"question": "How many?", "answer": "03", "image": "a.png"},
                           {"question": "Is it red?", "answer": "Yes", "image": "b.png"}])
        out = self.convert(src, jpeg_quality=None)
        self.assertEqual((out["rows"], out["duplicates"], out["refused"]), (2, 0, []))
        rows = self.read()
        self.assertEqual(rows[0]["label"], "3")
        self.assertEqual(rows[1]["label"], "yes")
        uri = rows[0]["input"]["image"]
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(uri.split(",", 1)[1]), png("red"))
        self.assertEqual(rows[0]["input"]["question"], "How many?")

    def test_jpeg_reencode(self):
        src = self.source([{"question": "q", "answer": "1", "image": "a.png"}])
        self.convert(src, jpeg_quality=90)
        uri = self.read()[0]["input"]["image"]
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        img = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        self.assertEqual(img.size, (32, 24))

    def test_duplicates_collapsed_conflict_refused(self):
        rows = [{"question": "q", "answer": "1", "image": "a.png"},
                {"question": "q", "answer": "1", "image": "a.png"}]
        out = self.convert(self.source(rows))
        self.assertEqual((out["rows"], out["duplicates"]), (1, 1))
        self.dst.unlink()
        rows[1]["answer"] = "2"
        out = self.convert(self.source(rows))
        self.assertEqual(out["rows"], 0)
        self.assertEqual(out["refused"][0]["line"], 2)
        self.assertFalse(self.dst.exists())

    def test_same_question_other_image_is_not_duplicate(self):
        rows = [{"question": "q", "answer": "1", "image": "a.png"},
                {"question": "q", "answer": "2", "image": "b.png"}]
        self.assertEqual(self.convert(self.source(rows))["rows"], 2)

    def test_refusals_by_line_without_content(self):
        rows = [{"question": "q", "answer": "maybe", "image": "a.png"},
                {"question": "q2", "answer": "1", "image": "missing.png"},
                {"question": "q3", "answer": "1", "image": "../x.png"},
                {"question": "q4", "answer": "1"},
                {"question": "q5", "answer": "1", "image": "bad.png"}]
        src = self.source(rows, {"a.png": png("red"), "bad.png": b"not an image"})
        out = self.convert(src)
        self.assertEqual([r["line"] for r in out["refused"]], [1, 2, 3, 4, 5])
        self.assertEqual(out["rows"], 0)
        self.assertNotIn("maybe", json.dumps(out))
        self.assertFalse(self.dst.exists())

    def test_over_cap_refused_with_fitting_count(self):
        rows = [{"question": f"q{i}", "answer": "1", "image": "a.png"} for i in range(10)]
        src = self.source(rows)
        with self.assertRaises(Refused) as cm:
            self.convert(src, jpeg_quality=None, max_bytes=1000)
        self.assertIn("rows fit", str(cm.exception))
        self.assertFalse(self.dst.exists())

    def test_large_image_is_downscaled_to_the_pixel_bound(self):
        src = self.source([{"question": "q", "answer": "1", "image": "big.png"}],
                          {"big.png": png("red", size=(1600, 800))})
        out = self.convert(src, jpeg_quality=None)
        self.assertEqual(out["resized"], 1)
        uri = self.read()[0]["input"]["image"]
        img = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        self.assertLessEqual(img.size[0] * img.size[1], data.MAX_PIXELS)
        self.assertEqual(img.size[0] / img.size[1], 2.0)

    def test_duplicate_detected_on_source_bytes_not_reencoded_bytes(self):
        rows = [{"question": "q", "answer": "1", "image": "a.png"}, {"question": "q", "answer": "1", "image": "a2.png"}]
        out = self.convert(self.source(rows, {"a.png": png("red"), "a2.png": png("red")}), jpeg_quality=90)
        self.assertEqual((out["rows"], out["duplicates"]), (1, 1))

    def test_cli_exit_codes(self):
        src = self.source([{"question": "q", "answer": "maybe", "image": "a.png"}])
        argv = ["--src", str(src), "--images", str(self.tmp / "images"), "--dst", str(self.dst)]
        self.assertEqual(data.main(argv), 3)
        src = self.source([{"question": "q", "answer": "1", "image": "a.png"}])
        self.assertEqual(data.main(argv), 0)

    def test_over_cap_message_gives_the_count(self):
        rows = [{"question": f"q{i}", "answer": "1", "image": "a.png"} for i in range(10)]
        with self.assertRaises(Refused) as cm:
            self.convert(self.source(rows), jpeg_quality=None, max_bytes=1000)
        self.assertRegex(str(cm.exception), r"\d+ of 10 rows fit")

    def test_oversized_image_file_refused(self):
        src = self.source([{"question": "q", "answer": "1", "image": "a.png"}])
        with mock.patch.object(data, "MAX_IMAGE_BYTES", 10):
            out = self.convert(src)
        self.assertEqual(out["refused"][0]["line"], 1)

    def test_symlinked_directory_in_the_image_path_refused(self):
        (self.tmp / "elsewhere").mkdir()
        (self.tmp / "elsewhere" / "a.png").write_bytes(png("red"))
        os.symlink(self.tmp / "elsewhere", self.tmp / "images" / "link")
        src = self.source([{"question": "q", "answer": "1", "image": "link/a.png"}], {})
        self.assertEqual(self.convert(src)["refused"][0]["line"], 1)

    def test_destination_exists_refused(self):
        self.dst.write_text("x")
        with self.assertRaises(Refused):
            self.convert(self.source([{"question": "q", "answer": "1", "image": "a.png"}]))

    def test_line_separator_in_question_survives(self):
        src = self.source([{"question": "a b", "answer": "1", "image": "a.png"}])
        self.convert(src)
        self.assertEqual(self.read()[0]["input"]["question"], "a b")

    def test_empty_source_refused(self):
        src = self.tmp / "e.jsonl"
        src.write_bytes(b"")
        with self.assertRaises(Refused):
            self.convert(src)


if __name__ == "__main__":
    unittest.main()
