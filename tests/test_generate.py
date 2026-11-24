import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from gatedtrain.ainative.batch import read_predictions

import gatedtrain_pack_vlm_clevr

PKG = Path(gatedtrain_pack_vlm_clevr.__file__).parent
SCRIPT = PKG / "generate_vllm.py"
FAKE = Path(__file__).parent / "fakevllm"
SAMPLER = {"name": "vllm", "temperature": 0.7, "top_p": 1.0, "max_new_tokens": 512}
QUESTIONS = ["How many cubes?", "Is the sphere red?", "Héllo – how many?"]


def uri(size, kind="png") -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, "red").save(buf, kind.upper() if kind == "png" else "JPEG")
    return f"data:image/{kind};base64," + base64.b64encode(buf.getvalue()).decode()


def _canonical(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = self.tmp / "out"
        self.out.mkdir()
        self.model = self.tmp / "model"
        self.model.mkdir()
        self.record = self.tmp / "record.jsonl"
        self.write_items([{"question": q, "image": uri((10 + i, 8), "png" if i % 2 == 0 else "jpeg")}
                          for i, q in enumerate(QUESTIONS)])

    def write_items(self, inputs):
        (self.tmp / "items.jsonl").write_text(
            "".join(_canonical({"item_id": i, "input": x}) for i, x in enumerate(inputs)), encoding="utf-8")

    def run_script(self, sampler=SAMPLER, seeds=(11,), extra_env=None, timeout=60):
        (self.tmp / "sampler.json").write_text(json.dumps({"sampler": sampler, "seeds": list(seeds)}),
                                               encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(FAKE), "FAKEVLLM_RECORD": str(self.record),
               "HOME": str(self.tmp), **(extra_env or {})}
        argv = [sys.executable, str(SCRIPT), "--items", str(self.tmp / "items.jsonl"),
                "--sampler", str(self.tmp / "sampler.json"), "--model", str(self.model), "--out", str(self.out)]
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env, cwd=self.tmp)

    def events(self, kind):
        if not self.record.exists():
            return []
        docs = [json.loads(x) for x in self.record.read_text(encoding="utf-8").split("\n") if x]
        return [d for d in docs if d["event"] == kind]

    def test_prompt_is_the_recipes_file(self):
        text = (PKG / "runtime" / "prompt_clevr_cot.txt").read_text(encoding="utf-8")
        ns: dict = {}
        exec(compile(SCRIPT.read_text(encoding="utf-8").split("SAMPLER_KEYS")[0].split("import sys\n")[1], "p", "exec"), ns)
        self.assertEqual(ns["PROMPT"], text)

    def test_predictions_contract(self):
        res = self.run_script(seeds=(11, 12))
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(os.listdir(self.out), ["predictions.jsonl"])
        lines = (self.out / "predictions.jsonl").read_text(encoding="utf-8").split("\n")[:-1]
        docs = [json.loads(x) for x in lines]
        self.assertEqual([(d["seed"], d["item_id"]) for d in docs],
                         [(11, 0), (11, 1), (11, 2), (12, 0), (12, 1), (12, 2)])
        self.assertTrue(all(_canonical(d) == x + "\n" for d, x in zip(docs, lines)))
        self.assertEqual(len(read_predictions(self.out, 3, [11, 12])), 6)

    def test_training_message_image_first_and_no_template_kwargs(self):
        res = self.run_script()
        self.assertEqual(res.returncode, 0, res.stderr)
        calls = self.events("chat_template")
        self.assertEqual(len(calls), 3)
        text = (PKG / "runtime" / "prompt_clevr_cot.txt").read_text(encoding="utf-8")
        for call, q in zip(calls, QUESTIONS):
            self.assertEqual(call["kwargs"], {"tokenize": False, "add_generation_prompt": True})
            [msg] = call["messages"]
            self.assertEqual(msg["role"], "user")
            self.assertEqual(msg["content"], [{"type": "image"}, {"type": "text", "text": text.format(q)}])

    def test_engine_args_and_image_sizes(self):
        res = self.run_script()
        self.assertEqual(res.returncode, 0, res.stderr)
        [llm] = self.events("llm")
        self.assertEqual(llm["kwargs"], {"dtype": "bfloat16", "seed": 0, "max_model_len": 1024,
                                         "gpu_memory_utilization": 0.85, "limit_mm_per_prompt": {"image": 1},
                                         "mm_processor_cache_gb": 0})
        [proc] = self.events("processor")
        self.assertEqual(proc["kwargs"], {"local_files_only": True})
        [gen] = self.events("generate")
        self.assertEqual(gen["images"], [[10, 8], [11, 8], [12, 8]])
        self.assertEqual(gen["sampling"], {"temperature": 0.7, "top_p": 1.0, "max_tokens": 512, "seed": 11})

    def test_env_set_before_import(self):
        res = self.run_script()
        self.assertEqual(res.returncode, 0, res.stderr)
        [imp] = self.events("import")
        self.assertEqual(imp["env"]["VLLM_CACHE_ROOT"], "/tmp/cache/vllm")
        self.assertEqual(imp["env"]["HF_HOME"], "/tmp/cache/hf")

    def test_bad_sampler_refused(self):
        bad = [dict(SAMPLER, top_k=20), dict(SAMPLER, enable_thinking=False),
               {k: v for k, v in SAMPLER.items() if k != "temperature"}, dict(SAMPLER, max_new_tokens="512"), "vllm"]
        for sampler in bad:
            with self.subTest(sampler=sampler):
                self.record.unlink(missing_ok=True)
                res = self.run_script(sampler=sampler)
                self.assertEqual(res.returncode, 2)
                self.assertIn("sampler", res.stderr)
                self.assertEqual(os.listdir(self.out), [])
                self.assertEqual(self.events("import"), [])

    def test_bad_items_refused_before_engine(self):
        for bad in ({"question": "q", "image": "data:image/gif;base64,AAAA"},
                    {"question": "q", "image": "data:image/png;base64,@@@"},
                    {"question": "q"}, "just text"):
            with self.subTest(bad=bad):
                self.record.unlink(missing_ok=True)
                self.write_items([bad])
                res = self.run_script()
                self.assertEqual(res.returncode, 2)
                self.assertIn("refused", res.stderr)
                self.assertNotIn("just text", res.stderr)
                self.assertEqual(os.listdir(self.out), [])
                self.assertEqual(self.events("import"), [])

    def test_hostile_images_refused_before_engine(self):
        import struct
        def raw_uri(raw):
            return "data:image/png;base64," + base64.b64encode(raw).decode()
        big = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 30000, 30000) + b"\x08\x02\x00\x00\x00"
        for name, image in (("not-png-bytes", raw_uri(b"GIF89a....")), ("bomb", raw_uri(big)),
                            ("truncated", raw_uri(b"\x89PNG\r\n\x1a\nshort"))):
            with self.subTest(name):
                self.record.unlink(missing_ok=True)
                self.write_items([{"question": "q", "image": image}])
                res = self.run_script()
                self.assertEqual(res.returncode, 2, res.stderr)
                self.assertIn("refused", res.stderr)
                self.assertEqual(os.listdir(self.out), [])
                self.assertEqual(self.events("import"), [])

    def test_malformed_item_lines_refused_cleanly(self):
        for text in ('{"item_id": 1}\n', "not json\n", '{"item_id": 1, "input": null}\n'):
            with self.subTest(text=text):
                (self.tmp / "items.jsonl").write_text(text)
                res = self.run_script()
                self.assertEqual(res.returncode, 2, res.stderr)
                self.assertNotIn("Traceback", res.stderr)

    def test_exits_without_engine_teardown(self):
        res = self.run_script(extra_env={"FAKEVLLM_EXIT_HANG_S": "600"}, timeout=30)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self.events("atexit"), [])

    def test_short_output_is_an_error(self):
        res = self.run_script(extra_env={"FAKEVLLM_SHORT": "1"})
        self.assertEqual(res.returncode, 1)
        self.assertEqual(os.listdir(self.out), [])


if __name__ == "__main__":
    unittest.main()
