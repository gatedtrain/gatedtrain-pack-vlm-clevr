"""The recipe's cut writer (`runtime/ptc_vlm_hooks.py write_cut`) on a NeMo RL step directory laid out as NeMo RL
writes it, outside any container, and the stage that consumes its cut. The first GPU job found nothing wrong with
fakes; this runs the real writer (the lesson: test copied runtime code on the new input shapes before the GPU)."""

import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import gatedtrain_pack_vlm_clevr
from gatedtrain_pack_vlm_clevr import stages

RUNTIME = Path(gatedtrain_pack_vlm_clevr.__file__).resolve().parent / "runtime"


def load_hooks():
    spec = importlib.util.spec_from_file_location("pack_ptc_vlm_hooks", RUNTIME / "ptc_vlm_hooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class WriteCutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base = self.tmp / "inputs" / "base-model"          # as the job mounts it: no MODELS.json beside it
        self.base.mkdir(parents=True)
        for n in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja", "config.json"):
            (self.base / n).write_text(f"base {n}\n")
        step = self.tmp / "nemo" / "step_60"
        cons = step / "policy" / "weights" / "model" / "consolidated"
        tok = step / "policy" / "tokenizer"
        cons.mkdir(parents=True)
        tok.mkdir(parents=True)
        (cons / "model-00001-of-00001.safetensors").write_bytes(b"weights")
        (cons / "model.safetensors.index.json").write_text("{}")
        (cons / "config.json").write_text("{}")
        for n in ("tokenizer.json", "tokenizer_config.json"):
            (tok / n).write_text(f"tok {n}\n")
        self.step, self.out = step, self.tmp / "out"
        self.out.mkdir()

    def write(self):
        hooks = load_hooks()
        meta = {"seed": 7100, "profile": "full", "stage": "vlm_grpo", "init_checkpoint": None,
                "script_sha256": "sha256:" + "0" * 64, "examples_consumed": lambda s: s * 16}
        return hooks.write_cut(self.step, 60, self.out, self.base, meta)

    def test_writes_a_cut_without_the_recipes_data_directory(self):
        final = self.write()
        cut = json.loads((final / "cut.json").read_text())
        self.assertEqual(cut["global_step"], 60)
        self.assertIn("sampler.json", cut["deployable"])
        self.assertNotIn("base_ref.json", cut["deployable"])
        for name, sha in cut["files"].items():
            self.assertEqual("sha256:" + hashlib.sha256((final / name).read_bytes()).hexdigest(), sha)

    def test_sampler_is_the_registered_deployment_sampler(self):
        final = self.write()
        self.assertEqual(json.loads((final / "sampler.json").read_text()), stages.SAMPLER)

    def test_base_ref_hashes_the_base_files(self):
        final = self.write()
        ref = json.loads((final / "base_ref.json").read_text())
        self.assertEqual(ref["files"]["config.json"],
                         "sha256:" + hashlib.sha256((self.base / "config.json").read_bytes()).hexdigest())
        self.assertIsNone(ref["init_checkpoint"])

    def test_the_stage_accepts_the_real_cut(self):
        final = self.write()
        for d in (self.out / "cuts", final):
            os.chmod(d, 0o700)
        names = stages.Train._open_cut(self.out, 60)
        self.assertEqual(sorted(names), sorted(json.loads((final / "cut.json").read_text())["deployable"]))
        self.assertTrue(os.stat(final / "config.json").st_mode & 0o004)


if __name__ == "__main__":
    unittest.main()
