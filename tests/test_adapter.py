import ast
import difflib
import hashlib
import os
import unittest
from pathlib import Path
from unittest import mock

from gatedtrain import spec
from gatedtrain.errors import Refused
from gatedtrain.ext import conformance, registry

import gatedtrain_pack_vlm_clevr
from gatedtrain_pack_vlm_clevr import adapter

PACKAGE = Path(gatedtrain_pack_vlm_clevr.__file__).resolve().parent
RECIPE = Path(os.environ.get("GATEDTRAIN_RECIPE_VLM_CLEVR_DIR")
              or PACKAGE.parents[2] / "gatedtrain-recipe-nemo-rl-vlm-clevr")
_PATCHES = []


def setUpModule():
    for table in (registry.ADAPTERS, registry.TRAINER_ROWS, registry.EVALUATOR_ROWS, spec.TRAINER_BACKENDS,
                  spec.EVALUATOR_BACKENDS):
        _PATCHES.append(mock.patch.dict(table))
        _PATCHES[-1].start()
    adapter.install()


def tearDownModule():
    while _PATCHES:
        _PATCHES.pop().stop()


class AdapterTests(unittest.TestCase):
    def test_conforms(self):
        self.assertEqual(conformance.adapter_problems(adapter.NAME), [])

    def test_install_is_idempotent(self):
        adapter.install()
        adapter.install()
        self.assertEqual(registry.ADAPTERS[adapter.NAME], adapter.PLUGIN["adapters"][adapter.NAME])

    def test_row(self):
        row = registry.TRAINER_ROWS[adapter.NAME]
        self.assertEqual(row["uses"], ("rl.nemo_rl_vlm@0.1",))
        self.assertEqual(row["evaluators"], (adapter.EVALUATOR_ID,))
        self.assertIn("{output_root}", row["input_roots"])
        self.assertIn(adapter.RUNTIME, row["input_roots"])
        self.assertFalse(row["spec_binding"])

    def test_run_options_keep_the_rootless_daemons_memlock(self):
        opts = registry.TRAINER_ROWS[adapter.NAME]["run_options"][adapter.SCRIPT]
        self.assertNotIn("memlock=-1", " ".join(opts))
        self.assertIn("--shm-size=16g", opts)
        self.assertEqual(opts[opts.index("-w") + 1], "/opt/nemo-rl")

    def test_scorer_hash_is_answer_py(self):
        want = "sha256:" + hashlib.sha256((PACKAGE / "answer.py").read_bytes()).hexdigest()
        self.assertEqual(registry.EVALUATOR_ROWS[adapter.EVALUATOR_ID]["scorer"]["code_sha256"], want)

    def test_collect_interface_refuses(self):
        a = registry.adapter(adapter.NAME)
        with self.assertRaises(Refused):
            a.find_cuts("x")
        self.assertEqual(a.metric(adapter.EVALUATOR_ID)["metric"], "exact_match")
        with self.assertRaises(KeyError):
            a.metric("other")

    def test_no_framework_imports(self):
        tree = ast.parse((PACKAGE / "adapter.py").read_text())
        names = {n.module.split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
                 for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                 for a in (n.names if isinstance(n, ast.Import) else [n])}
        self.assertFalse(names & {"torch", "vllm", "transformers", "nemo_rl", "ray"})

    def test_hooks_differ_from_the_recipe_only_where_documented(self):
        if not (RECIPE / "runtime").is_dir():
            if os.environ.get("GATEDTRAIN_REQUIRE_CONTAINER_USER"):         # CI: the recipe is checked out
                self.fail("the recipe checkout is required (GATEDTRAIN_RECIPE_VLM_CLEVR_DIR)")
            self.skipTest("the recipe checkout is not beside the pack (GATEDTRAIN_RECIPE_VLM_CLEVR_DIR)")
        old = (RECIPE / "runtime" / "ptc_vlm_hooks.py").read_text().splitlines()
        new = (PACKAGE / "runtime" / "ptc_vlm_hooks.py").read_text().splitlines()
        removed = [l[1:] for l in difflib.unified_diff(old, new, lineterm="", n=0) if l[:1] == "-" and l[:3] != "---"]
        for l in removed:
            self.assertTrue(any(t in l for t in ("gen_vqa", "sys.path.insert", "base_models", "base = ", "repo",
                                                 "revision", "init_checkpoint", "MODELS")), l)
        self.assertLessEqual(len(removed), 8, removed)

    def test_packaged_scripts_match_the_recipe(self):
        if not (RECIPE / "runtime").is_dir():
            if os.environ.get("GATEDTRAIN_REQUIRE_CONTAINER_USER"):         # CI: the recipe is checked out
                self.fail("the recipe checkout is required (GATEDTRAIN_RECIPE_VLM_CLEVR_DIR)")
            self.skipTest("the recipe checkout is not beside the pack (GATEDTRAIN_RECIPE_VLM_CLEVR_DIR)")
        for name in ("vlm_env_logged.py", "prompt_clevr_cot.txt"):
            with self.subTest(name=name):
                self.assertEqual((PACKAGE / "runtime" / name).read_bytes(), (RECIPE / "runtime" / name).read_bytes())
        old = (RECIPE / "runtime" / "run_vlm_nemo.py").read_text().splitlines()
        new = (PACKAGE / "runtime" / "run_vlm_nemo.py").read_text().splitlines()
        changed = [l for l in difflib.unified_diff(old, new, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        removed = [l for l in changed if l[0] == "-"]
        self.assertEqual(len(removed), 6, removed)          # the usage line, DATA, and the four DATA uses
        for l in changed:
            self.assertTrue("DATA" in l or "data" in l or "train" in l or "Deviation" in l or l[1:].strip() == ""
                            or "usage" in l.lower() or "run_vlm_nemo.py --profile" in l or "`<runtime>" in l, l)


if __name__ == "__main__":
    unittest.main()
