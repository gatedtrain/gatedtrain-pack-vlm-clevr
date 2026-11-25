"""The pack stages: profile, base files, train set, recipe, smoke and train, through gatedtrain's `LocalBackend` on
`FakeRuntime` (no Docker, no GPU). The fake job plays `run_vlm_nemo.py` by writing what it writes."""

import base64
import datetime
import hashlib
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gatedtrain import spec
from gatedtrain.ainative.backend import LocalBackend, StageBackend
from gatedtrain.ainative.project import create_project
from gatedtrain.ainative.stage import StageContext
from gatedtrain.ainative.trust_stages import SCHEMA_TRAIN_PART
from gatedtrain.errors import Refused
from gatedtrain.ext import launch, registry
from gatedtrain.testing.fake_runtime import FakeRuntime

from _images import TIERB_IMAGE, add_owner_image, isolate_owner_config
from _pack_fakes import CUT_FILES, play_training, png, write_base
from gatedtrain_pack_vlm_clevr import adapter, stages
from gatedtrain_pack_vlm_clevr.stages import (BuildTrainSet, ChooseTaskAndBase, ProfileData, Smoke, Train,
                                              WriteRecipe)

FAKE_GID = 2**31 - 3
CLOCK = datetime.datetime(2026, 11, 24, tzinfo=datetime.timezone.utc)


def uri(i: int) -> str:
    return "data:image/png;base64," + base64.b64encode(png(i)).decode()


def mode(path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


class _Layout(unittest.TestCase):
    def setUp(self):
        for table in (registry.ADAPTERS, registry.TRAINER_ROWS, registry.EVALUATOR_ROWS, spec.TRAINER_BACKENDS,
                      spec.EVALUATOR_BACKENDS):
            p = mock.patch.dict(table)
            p.start()
            self.addCleanup(p.stop)
        adapter.install()
        patcher = mock.patch.object(stages, "PROMPTS_PER_STEP", 0)    # no row minimum for the small fixtures
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = Path(os.path.realpath(tempfile.mkdtemp(prefix="tmp-job-", dir=Path(__file__).resolve().parent)))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.owner_config = isolate_owner_config(self, self.tmp)
        self.work, self.out_root, self.hub = self.tmp / "work", self.tmp / "out", self.tmp / "hub"
        for d in (self.work, self.out_root):
            d.mkdir()
        self.out_root.chmod(0o755)
        write_base(self.hub)
        self.cfg = launch.HostConfig(host_work=str(self.work), output_root=str(self.out_root),
                                     primary_gid=lambda uid: FAKE_GID, rootless=True)
        data = self.tmp / "data.jsonl"
        data.write_text("{}\n")
        self.proj = create_project(self.tmp / "proj", goal="g", data=data, pipeline="p.yaml",
                                   budget={"gpu_hours": 10.0, "llm_usd": 0})
        self.rewards, self.texts, self.seen = (1.0, 0.0, 0.0, 0.0), None, []
        self.rt = FakeRuntime({TIERB_IMAGE: self.play})
        self.backend = LocalBackend(self.proj, self.rt, host=self.cfg, sleep=lambda s: None, clock=lambda: CLOCK)
        self.params = {"image": TIERB_IMAGE}

    def play(self, run_spec, host_path) -> int:
        self.seen.append(host_path("/inputs/train.jsonl").read_bytes())
        return play_training(run_spec, host_path, rewards=self.rewards, texts=self.texts)

    def ctx(self, ref, inputs, params=None, *, gpu=False):
        sb = StageBackend(self.backend, stage=ref, allow_gpu=gpu, planning=False)
        return StageContext(self.proj, inputs, dict(params or {}), None, None, stage=ref, backend=sb)

    def train_part(self, n=12, rows=None):
        rows = rows or [{"input": {"question": f"How many cubes {i}?", "image": uri(i % 5)},
                         "label": ["yes", "3", "03"][i % 3]} for i in range(n)]
        rec = {"schema": SCHEMA_TRAIN_PART, "schema_version": 1, "n": len(rows), "rows": rows}
        return self.proj.write_artifact("intake_seal", "train_part", rec, SCHEMA_TRAIN_PART)

    def base(self, tp=None):
        tp = tp or self.train_part()
        prof = ProfileData().run(self.ctx("profile_data", {"train_part": tp}), {})["profile"]
        ctx = self.ctx("choose_task_and_base", {"profile": prof}, {"base_model_dir": str(self.hub)})
        return tp, ctx, ChooseTaskAndBase().run(ctx, {})

    def upto_recipe(self, **params):
        tp, _, out = self.base()
        ts = BuildTrainSet().run(self.ctx("build_train_set", {"train_part": tp}), {})["train_set"]
        recipe = WriteRecipe().run(self.ctx("write_recipe", {"train_set": ts}, params), {})["recipe"]
        return {"base": out["base"], "train_set": ts, "recipe": recipe}

    def smoke(self, inputs):
        ctx = self.ctx("smoke", inputs, self.params, gpu=True)
        return ctx, Smoke().run(ctx, {})

    def train(self, inputs, smoke_ref):
        ctx = self.ctx("train", {**inputs, "smoke": smoke_ref}, self.params, gpu=True)
        return ctx, Train().run(ctx, {})

    def assertRule(self, cm, rule, *words):
        self.assertEqual({p.rule for p in cm.exception.problems}, {rule}, str(cm.exception))
        for w in words:
            self.assertIn(w, str(cm.exception))


class ProfileTests(_Layout):
    def test_profile(self):
        tp = self.train_part(12)
        ctx = self.ctx("profile_data", {"train_part": tp})
        out = ProfileData().run(ctx, {})
        p = self.proj.read_artifact(out["profile"])
        self.assertEqual((p["n"], p["n_distinct_images"], p["n_duplicate_keys"]), (12, 5, 0))
        self.assertEqual(p["labels"], {"3": 8, "yes": 4})
        self.assertEqual(ProfileData().check(ctx, out), [])
        self.assertIn("12 training rows", ProfileData().plan(ctx).summary)

    def test_duplicate_keys_counted(self):
        row = {"input": {"question": "same?", "image": uri(1)}, "label": "1"}
        other = {"input": {"question": "same?", "image": uri(2)}, "label": "1"}
        tp = self.train_part(rows=[row, dict(row), other])
        p = self.proj.read_artifact(ProfileData().run(self.ctx("profile_data", {"train_part": tp}), {})["profile"])
        self.assertEqual((p["n"], p["n_distinct_images"], p["n_duplicate_keys"]), (3, 2, 1))

    def test_percentiles_and_single_row(self):
        tp = self.train_part(rows=[{"input": {"question": "abc", "image": uri(1)}, "label": "7"}])
        p = self.proj.read_artifact(ProfileData().run(self.ctx("profile_data", {"train_part": tp}), {})["profile"])
        self.assertEqual(p["question_chars"], {"min": 3, "max": 3, "p50": 3, "p90": 3, "p99": 3})
        self.assertEqual(p["labels"], {"7": 1})

    def test_refusals_name_indexes_not_content(self):
        good = {"input": {"question": "SECRET?", "image": uri(1)}, "label": "1"}
        cases = {
            "shape": {"input": "just text", "label": "1"},
            "label": {**good, "label": "maybe"},
            "image": {"input": {"question": "q", "image": "data:image/gif;base64,AAAA"}, "label": "1"},
            "magic": {"input": {"question": "q", "image": "data:image/png;base64," + base64.b64encode(b"nope").decode()},
                      "label": "1"},
            "b64": {"input": {"question": "q", "image": "data:image/png;base64,@@"}, "label": "1"},
        }
        for name, bad in cases.items():
            with self.subTest(name):
                tp = self.train_part(rows=[good, bad])
                with self.assertRaises(Refused) as cm:
                    ProfileData().plan(self.ctx("profile_data", {"train_part": tp}))
                self.assertRule(cm, "VLM-1", "first row index from 0: 1")
                self.assertNotIn("SECRET", str(cm.exception))


class BaseTests(_Layout):
    def test_copies_allowed_files_and_check(self):
        _, ctx, out = self.base()
        base = self.proj.read_artifact(out["base"])
        names = sorted(Path(f["path"]).name for f in base["files"])
        self.assertNotIn("LICENSE", names)
        self.assertIn("preprocessor_config.json", names)
        self.assertEqual(len(names), 10)
        self.assertTrue(all(mode(f["path"]) == 0o644 for f in base["files"]))
        self.assertEqual(ChooseTaskAndBase().check(ctx, out), [])
        dep = self.proj.read_artifact(out["deployment"])
        self.assertEqual((dep["sampler"], dep["eval_seeds"]), (stages.SAMPLER, [11]))
        self.assertEqual(stages.SAMPLER, {"name": "vllm", "decoding": "greedy", "temperature": 0, "top_p": 1.0,
                                          "max_new_tokens": 512})

    def test_sampled_decoding_stays_available(self):
        tp = self.train_part()
        prof = ProfileData().run(self.ctx("profile_data", {"train_part": tp}), {})["profile"]
        ctx = self.ctx("choose_task_and_base", {"profile": prof}, {"base_model_dir": str(self.hub),
                                                                   "decoding": "sampled"})
        out = ChooseTaskAndBase().run(ctx, {})
        dep = self.proj.read_artifact(out["deployment"])
        self.assertEqual((dep["sampler"], dep["eval_seeds"]), (stages.SAMPLED_SAMPLER, [11, 12, 13, 14]))
        self.assertEqual(stages.SAMPLED_SAMPLER["temperature"], 0.7)
        self.assertEqual(ChooseTaskAndBase().check(ctx, out), [])
        greedy_ctx = self.ctx("choose_task_and_base", {"profile": prof}, {"base_model_dir": str(self.hub)})
        self.assertEqual(len(ChooseTaskAndBase().check(greedy_ctx, out)), 1)     # the deployment is not the chosen one

    def test_unknown_decoding_refused(self):
        tp = self.train_part()
        prof = ProfileData().run(self.ctx("profile_data", {"train_part": tp}), {})["profile"]
        with self.assertRaises(Refused) as cm:
            ChooseTaskAndBase().plan(self.ctx("c", {"profile": prof}, {"base_model_dir": str(self.hub),
                                                                      "decoding": "beam"}))
        self.assertRule(cm, "VLM-2", "decoding")

    def test_refusals(self):
        tp = self.train_part()
        prof = ProfileData().run(self.ctx("profile_data", {"train_part": tp}), {})["profile"]
        def attempt(**params):
            return ChooseTaskAndBase().plan(self.ctx("c", {"profile": prof}, params))
        with self.assertRaises(Refused) as cm:
            attempt()
        self.assertRule(cm, "VLM-2", "base_model_dir")
        (self.hub / "preprocessor_config.json").unlink()
        with self.assertRaises(Refused) as cm:
            attempt(base_model_dir=str(self.hub))
        self.assertRule(cm, "VLM-2", "preprocessor_config.json")
        (self.hub / "preprocessor_config.json").write_text("{}")
        (self.hub / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
        with self.assertRaises(Refused) as cm:
            attempt(base_model_dir=str(self.hub))
        self.assertRule(cm, "VLM-2", stages.ARCHITECTURE)


class TrainSetTests(_Layout):
    def test_images_and_rows(self):
        tp = self.train_part(12)
        ctx = self.ctx("build_train_set", {"train_part": tp})
        out = BuildTrainSet().run(ctx, {})
        ts = self.proj.read_artifact(out["train_set"])
        self.assertEqual((ts["n"], ts["n_images"]), (12, 5))
        images = Path(ts["images_dir"])
        self.assertEqual(len(list(images.iterdir())), 5)
        rows = [json.loads(x) for x in (Path(ts["directory"]) / "train.jsonl").read_text().split("\n") if x]
        self.assertEqual(len(rows), 12)
        first = rows[0]
        self.assertEqual(first["row_id"], "clevr-train-00000")
        img = Path(first["messages"][0]["content"][0]["image"])
        self.assertEqual(img.parent, images)
        self.assertEqual("sha256:" + hashlib.sha256(img.read_bytes()).hexdigest(), first["image_sha256"])
        self.assertEqual(first["messages"][0]["content"][1], {"type": "text", "text": "How many cubes 0?"})
        self.assertEqual(first["messages"][1]["content"], [{"type": "text", "text": "yes"}])
        self.assertEqual(rows[2]["messages"][1]["content"][0]["text"], "3")          # "03" canonical
        self.assertEqual(BuildTrainSet().check(ctx, out), [])

    def test_check_detects_changed_file(self):
        tp = self.train_part()
        ctx = self.ctx("build_train_set", {"train_part": tp})
        out = BuildTrainSet().run(ctx, {})
        ts = self.proj.read_artifact(out["train_set"])
        with open(Path(ts["directory"]) / "train.jsonl", "ab") as f:
            f.write(b"\n")
        self.assertEqual([p.rule for p in BuildTrainSet().check(ctx, out)], ["VLM-1"])

    def test_rerun_keeps_files(self):
        tp = self.train_part()
        ctx = self.ctx("build_train_set", {"train_part": tp})
        ts = self.proj.read_artifact(BuildTrainSet().run(ctx, {})["train_set"])
        before = os.stat(Path(ts["directory"]) / "train.jsonl").st_ino
        BuildTrainSet().run(ctx, {})
        self.assertEqual(os.stat(Path(ts["directory"]) / "train.jsonl").st_ino, before)


class MessageParityTests(_Layout):
    def test_training_row_and_generator_message_agree(self):
        """The row `train_rows` writes, formatted with the prompt NeMo RL applies, is the message the generator sends:
        image part first, then the recipe's prompt around the same question."""
        import importlib.util
        from gatedtrain_pack_vlm_clevr import job as job_mod
        gen_path = job_mod.PACKAGE_DIR / "generate_vllm.py"
        gen = importlib.util.module_from_spec(importlib.util.spec_from_file_location("gen_for_parity", gen_path))
        gen.__spec__.loader.exec_module(gen)
        prompt = (job_mod.PACKAGE_DIR / "runtime" / "prompt_clevr_cot.txt").read_text(encoding="utf-8")
        self.assertEqual(gen.PROMPT, prompt)
        question = "Is the red cube left of the sphere?"
        rows = [{"input": {"question": question, "image": uri(3)}, "label": "yes"}]
        [row] = stages.train_rows(rows, "/data/images")
        user = row["messages"][0]["content"]
        self.assertEqual([p["type"] for p in user], ["image", "text"])
        self.assertEqual(prompt.format(user[1]["text"]), gen.PROMPT.format(question))

    def test_cut_picking_ignores_partial_and_malformed_directories(self):
        self.assertEqual(stages._cut_step("checkpoint-60"), 60)
        for name in (".checkpoint-60.partial", "checkpoint-6a", "checkpoint-", "step_60"):
            self.assertIsNone(stages._cut_step(name))


class RecipeTests(_Layout):
    def test_recipe_and_params(self):
        inputs = self.upto_recipe()
        r = self.proj.read_artifact(inputs["recipe"])
        self.assertEqual((r["profile_steps"], r["memory_rung"], r["seed"]), ({"smoke": 5, "full": 60}, "a", 7100))
        ctx = self.ctx("write_recipe", {"train_set": inputs["train_set"]})
        self.assertEqual(WriteRecipe().check(ctx, {"recipe": inputs["recipe"]}), [])
        for params in ({"profile_steps": {"smoke": 5, "full": 100}}, {"memory_rung": "z"}):
            with self.subTest(params), self.assertRaises(Refused) as cm:
                WriteRecipe().plan(self.ctx("write_recipe", {"train_set": inputs["train_set"]}, params))
            self.assertRule(cm, "VLM-5")


class RowCountTests(_Layout):
    def test_too_few_rows_for_the_steps_refused_before_any_gpu(self):
        with mock.patch.object(stages, "PROMPTS_PER_STEP", 16):
            tp = self.train_part(500)
            _, _, out = self.base(tp)
            ts = BuildTrainSet().run(self.ctx("build_train_set", {"train_part": tp}), {})["train_set"]
            with self.assertRaises(Refused) as cm:
                WriteRecipe().plan(self.ctx("write_recipe", {"train_set": ts}))
        self.assertRule(cm, "VLM-5", "960", "500")


class ImageBoundTests(_Layout):
    def test_oversized_and_unreadable_dimensions_refused(self):
        import struct
        big = stages.PNG_MAGIC + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 1024, 1024) + b"\x08\x02\x00\x00\x00"
        for name, raw in (("big", big), ("truncated", stages.PNG_MAGIC + b"short")):
            with self.subTest(name):
                good = {"input": {"question": "q", "image": uri(1)}, "label": "1"}
                bad = {"input": {"question": "q", "image": "data:image/png;base64," + base64.b64encode(raw).decode()},
                       "label": "1"}
                with self.assertRaises(Refused) as cm:
                    ProfileData().plan(self.ctx("profile_data", {"train_part": self.train_part(rows=[good, bad])}))
                self.assertRule(cm, "VLM-1", "first row index from 0: 1")

    def test_image_size_of_png_and_jpeg(self):
        import io
        from PIL import Image
        for kind in ("PNG", "JPEG"):
            buf = io.BytesIO()
            Image.new("RGB", (37, 21), "red").save(buf, kind)
            self.assertEqual(stages.image_size(buf.getvalue()), (37, 21), kind)
        self.assertIsNone(stages.image_size(b"nope"))


class TrainDataVerificationTests(_Layout):
    def test_swapped_image_refused_at_job_time(self):
        inputs = self.upto_recipe()
        ts = self.proj.read_artifact(inputs["train_set"])
        victim = sorted(Path(ts["images_dir"]).iterdir())[0]
        os.chmod(victim, 0o644)
        victim.write_bytes(stages.PNG_MAGIC + b"swapped")
        with self.assertRaises(Refused) as cm:
            self.smoke(inputs)
        self.assertRule(cm, "VLM-1", "image")

    def test_record_pointing_elsewhere_refused(self):
        inputs = self.upto_recipe()
        ts = dict(self.proj.read_artifact(inputs["train_set"]))
        ts["images_dir"] = str(self.tmp)
        bad = {**inputs, "train_set": self.proj.write_artifact("x", "train_set", ts, stages.SCHEMA_TRAIN_SET)}
        recipe = dict(self.proj.read_artifact(inputs["recipe"]))
        recipe["train_set_sha256"] = bad["train_set"].sha256
        bad["recipe"] = self.proj.write_artifact("y", "recipe", recipe, stages.SCHEMA_RECIPE)
        with self.assertRaises(Refused) as cm:
            self.smoke(bad)
        self.assertRule(cm, "VLM-1")


class LogSizeTests(unittest.TestCase):
    def test_oversized_log_refused(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "env_calls.jsonl").write_bytes(b"x" * 10)
        with mock.patch.object(stages, "MAX_LOG_BYTES", 5):
            with self.assertRaises(Refused) as cm:
                stages._jsonl(tmp, {"env_calls.jsonl": {"sha256": "sha256:" + "0" * 64}}, "env_calls.jsonl")
        self.assertIn("larger than", str(cm.exception))


class JobStageTests(_Layout):
    def test_smoke_then_train(self):
        inputs = self.upto_recipe()
        sctx, smoke = self.smoke(inputs)
        s = self.proj.read_artifact(smoke["smoke"])
        self.assertEqual((s["verdict"], s["n_samples"], s["reward_mean"]), ("PASS", 80, 0.25))
        self.assertEqual(Smoke().check(sctx, smoke), [])
        tctx, model = self.train(inputs, smoke["smoke"])
        m = self.proj.read_artifact(model["model"])
        self.assertEqual((m["label"], m["checkpoint"]), ("candidate", 60))
        self.assertEqual(sorted(Path(f["path"]).name for f in m["files"]), sorted(CUT_FILES))
        self.assertTrue(all(mode(f["path"]) & 0o004 for f in m["files"]))
        self.assertEqual(Train().check(tctx, model), [])
        self.assertEqual(len(self.seen), 2)
        self.assertEqual(self.seen[0], self.seen[1])

    def test_smoke_all_zero_refused(self):
        inputs = self.upto_recipe()
        self.rewards = (0.0,)
        with self.assertRaises(Refused) as cm:
            self.smoke(inputs)
        self.assertRule(cm, "MM-38", "all rewards are 0")

    def test_smoke_one_repeated_completion_refused(self):
        inputs = self.upto_recipe()
        self.texts = ["same"]
        with self.assertRaises(Refused) as cm:
            self.smoke(inputs)
        self.assertRule(cm, "MM-38", "1 distinct completion")

    def test_train_refuses_without_smoke_pass_or_other_inputs(self):
        inputs = self.upto_recipe()
        _, smoke = self.smoke(inputs)
        other = {**inputs, "train_set": self.proj.write_artifact("x", "train_set", {"a": 1}, "gatedtrain_pack_vlm_clevr.train_set")}
        with self.assertRaises(Refused) as cm:
            Train().plan(self.ctx("train", {**other, "smoke": smoke["smoke"]}, self.params, gpu=True))
        self.assertRule(cm, "MM-38")

    def test_bad_image_param_refused(self):
        inputs = self.upto_recipe()
        for image in (None, "", "docker.io/library/python:3.12"):
            with self.subTest(image=image), self.assertRaises(Refused):
                Smoke().plan(self.ctx("smoke", inputs, {"image": image}, gpu=True))

    def test_stage_accepts_a_derived_image_the_owner_added(self):
        inputs = self.upto_recipe()
        derived = "sha256:" + "cd" * 32
        with self.assertRaises(Refused) as cm:
            Smoke().plan(self.ctx("smoke", inputs, {"image": derived}, gpu=True))
        self.assertIn("gatedtrain image add", cm.exception.problems[0].fix)
        add_owner_image(self.owner_config, "nemo-rl/tierb-rootless", derived)
        Smoke().plan(self.ctx("smoke", inputs, {"image": derived}, gpu=True))

    def test_cut_hash_mismatch_refused(self):
        inputs = self.upto_recipe()
        _, smoke = self.smoke(inputs)
        real = stages.Train._open_cut

        def corrupt(top, last):
            (top / "cuts" / f"checkpoint-{last}" / "config.json").write_text("tampered")
            return real(top, last)
        with mock.patch.object(stages.Train, "_open_cut", staticmethod(corrupt)):
            with self.assertRaises(Refused) as cm:
                self.train(inputs, smoke["smoke"])
        self.assertRule(cm, "VLM-3", "does not match the hash")


class SmokeMeasureTests(unittest.TestCase):
    def test_measures_and_reasons(self):
        calls = [{"reward": r, "response_sha256": h} for r, h in ((0.0, "a"), (0.2, "b"), (0.0, "a"))] * 3
        m = stages.smoke_measures(calls)
        self.assertEqual((m["n_samples"], m["n_distinct_completions"]), (9, 2))
        self.assertEqual(stages.smoke_reasons(m), [])
        self.assertTrue(stages.smoke_reasons(stages.smoke_measures(calls[:3])))      # too few samples

    def test_malformed_line_raises(self):
        for bad in ({"reward": 1.0}, {"reward": "x", "response_sha256": "a"}, "text", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                stages.smoke_measures([bad])


class StageShapeTests(unittest.TestCase):
    def test_names_and_declarations(self):
        names = [c.name for c in (ProfileData, ChooseTaskAndBase, BuildTrainSet, WriteRecipe, Smoke, Train)]
        self.assertEqual(names, ["clevr_profile_data", "clevr_choose_task_and_base", "clevr_build_train_set",
                                 "clevr_write_recipe", "clevr_smoke", "clevr_train"])
        self.assertTrue(Train.trains and not Smoke.trains)
        self.assertEqual(Smoke.needs, frozenset({"gpu"}))


if __name__ == "__main__":
    unittest.main()
