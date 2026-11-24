"""The pack's scenario (`scenario.SCENARIO`), its pipeline file and its entry points, as gatedtrain loads them."""

import hashlib
import importlib
import os
import unittest
from importlib import metadata
from pathlib import Path
from unittest import mock

from gatedtrain.ainative import scorers, stage as stage_mod, trust_stages
from gatedtrain.ainative.batch import BatchGenerator
from gatedtrain.ainative.pipeline import load_pipeline, pipeline_problems
from gatedtrain.errors import Refused

import gatedtrain_pack_vlm_clevr
from gatedtrain_pack_vlm_clevr import adapter, answer, scenario, stages

PACKAGE = Path(gatedtrain_pack_vlm_clevr.__file__).resolve().parent
STAGE_CLASSES = (stages.ProfileData, stages.ChooseTaskAndBase, stages.BuildTrainSet, stages.WriteRecipe,
                 stages.Smoke, stages.Train)
STEPS = {"profile_data": "clevr_profile_data", "choose_task_and_base": "clevr_choose_task_and_base",
         "build_train_set": "clevr_build_train_set", "write_recipe": "clevr_write_recipe", "smoke": "clevr_smoke",
         "train": "clevr_train"}


class _EP:
    def __init__(self, name, obj, value):
        self.name, self.value, self._obj, self.dist = name, value, obj, None

    def load(self):
        return self._obj


def scenario_eps(obj=None):
    return [_EP("vlm-clevr", scenario.SCENARIO if obj is None else obj,
                "gatedtrain_pack_vlm_clevr.scenario:SCENARIO")]


def stage_eps():
    return [_EP(c.name, c, f"gatedtrain_pack_vlm_clevr.stages:{c.__name__}") for c in STAGE_CLASSES]


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        trust_stages._reset_scenario_snapshot()
        self.addCleanup(trust_stages._reset_scenario_snapshot)
        p = mock.patch.object(trust_stages, "_scenario_entry_points", scenario_eps)
        p.start()
        self.addCleanup(p.stop)

    def test_scenario_loads(self):
        self.assertEqual(scenario.SCENARIO.name, "vlm-clevr")
        self.assertEqual(dict(scenario.SCENARIO.generators), {})
        self.assertEqual(dict(trust_stages.scenario_generators()), {})
        self.assertEqual(dict(trust_stages.scenario_batch_generators()), {"vlm_clevr_vllm": scenario.BATCH})
        self.assertEqual(trust_stages.scenario_pack_of("vlm_clevr_vllm"), "vlm-clevr")
        spec = scorers.resolve("vqa_answer")
        self.assertEqual((spec.name, spec.evaluator_id, spec.direction, spec.planning_sd, spec.score_range),
                         ("vqa_answer", "vqa_answer@0.1", "higher_is_better", 0.5, (0.0, 1.0)))
        self.assertLessEqual(spec.planning_sd, scorers.PLANNING_SD_BOUND)
        self.assertIs(spec.fn, answer.score)
        self.assertIs(spec.label_ok, answer.label_ok)
        decl = scenario.SCENARIO.scorers["vqa_answer"]
        self.assertEqual(decl["code_sha256"],
                         "sha256:" + hashlib.sha256((PACKAGE / "answer.py").read_bytes()).hexdigest())
        self.assertEqual((decl["what"], decl["unit"], list(decl["range"])),
                         ("canonical answer (number or yes/no) equals the label", "exact match", [0, 1]))
        self.assertEqual(adapter.EVALUATOR_ROWS[adapter.EVALUATOR_ID]["scorer"]["code_sha256"], decl["code_sha256"])

    def test_changed_scorer_code_refused(self):
        changed = scenario.Scenario(scenario.NAME, {}, scenario.SCENARIO.pipeline,
                                    {"vqa_answer": {**scenario.SCORERS["vqa_answer"],
                                                      "code_sha256": "sha256:" + "0" * 64}},
                                    {"vlm_clevr_vllm": scenario.BATCH})
        with mock.patch.object(trust_stages, "_scenario_entry_points", lambda: scenario_eps(changed)):
            trust_stages._reset_scenario_snapshot()
            with self.assertRaises(Refused) as cm:
                scorers.resolve("vqa_answer")
        self.assertEqual({p.rule for p in cm.exception.problems}, {"AN-17"})

    def test_batch_generator(self):
        gen = scenario.BATCH
        self.assertIsInstance(gen, BatchGenerator)
        self.assertEqual(gen.script, PACKAGE / "generate_vllm.py")
        self.assertTrue(gen.script.is_file())
        self.assertEqual((gen.gpu, gen.time_limit_s, gen.eval_trees), (True, 1200, True))

    def test_eval_python_default_and_override(self):
        self.assertEqual(scenario.eval_python({}), "/opt/gatedtrain-keeper/eval-venv/bin/python")
        self.assertEqual(scenario.eval_python({scenario.EVAL_PYTHON_ENV: ""}),
                         "/opt/gatedtrain-keeper/eval-venv/bin/python")
        self.assertEqual(scenario.eval_python({scenario.EVAL_PYTHON_ENV: "/srv/eval/venv/bin/python"}),
                         "/srv/eval/venv/bin/python")
        with self.assertRaises(ValueError):
            scenario.eval_python({scenario.EVAL_PYTHON_ENV: "venv/bin/python"})

    def test_eval_python_read_at_import(self):
        env = {k: v for k, v in os.environ.items() if k != scenario.EVAL_PYTHON_ENV}

        def reload(environ):
            with mock.patch.dict(os.environ, environ, clear=True):
                importlib.reload(scenario)
        self.addCleanup(reload, dict(os.environ))
        reload({**env, scenario.EVAL_PYTHON_ENV: "/srv/eval/venv/bin/python"})
        self.assertEqual(scenario.BATCH.interpreter, "/srv/eval/venv/bin/python")
        self.assertEqual(scenario.SCENARIO.batch_generators["vlm_clevr_vllm"].interpreter,
                         "/srv/eval/venv/bin/python")
        with self.assertRaises(ValueError):
            reload({**env, scenario.EVAL_PYTHON_ENV: "relative/python"})
        reload(env)
        self.assertEqual(scenario.BATCH.interpreter, "/opt/gatedtrain-keeper/eval-venv/bin/python")

    def test_pipeline(self):
        path = scenario.SCENARIO.pipeline
        self.assertEqual(path, PACKAGE / "pipeline.yaml")
        spec = load_pipeline(path)
        with mock.patch.object(stage_mod, "_plugin_entry_points", stage_eps):
            registry = stage_mod.stage_registry()
        self.assertEqual(pipeline_problems(spec, registry), [])
        refs = {r.name: r for r in spec.stages}
        self.assertEqual([r.name for r in spec.stages],
                         ["intake_seal", "profile_data", "choose_task_and_base", "preregister", "build_train_set",
                          "write_recipe", "smoke", "train", "evaluate_sealed", "gate", "report"])
        self.assertEqual({k: refs[k].stage for k in STEPS}, STEPS)
        for name in ("smoke", "train"):
            self.assertEqual(refs[name].inputs["base"], "choose_task_and_base.base")
            self.assertEqual(refs[name].params, {"image": None})
        self.assertEqual(refs["train"].inputs["smoke"], "smoke.smoke")
        self.assertEqual(refs["choose_task_and_base"].params, {"base_model_dir": None})
        self.assertEqual(refs["evaluate_sealed"].inputs["baseline_model"], "choose_task_and_base.base")
        self.assertEqual(refs["evaluate_sealed"].inputs["candidate_model"], "train.model")
        text = path.read_text()
        for key in ("image", "base_model_dir", "--pipeline"):
            self.assertIn(key, text)


class EntryPointTests(unittest.TestCase):
    """The installed distribution's entry points (after `pip install -e .`)."""

    def eps(self, group):
        return {ep.name: ep for ep in metadata.entry_points(group=group)
                if ep.dist is not None and ep.dist.metadata["Name"] == "gatedtrain-pack-vlm-clevr"}

    def test_entry_points(self):
        self.assertEqual({n: ep.value for n, ep in self.eps("gatedtrain.scenarios").items()},
                         {"vlm-clevr": "gatedtrain_pack_vlm_clevr.scenario:SCENARIO"})
        self.assertIs(self.eps("gatedtrain.scenarios")["vlm-clevr"].load(), scenario.SCENARIO)
        st = {n: ep.load() for n, ep in self.eps("gatedtrain.stages").items()}
        self.assertEqual(st, {c.name: c for c in STAGE_CLASSES})
        self.assertEqual(set(st), set(STEPS.values()))
        ad = self.eps("gatedtrain.adapters")
        self.assertEqual(set(ad), {"vlm-clevr-grpo"})
        self.assertIs(ad["vlm-clevr-grpo"].load(), adapter.PLUGIN)


if __name__ == "__main__":
    unittest.main()
