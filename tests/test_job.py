import dataclasses
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gatedtrain import spec
from gatedtrain.ainative.backend import job_problems
from gatedtrain.ainative.runtime import Mount
from gatedtrain.errors import Refused
from gatedtrain.ext import conformance, launch, registry
from gatedtrain.families.nemo_rl import NEMO_RL_IMAGE, NEMO_RL_PYTHON

from gatedtrain_pack_vlm_clevr import adapter, job

FAKE_GID = 2**31 - 3


class _Layout(unittest.TestCase):
    def setUp(self):
        for table in (registry.ADAPTERS, registry.TRAINER_ROWS, registry.EVALUATOR_ROWS, spec.TRAINER_BACKENDS,
                      spec.EVALUATOR_BACKENDS):
            p = mock.patch.dict(table)
            p.start()
            self.addCleanup(p.stop)
        adapter.install()
        self.tmp = Path(os.path.realpath(tempfile.mkdtemp(prefix="tmp-job-", dir=Path(__file__).resolve().parent)))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.work, self.out_root = self.tmp / "work", self.tmp / "out"
        base = self.out_root / "gatedtrain-0123abcd"
        self.inputs, self.images, self.base, self.output = (base / "train-inputs", base / "images",
                                                            base / "base-model", base / "train")
        for d in (self.work, self.inputs, self.images, self.base, self.output):
            d.mkdir(parents=True, exist_ok=True)
        (self.inputs / "train.jsonl").write_text("{}\n")
        (self.base / "config.json").write_text("{}")
        self.cfg = launch.HostConfig(host_work=str(self.work), output_root=str(self.out_root),
                                     primary_gid=lambda uid: FAKE_GID, rootless=True)

    def job(self, profile="smoke", seed=7100, **kw):
        j = job.train_job(profile=profile, image=kw.pop("image", NEMO_RL_IMAGE), inputs_dir=self.inputs,
                          images_dir=self.images, base_dir=self.base, seed=seed, **kw)
        return dataclasses.replace(j, output_dir=str(self.output))


class JobTests(_Layout):
    def test_conforms(self):
        self.assertEqual(conformance.adapter_problems(adapter.NAME), [])

    def test_job_problems_clean(self):
        self.assertEqual(job_problems(self.job(), self.cfg), [])
        self.assertEqual(job_problems(self.job("full"), self.cfg), [])

    def test_fields(self):
        j = self.job()
        runtime = adapter.RUNTIME
        self.assertEqual(j.adapter, adapter.NAME)
        self.assertEqual(dict(j.runtime), {"kind": "container", "image": NEMO_RL_IMAGE})
        self.assertEqual(j.argv, (NEMO_RL_PYTHON, f"{runtime}/run_vlm_nemo.py", "--profile", "smoke", "--out", "/out",
                                  "--seed", "7100", "--base", "/inputs/base-model", "--memory", "a",
                                  "--train-file", "/inputs/train.jsonl"))
        self.assertEqual(j.mounts, (
            Mount(runtime, runtime, True), Mount(str(self.images), str(self.images), True),
            Mount(str(self.inputs / "train.jsonl"), "/inputs/train.jsonl", True),
            Mount(str(self.base), "/inputs/base-model", True)))
        self.assertEqual((j.network, j.gpu, j.tmpfs, j.training_seed), ("none", True, ("/tmp",), 7100))
        self.assertEqual(self.job("full").gpu_hours_est, 1.0)

    def test_foreign_env_name_refused(self):
        j = self.job()
        bad = dataclasses.replace(j, env={**j.env, "HF_TOKEN": "x"})
        self.assertTrue(job_problems(bad, self.cfg))

    def test_unpinned_image_refused(self):
        with self.assertRaises(Refused) as cm:
            self.job(image="docker.io/library/python:3.12")
        self.assertEqual([p.rule for p in cm.exception.problems], ["VLM-4"])

    def test_derived_image_accepted_only_when_listed(self):
        derived = "sha256:" + "ab" * 32
        with self.assertRaises(Refused):
            self.job(image=derived)
        with mock.patch.dict(os.environ, {job.EXTRA_IMAGES_ENV: f"{derived}, {NEMO_RL_IMAGE}"}):
            j = self.job(image=derived)
            self.assertEqual(dict(j.runtime)["image"], derived)
            self.assertEqual(job_problems(j, self.cfg), [])
            self.assertEqual(job.accepted_images().count(NEMO_RL_IMAGE), 1)

    def test_malformed_extra_image_entry_refused_not_dropped(self):
        with mock.patch.dict(os.environ, {job.EXTRA_IMAGES_ENV: "nemo-rl:latest"}):
            with self.assertRaises(Refused) as cm:
                self.job()
        self.assertIn(job.EXTRA_IMAGES_ENV, str(cm.exception))

    def test_other_runtime_refused(self):
        with self.assertRaises(Refused) as cm:
            self.job(runtime_dir=self.tmp)
        self.assertIn(adapter.RUNTIME_ENV, str(cm.exception))

    def test_bad_profile_and_rung(self):
        with self.assertRaises(ValueError):
            self.job("nope")
        with self.assertRaises(ValueError):
            self.job(memory_rung="z")


if __name__ == "__main__":
    unittest.main()
