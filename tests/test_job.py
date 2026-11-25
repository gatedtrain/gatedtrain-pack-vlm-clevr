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
from gatedtrain.families import nemo_rl
from gatedtrain.families.nemo_rl import NEMO_RL_PYTHON

from _images import TIERB_IMAGE, add_owner_image, isolate_owner_config
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
        self.owner_config = isolate_owner_config(self, self.tmp)
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
        j = job.train_job(profile=profile, image=kw.pop("image", TIERB_IMAGE), inputs_dir=self.inputs,
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
        self.assertEqual(dict(j.runtime), {"kind": "container", "image": TIERB_IMAGE})
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

    def test_accepted_images_come_from_the_image_lock(self):
        """No ID is written in the pack: the allow-list is gatedtrain's accepted images of the three NeMo RL names."""
        want = []
        for logical in nemo_rl.LOGICAL_IMAGES:
            want += [i for i in nemo_rl.accepted_images(logical) if i not in want]
        self.assertEqual(list(job.accepted_images()), want)
        self.assertIn(TIERB_IMAGE, want)

    def test_derived_image_accepted_only_when_added_by_the_owner(self):
        derived = "sha256:" + "ab" * 32
        with self.assertRaises(Refused) as cm:
            self.job(image=derived)
        self.assertIn("gatedtrain image add", cm.exception.problems[0].fix)
        add_owner_image(self.owner_config, nemo_rl.IMAGE_ROOTLESS, derived)
        j = self.job(image=derived)
        self.assertEqual(dict(j.runtime)["image"], derived)
        self.assertEqual(job_problems(j, self.cfg), [])
        self.assertEqual(job.accepted_images().count(TIERB_IMAGE), 1)

    def test_owner_image_of_another_logical_image_is_accepted_too(self):
        """The owner may add to any logical NeMo RL image; the pack accepts all three names."""
        derived = "sha256:" + "ef" * 32
        add_owner_image(self.owner_config, nemo_rl.IMAGE_UID1000, derived)
        self.assertIn(derived, job.accepted_images())

    def test_untrustworthy_owner_config_is_refused_not_ignored(self):
        add_owner_image(self.owner_config, nemo_rl.IMAGE_ROOTLESS, "sha256:" + "ab" * 32)
        self.owner_config.chmod(0o666)                                   # writable by others: IMG-3
        with self.assertRaises(Refused) as cm:
            self.job()
        self.assertEqual([p.rule for p in cm.exception.problems], ["IMG-3"])

    def test_deprecated_environment_list_still_works_and_warns(self):
        """GATEDTRAIN_PACK_VLM_CLEVR_EXTRA_IMAGES is a deprecated, named workaround (see the README): it still adds
        the IDs, and every use says it is going away."""
        derived = "sha256:" + "cd" * 32
        with mock.patch.dict(os.environ, {job.EXTRA_IMAGES_ENV: f"{derived}, {TIERB_IMAGE}"}):
            with self.assertWarns(FutureWarning) as w:
                j = self.job(image=derived)
        self.assertEqual(dict(j.runtime)["image"], derived)
        self.assertIn("gatedtrain image add", str(w.warning))
        self.assertIn(job.EXTRA_IMAGES_REMOVAL, str(w.warning))
        with mock.patch.dict(os.environ, {job.EXTRA_IMAGES_ENV: f"{derived}, {TIERB_IMAGE}"}), \
                self.assertWarns(FutureWarning):
            self.assertEqual(job.accepted_images().count(TIERB_IMAGE), 1)

    def test_no_warning_without_the_environment_list(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.job()

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
