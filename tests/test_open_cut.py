"""`Train._open_cut` on a cut the container wrote: the container is not trusted, so a hostile `cut.json` must not make
the owner's stage change anything outside the cut's own files."""

import hashlib
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from gatedtrain.errors import Refused

from gatedtrain_pack_vlm_clevr import stages


def mode(p) -> int:
    return stat.S_IMODE(os.lstat(p).st_mode)


class OpenCutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.top = self.tmp / "top"
        self.cut = self.top / "cuts" / "checkpoint-60"
        self.cut.mkdir(parents=True, mode=0o700)
        os.chmod(self.top / "cuts", 0o700)
        self.victim = self.tmp / "victim.json"
        self.victim.write_text("secret")
        os.chmod(self.victim, 0o600)
        self.names, self.files = [], {}

    def add(self, name, data=b"x", write=True):
        if write:
            p = self.cut / name
            p.write_bytes(data)
            os.chmod(p, 0o600)
        self.names.append(name)
        self.files[name] = "sha256:" + hashlib.sha256(data).hexdigest()

    def finish(self, step=60):
        (self.cut / "cut.json").write_text(json.dumps({"deployable": self.names, "files": self.files,
                                                       "global_step": step}))

    def refused(self, *words):
        with self.assertRaises(Refused) as cm:
            stages.Train._open_cut(self.top, 60)
        for w in words:
            self.assertIn(w, str(cm.exception))
        return cm

    def test_good_cut_is_opened(self):
        self.add("config.json")
        self.add("model.safetensors")
        self.finish()
        self.assertEqual(sorted(stages.Train._open_cut(self.top, 60)), ["config.json", "model.safetensors"])
        self.assertTrue(mode(self.cut / "config.json") & 0o004)
        self.assertTrue(mode(self.cut) & 0o001)

    def test_absolute_and_parent_names_refused_and_untouched(self):
        for bad in (str(self.victim), "../../../victim.json", "sub/x.json"):
            with self.subTest(bad=bad):
                self.names, self.files = [], {}
                self.add("config.json")
                self.add(bad, data=b"secret", write=False)
                self.finish()
                self.refused("not a plain file name")
                self.assertEqual(mode(self.victim), 0o600)

    def test_disallowed_suffix_refused(self):
        self.add("config.json")
        self.add("evil.pt")
        self.finish()
        self.refused("suffix")

    def test_hash_mismatch_refused_before_any_chmod(self):
        self.add("config.json")
        (self.cut / "config.json").write_bytes(b"changed")
        os.chmod(self.cut / "config.json", 0o600)
        self.finish()
        self.refused("does not match the hash")
        self.assertEqual(mode(self.cut / "config.json"), 0o600)

    def test_symlinked_file_refused_and_target_untouched(self):
        self.add("config.json", data=b"secret", write=False)
        os.symlink(self.victim, self.cut / "config.json")
        self.finish()
        self.refused("not a regular file")
        self.assertEqual(mode(self.victim), 0o600)

    def test_symlinked_directory_refused_and_target_untouched(self):
        shutil.rmtree(self.top / "cuts")
        target = self.tmp / "ownerdir"
        target.mkdir(mode=0o700)
        (target / "checkpoint-60").mkdir(mode=0o700)
        os.symlink(target, self.top / "cuts")
        (target / "checkpoint-60" / "cut.json").write_text(json.dumps(
            {"deployable": [], "files": {}, "global_step": 60}))
        self.refused("not a directory")
        self.assertEqual(mode(target), 0o700)

    def test_cut_json_symlink_refused(self):
        self.add("config.json")
        real = self.tmp / "other-cut.json"
        real.write_text(json.dumps({"deployable": self.names, "files": self.files, "global_step": 60}))
        os.symlink(real, self.cut / "cut.json")
        self.refused("cut.json")

    def test_wrong_step_and_empty_deployable_refused(self):
        self.add("config.json")
        self.finish(step=59)
        self.refused("cut.json")
        self.names, self.files = [], {}
        self.finish()
        self.refused("cut.json")


if __name__ == "__main__":
    unittest.main()
