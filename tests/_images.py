"""Image fixtures for the pack's tests: the maintainer IDs come from gatedtrain's image lock (never written here), and
the owner's image config is a file in a temporary directory, so no test reads or depends on the real account's."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from gatedtrain import imagelock
from gatedtrain.families import nemo_rl

TIERB_IMAGE = imagelock.maintainer_ids(nemo_rl.IMAGE_TIERB)[0]
UID1000_IMAGE = imagelock.maintainer_ids(nemo_rl.IMAGE_UID1000)[0]


def isolate_owner_config(case: unittest.TestCase, root: Path) -> Path:
    """Point `XDG_CONFIG_HOME` at `root/xdg` for this test, so `gatedtrain.imagelock.user_config_path()` is a file of
    the test. Returns the config file's path (not created)."""
    home = Path(root) / "xdg"
    patcher = mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(home)})
    patcher.start()
    case.addCleanup(patcher.stop)
    return home / "gatedtrain" / "images.json"


def add_owner_image(path: Path, logical: str, image_id: str) -> None:
    """What `gatedtrain image add <logical> --id <image_id> ...` records, written through gatedtrain's own writer."""
    doc = {"schema": imagelock.CONFIG_SCHEMA, "schema_version": imagelock.SCHEMA_VERSION,
           "images": {logical: {"accepted": [{
               "id": image_id, "origin": imagelock.USER,
               "recipe": {"name": "test-recipe", "source": None, "commit": "0" * 40, "containerfile_sha256": None,
                          "build": None},
               "base_id": None, "built_utc": "2026-01-01", "added_by": "pack tests", "note": None}]}}}
    imagelock.write_user_config(doc, path)
