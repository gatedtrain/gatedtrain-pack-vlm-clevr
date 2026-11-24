"""The keeper's eval requirements (`eval-requirements.txt`, the input of `gatedtrain setup host --eval-requirements`),
the lock it was exported from, and the README that names the pack's entry points and params."""

import re
import tomllib
import unittest
from importlib import metadata
from pathlib import Path

import gatedtrain_pack_vlm_clevr
import yaml

PACKAGE = Path(gatedtrain_pack_vlm_clevr.__file__).resolve().parent
REPO = PACKAGE.parent.parent
REQUIREMENTS = PACKAGE / "eval-requirements.txt"
EVAL_DIR = PACKAGE / "eval"
PINS = {"vllm": "0.31.0", "transformers": "5.17.0", "torch": "2.13.0"}
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")
HASH = re.compile(r"--hash=sha256:[0-9a-f]{64}")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_fits(filename: str) -> bool:
    """A wheel CPython 3.12 on Linux x86_64 can install: python tag `py<N>` or `cp312` (or `cp3<M>` with `abi3`, M at
    most 12), ABI `none`, `abi3` or `cp312`, platform `any` or a manylinux (or linux) x86_64 tag."""
    parts = filename[:-len(".whl")].split("-")
    if len(parts) < 5:
        return False
    pys, abi, plats = parts[-3].split("."), parts[-2], parts[-1].split(".")
    py_ok = any(re.fullmatch(r"py\d+", t) or t == "cp312" for t in pys) or (
        abi == "abi3" and any(re.fullmatch(r"cp3(\d+)", t) and int(t[3:]) <= 12 for t in pys))
    plat_ok = plats == ["any"] or any(re.fullmatch(r"(many)?linux[0-9_]*_x86_64", t) for t in plats)
    return py_ok and abi in ("none", "abi3", "cp312") and plat_ok


def logical_lines(text: str) -> list[str]:
    """The requirement entries of a requirements file: comment lines dropped, continuations joined."""
    out, cur = [], ""
    for raw in text.splitlines():
        if raw.lstrip().startswith("#") or not raw.strip():
            continue
        cur += " " + raw.strip().rstrip("\\").strip()
        if not raw.rstrip().endswith("\\"):
            out.append(cur.strip())
            cur = ""
    return out


class EvalRequirementsTest(unittest.TestCase):
    def test_every_line_pinned_with_hash(self):
        entries = logical_lines(REQUIREMENTS.read_text())
        self.assertGreater(len(entries), 20)
        names = {}
        for entry in entries:
            self.assertFalse(entry.startswith("-"), entry)  # no index, no editable, no include
            m = PIN.match(entry)
            self.assertIsNotNone(m, entry)
            self.assertTrue(HASH.search(entry), f"{m.group(1)} has no sha256 hash")
            names[m.group(1).lower()] = m.group(2)
        for name, version in PINS.items():
            self.assertEqual(names.get(name), version)

    def test_no_sdist_only(self):
        lock = tomllib.loads((EVAL_DIR / "uv.lock").read_text())
        wanted = {norm(m.group(1)) for m in map(PIN.match, logical_lines(REQUIREMENTS.read_text())) if m}
        locked = {p["name"]: p for p in lock["package"]}
        self.assertLessEqual(wanted, set(locked))
        for name in wanted:
            wheels = locked[name].get("wheels") or []
            self.assertTrue(wheels, f"{name} has no wheel (sdist only)")
            ok = [w for w in wheels if wheel_fits(w["url"].rsplit("/", 1)[-1])]
            self.assertTrue(ok, f"{name} has no cp312/py3 wheel usable on manylinux x86_64")

    def test_wheel_fits(self):
        for name, fits in (("a-1-py3-none-any.whl", True), ("a-1-py2.py3-none-any.whl", True),
                           ("a-1-py37-none-any.whl", True), ("a-1-cp312-cp312-manylinux_2_28_x86_64.whl", True),
                           ("a-1-cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl", True),
                           ("a-1-cp312-cp312-manylinux_2_28_aarch64.whl", False),
                           ("a-1-cp312-cp312-macosx_11_0_arm64.whl", False),
                           ("a-1-cp311-cp311-manylinux_2_28_x86_64.whl", False),
                           ("a-1-py3-none-manylinux_2_27_x86_64.whl", True),
                           ("a-1-py3-none-musllinux_1_2_x86_64.whl", False),
                           ("a-1-cp313-abi3-manylinux_2_28_x86_64.whl", False),
                           ("a-1-cp312-cp312-win_amd64.whl", False)):
            with self.subTest(name=name):
                self.assertIs(wheel_fits(name), fits)

    def test_requirements_agree_with_the_lock(self):
        """Every `name==version` of eval-requirements.txt is the lock's version, and its hashes are the lock's wheel
        and sdist hashes of that version (the export of eval/uv.lock, not edited since)."""
        lock = tomllib.loads((EVAL_DIR / "uv.lock").read_text())
        locked = {p["name"]: p for p in lock["package"]}
        entries = logical_lines(REQUIREMENTS.read_text())
        for entry in entries:
            m = PIN.match(entry)
            name = norm(m.group(1))
            with self.subTest(name=name):
                pkg = locked.get(name)
                self.assertIsNotNone(pkg, f"{name} is not in eval/uv.lock")
                self.assertEqual(pkg["version"], m.group(2))
                files = list(pkg.get("wheels") or []) + ([pkg["sdist"]] if "sdist" in pkg else [])
                self.assertEqual(set(re.findall(r"--hash=(sha256:[0-9a-f]{64})", entry)), {f["hash"] for f in files})

    def test_eval_project_pins(self):
        doc = tomllib.loads((EVAL_DIR / "pyproject.toml").read_text())
        self.assertEqual(sorted(doc["project"]["dependencies"]), sorted(f"{k}=={v}" for k, v in PINS.items()))
        self.assertFalse(doc["tool"]["uv"]["package"])

    def test_requirements_and_lock_ship_in_the_wheel(self):
        data = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]
        shipped = data["gatedtrain_pack_vlm_clevr"]
        self.assertIn("eval-requirements.txt", shipped)
        self.assertIn("eval/*", shipped)

    def test_pack_dependencies_unchanged(self):
        self.assertEqual(tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"],
                         ["gatedtrain>=0.5,<0.6"])


class ReadmeTest(unittest.TestCase):
    def setUp(self):
        self.text = (REPO / "README.md").read_text()

    def test_readme_names_entry_points_and_params(self):
        project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
        names = [*project["scripts"], *project["entry-points"]["gatedtrain.stages"]]
        self.assertEqual(len(names), 7)
        for name in names:
            with self.subTest(name=name):
                self.assertIn(name, self.text)
        pipeline = yaml.safe_load((PACKAGE / "pipeline.yaml").read_text())
        for ref in pipeline["stages"]:
            for param in ref.get("params", {}):
                with self.subTest(param=param):
                    self.assertIn(param, self.text)
        for env in ("GATEDTRAIN_PACK_VLM_CLEVR_RUNTIME", "GATEDTRAIN_PACK_VLM_CLEVR_EVAL_PYTHON", "GATEDTRAIN_PACK_VLM_CLEVR_EXTRA_IMAGES",
                    "GATEDTRAIN_RECIPE_VLM_CLEVR_DIR", "GATEDTRAIN_SRC"):
            self.assertIn(env, self.text)

    def test_readme_states_the_pinned_facts(self):
        flat = " ".join(self.text.split())
        for needle in ("2fc06364715b967f1860aea9cf38778875588b17", "temperature of 0", "64 MiB", "NEMO_RL_IMAGES"):
            self.assertIn(needle, flat)

    def test_no_host_paths(self):
        for path in (REPO / "README.md", REPO / "THIRD_PARTY_NOTICES.md"):
            self.assertNotIn("/home/", path.read_text())


if __name__ == "__main__":
    unittest.main()
