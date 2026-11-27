# gatedtrain-pack-vlm-clevr

A scenario pack for [gatedtrain](https://github.com/gatedtrain/gatedtrain) (`gatedtrain>=0.7.2,<0.8`) for
vision-language models: it teaches a small VLM to answer counting and yes/no questions about pictures, and lets
gatedtrain decide, by a rule fixed before training, whether the trained model really is better.

It trains Qwen3.5-0.8B with NeMo RL's VLM GRPO (the recipe `gatedtrain-recipe-nemo-rl-vlm-clevr`) on your
question-image-answer rows on your GPU. Before training, gatedtrain's keeper seals part of your rows away, images
included; after training, the keeper answers the sealed questions with the base and the trained model and scores each
answer itself. The gate passes only when the trained model beats the base by the registered rule.

It is a prototype: the models it trains are not promotable.

## Changes in 0.3.0

- Needs gatedtrain 0.7 (`gatedtrain>=0.7,<0.8`). Its CPU tests (117) pass unchanged against 0.6.0 and the 0.7.0
  candidate; the changes below are the pack adopting what 0.7 offers.
- **Trainer images come from gatedtrain's image lock.** The pack no longer imports the image constants
  (`NEMO_RL_IMAGES`); `job.accepted_images()` is gatedtrain's `accepted_images(logical)` for `nemo-rl/tierb`,
  `nemo-rl/tierb-uid1000` and `nemo-rl/tierb-rootless`: the maintainer lock, plus the images you recorded with
  `gatedtrain image add` (see "What you need"). A refusal names `gatedtrain image add`; an untrustworthy image config is
  refused (IMG-3), not ignored.
- **`GATEDTRAIN_PACK_VLM_CLEVR_EXTRA_IMAGES` is deprecated.** It is a workaround kept for this release only, because the
  0.2.0 README told rootless hosts to use it. It still adds the IDs it lists and warns (`FutureWarning`) at every use.
  It is removed in 0.4.0; replace it with `gatedtrain image add`. Owner: the pack maintainer.
- **The Containerfile moved.** `containers/nemo-rl-rootless/` is no longer in this repository: it is gatedtrain's
  `containers/nemo-rl-rootless` (the same layer, checked file by file against this pack's copy; the build takes the base
  image as `--build-arg BASE`). The gatedtrain source checkout holds it; the wheel does not.
- Keeper modes: see "Keeper modes" below.
- Known limits of this release: no GPU end to end run was made with gatedtrain 0.7 (the CPU tests use fake containers);
  the image you build is host specific, so every rootless host runs `image add` once; a derived image is user-added in
  the report, so `gatedtrain verify --require-maintainer-images` exits 1 for a rootless-host run (the warning in the
  report names the image and its recipe); the keeper's eval venv still comes from `gatedtrain setup host`, a script for
  root, in both keeper modes (a user-owned eval tree for a same-user keeper has not been tried); a same-user keeper
  needs `--inference-image` too, because this pack's generator is a batch generator.

## Keeper modes

gatedtrain 0.7 has two keeper modes and every result states the one that produced it. **Same-user** (the default of
`gatedtrain setup keeper`, no sudo): the keeper runs under your own OS user and follows the procedure, but anything
running as you, such as an agent with a shell, can read the sealed rows, the labels and the keeper's key; the report
says the data was not protected from your account. **Separate-user** (`gatedtrain setup keeper --mode separate-user`,
one sudo): the keeper runs under its own OS user and the sealed rows are out of your account's reach. This pack works in
both; in same-user mode the pack needs to be installed in the environment you run `gatedtrain` from (the keeper runs
from it), in separate-user mode also in the keeper's own environment. Quote the mode line of the report when you quote a
result, and pass `--require-keeper-mode separate-user` to `project create` to refuse a weaker keeper. The run below was
made before gatedtrain 0.7 existed and does not state a keeper mode. See gatedtrain's
[Quickstart without sudo](https://github.com/gatedtrain/gatedtrain/blob/main/docs/sudo-free-quickstart.md) and
[AI-native projects](https://github.com/gatedtrain/gatedtrain/blob/main/docs/ai-native.md#keeper-modes).

## A tested run

One run on one 24 GB GPU with greedy decoding (the default), 2,086 CLEVR rows (one row per distinct image; 1,687 for
training, 399 sealed for the test), the 60-step recipe and the derived trainer image below:

| | Score on the sealed questions |
|---|---|
| Base (Qwen3.5-0.8B) | 0.4135 |
| Trained (60 GRPO steps) | 0.6241 |
| Gate | **PASS**: +0.2105, 97.68 % interval [0.1328, 0.2882], power 0.75 |

About 0.9 GPU hours in all (smoke 7 minutes, training 32 minutes, keeper evaluation 17 minutes: greedy decoding runs
one request at a time, see "Decoding"); `gatedtrain verify --keeper` accepted the report. The gate's test level was
0.0116 because the same sealed set had been used once before (an earlier attempt that the keeper refused for
non-deterministic decoding, below, and that spent alpha). Greedy decoding with one seed is the decoding of the recipe's
published gate (0.404 -> 0.730), but the data and the split differ, so the numbers are close in kind, not a
replication. The earlier run of version 0.1.0 (temperature 0.7 over four seeds, other data: 0.425 -> 0.567) is not
comparable with either. Your numbers will differ with other data, seeds and hardware. The CPU tests drive the whole
pipeline with fake containers.

## Decoding

By default the pack registers **greedy decoding**: `decoding: greedy`, temperature 0, top_p 1.0, at most 512 new tokens,
one evaluation seed (11), one sample per item. It needs gatedtrain 0.6 or later (this release needs 0.7): the keeper checks greedy decoding by
generating every sealed item once more under a seed the registration does not name and refuses the attempt if the
predictions differ (AN-26; the refused attempt costs its alpha).

**Determinism requirement.** vLLM's default batching is not deterministic for Qwen3.5-0.8B with a 512-token chain of
thought: on 399 items, 54 to 56 % of the base model's answers and 30 % of the trained model's differed between two
greedy passes (a repeat with the same seed in another process differed as much), and vLLM's batch-invariant mode does
not support Qwen3.5's attention. The pack's generator therefore runs greedy decoding **one request at a time**
(`max_num_seqs=1`) with prefix caching off, which gave 0 of 399 differing items for both models in the same probe; it
costs about 7 minutes per arm for 399 items. A change to the generator, the vLLM version or the GPU needs the same probe
again (two passes with different seeds on items outside the sealed set) before the keeper is used.

For the earlier mode set `decoding: sampled` in the `params` of `choose_task_and_base` (temperature 0.7, top_p 1.0,
four seeds 11-14); the keeper needs no determinism check for it.

## How it works

Same shape as `gatedtrain-pack-text-gsm8k`: the owner prepares data and trains; the keeper, a service of gatedtrain,
holds the sealed rows and runs the evaluation. Whether the keeper's OS user keeps the sealed rows from your account
depends on the keeper mode (see "Keeper modes"). Stages of the pipeline (`pipeline.yaml`):

`intake_seal` -> `profile_data` -> `choose_task_and_base` -> `preregister` -> `build_train_set` -> `write_recipe` ->
`smoke` (5 GRPO steps, GPU) -> `train` (60 GRPO steps, GPU) -> `evaluate_sealed` -> `gate` -> `report`.

Pack stages: `clevr_profile_data`, `clevr_choose_task_and_base`, `clevr_build_train_set`, `clevr_write_recipe`,
`clevr_smoke`, `clevr_train`. The scorer `vqa_answer` takes the last `<answer>` block (else the last integer, number
word up to twenty, or yes/no of the response) and compares it with the label.

## Data

Each row is `{"input": {"question": "...", "image": "data:image/jpeg;base64,..."}, "label": "3"}`. The image travels
inside the row because a sealed image must never lie in a file the owner or an agent can read. gatedtrain's data file is
limited to 64 MiB, so images should be small: a CLEVR PNG is about 190 KB (a few hundred rows), the same image as JPEG
(quality 90) a fraction of that (a few thousand rows). Convert CLEVR-style items (a JSONL of `{question, answer,
image}` plus the image directory) with

```
pip install "gatedtrain-pack-vlm-clevr[convert]"
gatedtrain-pack-vlm-clevr-data --src items.jsonl --images <image dir> --dst data.jsonl [--jpeg-quality 90 | --keep-images]
```

Images over 400,000 pixels (about 400 image tokens; the recipe's sequence limit is 1,024 tokens) are scaled down,
keeping their aspect ratio; the pack's stages and the generator refuse larger ones. Training needs at least 960 rows
(60 steps of 16 prompts, one pass over the rows) and the sealed part comes on top. Answers are normalised (`"03"` to `"3"`, `"Yes"` to `"yes"`, number words to digits). A refused row is named by line
number; nothing is written until every row passes, and a file over the limit is refused with the number of rows that
fit.

## What you need (host steps, run by you)

- Linux x86_64, one NVIDIA GPU with about 24 GB, rootless Docker with CDI (`gatedtrain setup check` lists what the host
  must provide; see gatedtrain's [Quickstart without sudo](https://github.com/gatedtrain/gatedtrain/blob/main/docs/sudo-free-quickstart.md)),
  a running keeper (`gatedtrain setup keeper`; for a batch generator such as this pack's, with `--inference-image`).
- A NeMo RL trainer image held by your rootless Docker daemon, by an ID gatedtrain accepts: `gatedtrain image list`
  shows them (the logical names `nemo-rl/tierb`, `nemo-rl/tierb-uid1000`, `nemo-rl/tierb-rootless`) and
  `gatedtrain image check <logical> --ref <ref>` tests one. The Tier B image is built from NeMo RL's Dockerfile
  (gatedtrain's `containers/nemo-rl-tierb`); a rebuilt image has another ID, so a build of yours is accepted only after
  you record it.
- On a rootless Docker host the Tier B image cannot run as is: NeMo RL writes lock files beside vLLM files that belong to
  uid 65532 in it, and the container's root (the owner, no capabilities) may not. Build gatedtrain's
  `containers/nemo-rl-rootless` (the same files, owned by root; no network; `--build-arg BASE=<the Tier B image>`) on
  your daemon, then record it once: `gatedtrain image add nemo-rl/tierb-rootless --id sha256:<id of your build> --recipe
  nemo-rl-rootless --recipe-commit <gatedtrain commit of the Containerfile>`, and use that ID as the `image` of `smoke`
  and `train`. The job's flags are unchanged (no capabilities, no network, user 0:0). The runtime options also leave out
  `--ulimit memlock=-1`, which a rootless daemon refuses. gatedtrain's [NeMo RL on a rootless
  host](https://github.com/gatedtrain/gatedtrain/blob/main/docs/rootless-nemo-rl.md) page has the lessons.
  `GATEDTRAIN_PACK_VLM_CLEVR_EXTRA_IMAGES` (comma separated IDs) still works in this release and warns; it is deprecated
  and removed in 0.4.0 (see "Changes in 0.3.0").
- The base model directory: `Qwen/Qwen3.5-0.8B` at commit `2fc06364715b967f1860aea9cf38778875588b17`, with its
  processor files (`preprocessor_config.json`, `video_preprocessor_config.json`) and `chat_template.jinja`.
- The keeper's eval venv, from `<this package>/eval-requirements.txt` (CPython 3.12, vLLM 0.31.0). A same-user keeper
  (the default) needs no sudo for it: `gatedtrain setup eval-tree --requirements <this package>/eval-requirements.txt
  --python-version 3.12 > gatedtrain-eval-tree.sh`, read and run the script as yourself (it needs `uv`; it changes only
  files under its root, `gatedtrain-keeper-eval` beside the keeper's state directory), then start the keeper with the
  `--eval-tree` and `--eval-python` options it prints and with `--inference-image`; the tree is yours, so it is not
  sealed from your account, the keeper checks it at start only (restart it after a change), and a separate-user keeper
  refuses it. A separate-user keeper needs the root-owned tree of `gatedtrain setup host --eval-requirements <this
  package>/eval-requirements.txt` (one sudo). The packages and versions as the text pack's lock (whether vLLM 0.31.0 loads Qwen3.5-0.8B with images is for the first GPU probe to show; none is recorded yet). Start the keeper with `--eval-python PATH` when the venv is
  not the one `gatedtrain setup host` builds (the value is part of the keeper's signed environment record); the variable
  `GATEDTRAIN_PACK_VLM_CLEVR_EVAL_PYTHON` still works as a deprecated workaround, prints a notice, and is removed in
  pack 0.4.0.
- Another runtime directory than the package's `runtime/` (it must hold the same scripts; the adapter reads it at import)
  is the `runtime_dir` key of the `[packs.vlm-clevr]` table in your host file. The variable
  `GATEDTRAIN_PACK_VLM_CLEVR_RUNTIME` still works as a deprecated workaround and prints a notice; it is removed with
  gatedtrain 0.9.0.

Copy `pipeline.yaml`, set `choose_task_and_base.base_model_dir` and the `image` of `smoke` and `train`, then
`gatedtrain project create ... --pipeline <copy> --keeper <socket>` and run the stages in order. Smoke runs before
training and refuses all-zero rewards and one repeated completion; NeMo RL writes no GPU memory log, so the memory
bound is the job finishing at the recipe's memory rung (`a`; `b` and `c` are the recipe's fallbacks via the
`memory_rung` parameter of `write_recipe`).

## Settings that depend on the GPU

Training runs under a memory rung (`a`, `b`, `c`: `write_recipe.memory_rung`), with vLLM `gpu_memory_utilization` 0.2
and `max_num_seqs` 128 for NeMo RL's colocated generation; the batch generator (`generate_vllm.py`) uses
`gpu_memory_utilization=0.85`, `max_model_len=1024` and, for greedy decoding, `max_num_seqs=1` with prefix caching off.
They were chosen and probed on one 24 GB GPU and are part of the pack's code (the generator's hash is registered); they
are not settings. `max_num_seqs=1` is what makes greedy decoding deterministic and must stay. The knobs a pipeline does
carry are not gates: `cuda_arch` changes only compile flags and `memory_rung` only how much memory training needs; the
smoke stage reports no memory bound here (NeMo RL writes no memory log), so no pipeline value can weaken a check.

**Re-probe on another GPU.** On a different GPU, vLLM or driver, run the smoke stage and the greedy determinism probe
under "Decoding" again: the memory rung that fits, the time of one arm against the generator's time limit, and the items
that differ between the two seeds (it must be 0).

## Deviations from the recipe

`runtime/run_vlm_nemo.py` takes its training file as `--train-file` instead of reading `<runtime>/data/train.jsonl`;
`ptc_vlm_hooks.py` no longer imports the recipe's `gen_vqa.py` or reads a `MODELS.json` (neither exists in the job): the
cut's `sampler.json` is the pack's registered deployment sampler and `base_ref.json` lists the base files' hashes. The
environment script and the prompt file are byte copies (tests check both against the recipe when it is beside the
pack, `GATEDTRAIN_RECIPE_VLM_CLEVR_DIR`; CI requires it). Training data, the sealed set, the sampler and the scorer's code path differ from
the recipe's published gate, so scores are not comparable with its `RESULT.md`.

## Tests

```
pip install -e ".[convert]" "gatedtrain==0.7.2"
python -m unittest discover -s tests -t tests
```

`GATEDTRAIN_SRC` (a gatedtrain v0.7.2 source checkout) enables the CPU end to end, which runs the stage machine, a
same-user keeper and fake containers; no GPU, Docker or model is used.

## Changes

### 0.3.2 (Unreleased)

Needs gatedtrain 0.7.2 (`gatedtrain>=0.7.2,<0.8`), which adds `gatedtrain setup eval-tree`. No code of the pack changed.

- **No sudo for the keeper's eval tree in a same-user keeper.** `gatedtrain setup eval-tree` prints a user-level script
  that builds it from this pack's `eval-requirements.txt` (see "What you need"); before, only the root script of
  `gatedtrain setup host` built it. A separate-user keeper still needs that script.
- Not changed, and not solved here: the NeMo RL trainer image (it is host specific and there is no registry to pull it
  from), the rootless derivative, and `memlock`.

### 0.3.1

Needs gatedtrain 0.7.1 (`gatedtrain>=0.7.1,<0.8`): it uses the host file's `[packs]` tables, `owner_gpu_mem_limit_gb` and the
keeper's `--eval-python`.


- **Owner-side channels.** The runtime directory is the host file's `[packs.vlm-clevr] runtime_dir` and the eval
  interpreter the keeper's `--eval-python`; the two variables are deprecated workarounds that print a notice (removed with
  gatedtrain 0.9.0 and pack 0.4.0). These need gatedtrain 0.7.1, which adds them. `GATEDTRAIN_PACK_VLM_CLEVR_EXTRA_IMAGES`
  (image IDs) is unchanged: image pins are moving to a lock file in a separate piece of work.

- **The GPU architecture is the owner's param.** `TORCH_CUDA_ARCH_LIST` and `FLASHINFER_CUDA_ARCH_LIST` were fixed at
  `8.9`, the GPU the recipe was written on. Set `cuda_arch` on `smoke` and `train` (for example `cuda_arch: "8.9"`,
  from `nvidia-smi --query-gpu=compute_cap`); without it the job exports neither and the toolchain reads the GPU. A
  pipeline copy that relied on the old behaviour on an 8.9 GPU adds it.
- Values the pack shared with gatedtrain are imported from it (the generator's time limit, the eval venv's location,
  the container run options); the image limits and magic bytes the stages repeated come from `data.py`, and the GPU-hour
  estimates are written once. `tests/test_single_source.py` checks the copies that must stay in the container scripts.

## License

Apache-2.0 (`LICENSE`, `NOTICE`); third-party material in `THIRD_PARTY_NOTICES.md`. Not affiliated with or endorsed by
the owners of the models, datasets or frameworks it uses.
