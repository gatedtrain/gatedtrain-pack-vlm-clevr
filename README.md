# gatedtrain-pack-vlm-clevr

A scenario pack for [gatedtrain](https://github.com/gatedtrain/gatedtrain) (`gatedtrain>=0.5,<0.6`) for
vision-language models: it teaches a small VLM to answer counting and yes/no questions about pictures, and lets
gatedtrain decide, by a rule fixed before training, whether the trained model really is better.

It trains Qwen3.5-0.8B with NeMo RL's VLM GRPO (the recipe `gatedtrain-recipe-nemo-rl-vlm-clevr`) on your
question-image-answer rows on your GPU. Before training, gatedtrain's keeper seals part of your rows away, images
included; after training, the keeper answers the sealed questions with the base and the trained model and scores each
answer itself. The gate passes only when the trained model beats the base by the registered rule.

It is a prototype: the models it trains are not promotable.

## A tested run

One run on one 24 GB GPU, with 1,834 CLEVR rows (one row per distinct image; 1,435 for training, 399 sealed for the
test), the 60-step recipe and the derived trainer image below:

| | Score on the sealed questions |
|---|---|
| Base (Qwen3.5-0.8B) | 0.425 |
| Trained (60 GRPO steps) | 0.567 |
| Gate | **PASS**: +0.142, 95.25 % interval [0.0896, 0.1924], power 0.97 |

About 0.5 GPU hours in all (smoke 6 minutes, training 21 minutes, keeper evaluation 4 minutes); `gatedtrain verify
--keeper` accepted the report. Scores are at temperature 0.7 over four seeds, so they are not comparable with the
recipe's published greedy gate. Your numbers will differ with other data, seeds and hardware. The CPU tests drive the
whole pipeline with fake containers.

## How it works

Same shape as `gatedtrain-pack-text-gsm8k`: the owner prepares data and trains; the keeper, a service under its own OS
user, holds the sealed rows and runs the evaluation. Stages of the pipeline (`pipeline.yaml`):

`intake_seal` -> `profile_data` -> `choose_task_and_base` -> `preregister` -> `build_train_set` -> `write_recipe` ->
`smoke` (5 GRPO steps, GPU) -> `train` (60 GRPO steps, GPU) -> `evaluate_sealed` -> `gate` -> `report`.

Pack stages: `clevr_profile_data`, `clevr_choose_task_and_base`, `clevr_build_train_set`, `clevr_write_recipe`,
`clevr_smoke`, `clevr_train`. The scorer `vqa_answer` takes the last `<answer>` block (else the last integer, number
word up to twenty, or yes/no of the response) and compares it with the label.

The deployment sampler is temperature 0.7, top_p 1.0, at most 512 new tokens, four seeds (11-14): gatedtrain's
registration refuses a temperature of 0, so the recipe's greedy gate cannot be registered here.

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

- Linux x86_64, one NVIDIA GPU with about 24 GB, rootless Docker with CDI (see gatedtrain's `setup host`), a running
  keeper (`gatedtrain setup host`).
- The derived NeMo RL image, held by your rootless Docker daemon under one of gatedtrain's pinned image IDs
  (`gatedtrain.families.nemo_rl.NEMO_RL_IMAGES`). A rebuilt image has another ID and is refused; copy the pinned one
  (`docker save` from the daemon that holds it, `docker load` into yours).
- On a rootless Docker host the pinned image cannot run as is: NeMo RL writes lock files beside vLLM files that belong to
  uid 65532 in it, and the container's root (the owner, no capabilities) may not. Build the derived image of
  `containers/nemo-rl-rootless/` (the same files, owned by root; no network) on your daemon, list its ID in
  `GATEDTRAIN_PACK_VLM_CLEVR_EXTRA_IMAGES` (comma separated) and use it as the `image` of `smoke` and `train`. It is
  not the image gatedtrain pins, deliberately; the job's flags are unchanged (no capabilities, no network, user 0:0).
  The runtime options also leave out `--ulimit memlock=-1`, which a rootless daemon refuses.
- The base model directory: `Qwen/Qwen3.5-0.8B` at commit `2fc06364715b967f1860aea9cf38778875588b17`, with its
  processor files (`preprocessor_config.json`, `video_preprocessor_config.json`) and `chat_template.jinja`.
- The keeper's eval venv (`gatedtrain setup host --eval-requirements <this package>/eval-requirements.txt`, CPython
  3.12, vLLM 0.31.0): the same packages and versions as the text pack's lock (whether vLLM 0.31.0 loads Qwen3.5-0.8B with images is for the first GPU probe to show; none is recorded yet). Set `GATEDTRAIN_PACK_VLM_CLEVR_EVAL_PYTHON` when the venv is not
  at `/opt/gatedtrain-keeper/eval-venv/bin/python`.
- `GATEDTRAIN_PACK_VLM_CLEVR_RUNTIME` names another runtime directory than the package's `runtime/` (it must hold the
  same scripts; the adapter reads it at import).

Copy `pipeline.yaml`, set `choose_task_and_base.base_model_dir` and the `image` of `smoke` and `train`, then
`gatedtrain project create ... --pipeline <copy> --keeper <socket>` and run the stages in order. Smoke runs before
training and refuses all-zero rewards and one repeated completion; NeMo RL writes no GPU memory log, so the memory
bound is the job finishing at the recipe's memory rung (`a`; `b` and `c` are the recipe's fallbacks via the
`memory_rung` parameter of `write_recipe`).

## Deviations from the recipe

`runtime/run_vlm_nemo.py` takes its training file as `--train-file` instead of reading `<runtime>/data/train.jsonl`;
`ptc_vlm_hooks.py` no longer imports the recipe's `gen_vqa.py` or reads a `MODELS.json` (neither exists in the job): the
cut's `sampler.json` is the pack's registered deployment sampler and `base_ref.json` lists the base files' hashes. The
environment script and the prompt file are byte copies (tests check both against the recipe when it is beside the
pack, `GATEDTRAIN_RECIPE_VLM_CLEVR_DIR`; CI requires it). Training data, the sealed set, the sampler and the scorer's code path differ from
the recipe's published gate, so scores are not comparable with its `RESULT.md`.

## Tests

```
pip install -e ".[convert]" "gatedtrain==0.5.0"
python -m unittest discover -s tests -t tests
```

`GATEDTRAIN_SRC` (a gatedtrain v0.5.0 source checkout) enables the CPU end to end, which runs the stage machine, a
same-user keeper and fake containers; no GPU, Docker or model is used.

## License

Apache-2.0 (`LICENSE`, `NOTICE`); third-party material in `THIRD_PARTY_NOTICES.md`. Not affiliated with or endorsed by
the owners of the models, datasets or frameworks it uses.
