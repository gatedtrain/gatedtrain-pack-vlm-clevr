# Third-party notices

This pack ports code and one prompt from the gatedtrain recipe `gatedtrain-recipe-nemo-rl-vlm-clevr`
(<https://github.com/gatedtrain/gatedtrain-recipe-nemo-rl-vlm-clevr> at commit `92b51a69a6ca060e49afeaf667b520239ff581d3`,
Apache-2.0, the same license as this pack): `src/gatedtrain_pack_vlm_clevr/runtime/` holds its launcher
`run_vlm_nemo.py` (the training file is an argument; see the README), its hooks `ptc_vlm_hooks.py` (the cut's sampler
and base reference no longer come from the recipe's `gen_vqa.py` and `MODELS.json`; see the README), its environment
`vlm_env_logged.py` unchanged, and the prompt `prompt_clevr_cot.txt`. Names such as `ptc_` and the schema ids `ptc_mm.*`
are kept from the recipe, whose published run carries them. These scripts call NeMo RL (Apache-2.0, below) at run time
and contain none of its source. Its test fixtures are synthetic rows and images, not CLEVR data. The lock files
(`eval/uv.lock`, `eval-requirements.txt`) name third-party packages that are installed from PyPI under their own
licenses; nothing of them is copied here.

## NeMo RL VLM prompt

- **What:** `runtime/prompt_clevr_cot.txt` is a byte copy of NeMo RL's `examples/prompts/clevr_cogent_cot.txt`, so the
  training prompt and the evaluation prompt are the recipe's.
- **Source:** NeMo RL, <https://github.com/NVIDIA-NeMo/RL> at commit `4f39fd86289a2ffd5aaf5d202702f7fd7185c924`.
- **License:** Apache-2.0.

## Models and data the pack uses (not included)

- Qwen3.5-0.8B (`Qwen/Qwen3.5-0.8B`, Apache-2.0), downloaded by the owner.
- CLEVR-CoGenT (CC BY 4.0) when the owner converts it as data; nothing of it is redistributed here.
