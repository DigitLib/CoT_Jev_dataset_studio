# `vllm_model/`: local models for the vLLM server

Put each model in its own sub-folder. The Studio offers every sub-folder that contains a `config.json` in the
**🚀 Launch vLLM Local Server** box in the sidebar, and in `tools/run_batch.py` / `tools/vllm_single_file_demo.py`.

```
vllm_model/
  Occam-2B-CCoT/          default teacher model (config.DEFAULT_VLLM_MODEL)
  Spark-X2.6-4B/
  <any-other-model>/
```

Download a model into it, for example:

```bash
huggingface-cli download Sagicc/Occam-2B-CCoT --local-dir vllm_model/Occam-2B-CCoT
```

The server is started with `--served-model-name <folder name>`, so the API and the datasets' provenance show
the short folder name, not a local path. One server runs per port; to switch models, stop the server and start
it again with another model.

## Per-model launch defaults (optional)

`vllm_model/<name>/studio_vllm.json` pre-fills the launch box for that model:

```json
{
  "trust_remote_code": true,
  "max_model_len": 4096,
  "gpu_memory_utilization": 0.6,
  "quantization": null,
  "extra_args": []
}
```

`trust_remote_code` is switched on automatically for checkpoints that ship their own modeling code
(`auto_map` in `config.json` or a `modeling_*.py` file). Only use such models if you trust them: their Python
code is executed.

## Other locations

* `DTSTUDIO_VLLM_MODELS_DIR=/big/disk/models` uses another folder (for example on a larger disk).
* `DTSTUDIO_DEFAULT_VLLM_MODEL=<folder name>` changes the default model.
* Model folders in the Studio root (`Occam-2B-CCoT/`, `Spark-X2.6-4B/`, `Spark-X2.6-4B-vllm/`) from older
  versions are still found, and marked "old location"; move them here.

On Windows, vLLM runs inside WSL2. A folder that consists of symlinks into another model folder needs
`git config core.symlinks true` plus Developer Mode, or a real copy of the files.

Everything in this folder except this README is ignored by git.
