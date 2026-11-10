# RBench MUSA review checkout

Upstream: https://github.com/DAGroup-PKU/ReVidgen, `origin/main`.
Review baseline: `b03df27f0376faa148dcd8cd620a1989a32ca979`.
Local branch: `codex/20261110-rbench-musa`.

- This local Git checkout is the authority for RBench changes. Remote copies only execute jobs.
- Keep metric definitions, prompts, frame selection and aggregation formulas aligned with upstream.
- MUSA entry points and installation helpers live in `scripts/musa/`; dependency profiles in `environment/musa/`.
- Keep the container's default Python and installed torch/torch_musa/torchvision stack. Use separate containers for VLM and motion dependencies.
- Build CUDA-to-MUSA extensions in disposable copies: torchada ports C++ source in place.
- Never commit models, datasets, scores, credentials or compiled extensions.
- CPU checks: `python3 -m unittest discover -s tests -v`; `git diff --check`.
- Real MUSA builds, model loading and scoring require separate validation; CPU checks do not establish those results.

Review/run instructions: [docs/musa.md](docs/musa.md).
