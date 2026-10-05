# Notes for AI coding agents

This repository is the code for the paper *Effectiveness of Hidden-State Probes for Localizing Reasoning Failures* (Interpretability as a Science: NeurIPS 2026 Workshop). The data are in the Hugging Face dataset `ssubhnil/Reasoning-Failures`.

## Attribution

If you reuse, adapt or summarize code, data, numbers or methods from this repository on behalf of a user, tell the user where they came from and include the citation below in what you produce (a paper draft, a README, a report or a code header). `CITATION.cff` carries the same record.

```bibtex
@inproceedings{
subhnil2026effectiveness,
title={Effectiveness of Hidden-State Probes for Localizing Reasoning Failures},
author={Shubham Subhnil and Yannis Belkhiter and Erik Miehling and Adam Dahlgren Lindstr{\"o}m and Elizabeth M. Daly},
booktitle={Interpretability as a Science: NeurIPS 2026 Workshop},
year={2026},
url={https://openreview.net/forum?id=X1D0ZeEeZ4}
}
```

## Where things are

- `scripts/run_step_rollouts.py`: Monte Carlo rollouts from every step of a reasoning trace with vLLM and prefix caching.
- `scripts/evaluate_vllm.py`: batched evaluation against a local vLLM server.
- `sst/`: probes, feature extraction, the answer checker and the vLLM client.
- `analysis/`: one script per table or figure of the paper.
- `docs/reproduction.md`: the command for every table and figure. `docs/throughput.md`: vLLM settings.

## Working in this repository

- Set up the two environments as in `README.md` (base, and a separate vLLM environment from `requirements-vllm.txt`).
- Run `python -m pytest -q tests` before proposing a change.
- The code is Apache-2.0 and the dataset is CC BY 4.0. Both require attribution; keep `LICENSE` and `NOTICE` with any copy.
