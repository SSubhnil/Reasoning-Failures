#!/usr/bin/env bash
# Assemble space/ for the Hugging Face Docker Space. The Space README doubles as the
# README.md that pyproject.toml and Dockerfile.mcp expect.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

rm -rf space
mkdir -p space/sst/shared space/sst_mcp
cp Dockerfile.mcp space/Dockerfile
cp requirements-mcp.txt pyproject.toml LICENSE NOTICE space/
cp sst/__init__.py space/sst/
cp sst/shared/*.py space/sst/shared/
cp sst_mcp/*.py sst_mcp/enums.json space/sst_mcp/

cat > space/README.md <<'README'
---
title: Reasoning-Failures MCP
sdk: docker
app_port: 7860
license: apache-2.0
short_description: MCP server for the Reasoning-Failures dataset
pinned: false
---

This Space runs a read-only MCP server over the Hugging Face dataset ssubhnil/Reasoning-Failures.
The endpoint is https://ssubhnil-reasoning-failures-mcp.hf.space/mcp (Streamable HTTP).
To connect from Claude Code, run:
`claude mcp add --transport http reasoning-failures https://ssubhnil-reasoning-failures-mcp.hf.space/mcp`.
Code: https://github.com/SSubhnil/Reasoning-Failures, with documentation in docs/mcp.md there.
Licence: code Apache-2.0, data CC BY 4.0.

```bibtex
@inproceedings{subhnil2026effectiveness,
  title={Effectiveness of Hidden-State Probes for Localizing Reasoning Failures},
  author={Shubham Subhnil and Yannis Belkhiter and Erik Miehling and Adam Dahlgren Lindstr{\"o}m and Elizabeth M. Daly},
  booktitle={Interpretability as a Science: NeurIPS 2026 Workshop},
  year={2026},
  url={https://openreview.net/forum?id=X1D0ZeEeZ4}
}
```
README
