# Vendored code notice

The files in this directory are vendored from Anthropic's **jacobian-lens**
repository (package `jlens` v0.1.0), companion code to the paper
*"Verbalizable Representations Form a Global Workspace in Language Models"*
(https://transformer-circuits.pub/2026/workspace/index.html).

- Source: local clone of `jacobian-lens`, commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` (2026-07-01)
- Vendored on: 2026-07-07
- License: **Apache-2.0** (see `LICENSE` in this directory). This differs from
  the rest of Miru Tracer, which is released under the Unlicense.

## Files and modifications

| File | Modifications from upstream |
|---|---|
| `hooks.py` | Import paths rewritten `jlens.*` → `miru_tracer.core._jlens.*` |
| `lens.py` | Import paths rewritten |
| `fitting.py` | Import paths rewritten |
| `protocol.py` | Import paths rewritten |
| `hf.py` | Import paths rewritten |
| `__init__.py` | New file (subset of upstream public API re-exports) |

Not vendored: `vis.py` (d3/HTML visualization — Miru Tracer renders lens data
with its own Plotly code), `examples.py`, `_logging.py`, `data/`.
