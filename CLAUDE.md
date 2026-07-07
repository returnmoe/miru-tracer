# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Miru Tracer is an experimental Gradio-based web application for interactive analysis of LLM text generation. It allows stepping through token generation one token at a time, visualizing probability distributions, manually selecting tokens, and logging generation sessions for analysis.

**Target users**: Researchers, educators, and developers debugging LLM prompts and generation behavior.

**Tech stack**: Python 3.11+, PyTorch 2.x, Transformers 5.x, Gradio 6.x, Plotly. Installable package (`pip install -e .`), src layout.

## Development Commands

```bash
# Install (CPU torch; the venv lives at .venv)
pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cpu

# Run the app (opens at http://127.0.0.1:7860)
miru-tracer            # or: python -m miru_tracer
MIRU_DEBUG=1 miru-tracer   # debug logging

# Tests
pytest                     # unit tests: fast, offline, tiny in-code model
pytest -m integration      # Qwen/Qwen3-0.6B end-to-end (downloads ~1.4GB once)

# Lint
ruff check src tests
```

Never `cd src` — the package is installed and imported as `miru_tracer.*`.

## Architecture

### Core (`src/miru_tracer/core/`)

**LLMTracer (tracer.py)** — the generation engine. Key design invariant:
`_next_raw_logits()` is the ONLY place the model is called. It keeps the KV
cache length equal to the sequence length after every forward and recovers
(crop or recompute) if they ever disagree — do not add model calls elsewhere.
Raw (pre-temperature) logits for the current position are memoized once per
position; `peek()` and `step()` derive temperature/top-k/top-p views from them
as cheap tensor ops, so changing display parameters never re-runs the model.
`undo(n)` crops the cache (`DynamicCache.crop`) and truncates `input_ids`; the
prompt is never re-tokenized after `reset()`. EOS is a frozenset built from
BOTH `tokenizer.eos_token_id` and `model.generation_config.eos_token_id`
(either may be an int, list, or None) — always check via `tracer.is_eos()`.

**sampling.py** — pure tensor functions (`SamplingParams`, `filter_top_k`,
`filter_top_p`, `select_token`, `entropy`). No model, no cache; this is the
easiest place to unit test sampling behavior. Reused by the tracer AND the
interactive-mode preview.

**schema.py** — `TokenStep` and the export JSON schema. Current is v2
(`schema_version: 2`, field `full_probs`); v1 logs used the misnamed
`all_logits` (values were probabilities). `parse_log()` reads both — keep it
backward compatible when evolving the schema.

**model_manager.py** — singleton loader. transformers 5: use `dtype=` (not
`torch_dtype=`) and `AutoModelForImageTextToText` (Vision2Seq was removed).
Quantization requires CUDA; on CPU it is skipped with an explicit
`quantization_note` in the returned info.

**session_manager.py** — thread-safe session store for Interactive Mode.
Gradio state carries only the session-id string; handlers call
`get_session(id)` once and hold `session.lock` while touching the tracer.
Lock ordering is global → session; never call manager methods while holding a
session lock.

### UI (`src/miru_tracer/ui/`)

One module per tab plus:
- **helpers.py** — shared chat-JSON validation, `ExportManager` (ONE stable
  export file per tab, overwritten in place — never create per-call temp
  files), numeric probability table, param clamping. Add shared logic here,
  not in tabs.
- **theme.py** — `MiruTheme`, CSS, footer JS. Gradio 6 takes these at
  `launch()`, not in the Blocks constructor; `launch_kwargs(version)` bundles
  them for app and tests.

Gradio 6 notes: `show_copy_button=` is now `buttons=["copy"]`; `gr.update()`
still works; generator handlers + `cancels=[...]` power the Stop buttons
(requires `demo.queue()`).

### Visualization (`src/miru_tracer/visualization/plots.py`)

Functions take a plain `list[TokenStep]` (live `tracer.history` or
`parse_log(...).history`) — never a tracer. Entropy is exact when
`full_probs` was logged, otherwise a renormalized top-k entropy and labeled
as such ("Top-k entropy") — keep that labeling honest.

## Testing Conventions

- Unit tests use a session-scoped tiny random Llama (2 layers, vocab 260)
  built from config in `tests/conftest.py` — offline, <1s to build, same
  cache/GQA machinery as Qwen3. The byte-level tokenizer is built in-code.
- `tests/unit/test_tracer_cache.py` is the regression suite for the historical
  KV-cache desync bug: forward counting, manufactured desync, randomized op
  sequences compared against no-cache ground truth. If you touch tracer
  internals, these must stay green.
- Integration tests (`-m integration`, excluded by default via pytest
  addopts) use `Qwen/Qwen3-0.6B` in fp32 on CPU and drive the real app with
  `gradio_client`.
- `tests/data/legacy_log_v1.json` pins the v1 export format; schema changes
  must keep it parseable.

## Configuration

Environment parsing lives ONLY in `config.py` (`env_bool`/`env_int`/
`env_str` + `Settings.from_env`). `MIRU_DEBUG` accepts 1/true/yes/on;
`MIRU_SERVER_NAME`/`MIRU_SERVER_PORT` bind the server (GRADIO_SERVER_* work
as fallbacks). Docker: `MIRU_SSH_ENABLE=1` (+ `MIRU_SSH_AUTHORIZED_KEYS`,
`MIRU_SSH_PORT`) enables the optional SSH server in the container.

## Gotchas

1. **Never call the model outside `_next_raw_logits()`** — the KV cache is
   mutated in place by forwards; a stray forward corrupts it. (Tests catch
   this.)
2. **Interactive handlers**: resolve the session once, hold `session.lock`
   for the whole operation, and return/yield the shared output tuple via
   `render_state`/`error_state`.
3. **Exports**: go through `ExportManager.prepare()` — and only at the end of
   an operation, never inside a per-token loop.
4. **Temperature 0** is fine: `SamplingParams` clamps it and torch softmax is
   numerically stable, but always build params via
   `helpers.ui_sampling_params` from widget values.
5. **Gradio State**: complex objects (model/tracer) only in Logging Mode's
   single-run state; Interactive Mode must keep using session ids.
