# Miru Tracer

An experimental tool for interactive analysis of LLM text generation, token by
token.

> **⚠️ Disclaimer**: This is an experimental tool in active development.
> Features may change, and you may encounter bugs or unexpected behavior. Use at
> your own discretion.

## What is this?

Miru Tracer is a Gradio web interface that lets you step through LLM text
generation one token at a time. It shows you the probability distributions, lets
you manually pick tokens, and helps you understand what's happening inside the
generation process.

Useful for:

- Understanding how LLMs generate text
- Debugging prompts and generation settings
- Exploring alternative generation paths
- Educational purposes

## Features

- Load any HuggingFace model (with optional 4-bit/8-bit quantization on CUDA)
- Tokenize text and inspect individual tokens
- Interactive mode: step through generation manually, pick specific tokens,
  undo steps, jump to any earlier step
- Logging mode: generate text automatically while recording per-step
  probability distributions (raw and temperature-adjusted)
- Analysis: visualize exported logs with heatmaps and confidence curves —
  current and older log formats both load
- **Lens**: read out what intermediate layers are "thinking" with the logit
  lens and the [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html)
  (Anthropic, 2026) — per-position, per-layer readouts, aggregated readout
  browsing, pinned-token rank tracking
- **Interventions**: steer, swap, or ablate lens readout directions during
  generation, with any number of interventions active simultaneously

## Installation

Requires Python 3.11+. Works on CPU; a CUDA GPU helps with larger models.

```bash
git clone <repository-url>
cd miru-tracer
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# CPU (default; also what CI uses)
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu

# CUDA instead: install plain torch and the gpu extras
# pip install -e .[gpu]
```

`constraints.txt` pins a fully verified dependency set if you want exact
reproducibility: `pip install -e . -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu`

## Usage

```bash
miru-tracer          # or: python -m miru_tracer
```

Opens at `http://127.0.0.1:7860`

### Basic workflow

1. Load a model in the "Model Loader" tab (try `Qwen/Qwen3-0.6B` for a small,
   capable test model)
2. Go to "Interactive Mode"
3. Enter a prompt and click "Initialize"
4. Click "Next Step" to generate tokens one at a time
5. See the probability distribution for each step
6. Optionally override token selection, undo steps, or jump to a step

### Logging mode

Use "Logging Mode" to automatically generate text while recording
probabilities. Export the log as JSON and analyze it in the "Log Analysis" tab.

### Lenses and interventions

> **⚠️ Experimental**: the lens/intervention features were implemented very
> recently and are still being tested — readouts and intervention effects
> from our implementation may currently yield nonsense. Cross-check before
> drawing conclusions. See [docs/lens-tutorial.md](docs/lens-tutorial.md).

The **logit lens** projects each layer's residual stream through the model's
own unembedding; it works out of the box for any loaded model. The
**Jacobian lens** first transports the residual through a fitted per-layer
matrix `J_ℓ = E[∂h_final/∂h_ℓ]`, recovering interpretable readouts in early
and middle layers where the logit lens fails.

The Jacobian lens needs a **fit file** per model. Fitting is compute-heavy
(many backward passes per prompt) — run it **on a GPU instance** and bring
the file back:

```bash
# on the GPU box
miru-tracer-fit-lens Qwen/Qwen3-0.6B --dim-batch 32
# then load lens.safetensors via the Lens tab's "fit file" section, or copy it
# to ~/.cache/miru-tracer/lenses/<model>/lens.safetensors (override with
# MIRU_LENS_DIR)
```

By default the fitter considers up to 1,000 prompts, truncates each sequence
at 128 tokens, and will not stop for convergence before 100 prompts succeed.
After that floor, it stops when the rolling arithmetic mean of the latest 10
relative-change updates falls below 0.002; for example, the Neuronpedia
[Qwen3-4B fit](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/qwen3-4b/jlens/Salesforce-wikitext/config.yaml)
converged after 479 prompts.
Tune the floor and window with `--min-prompts` and `--stop-window`, or pass
`--stop-at-delta 0` to force the full 1,000-prompt budget. Fitting is
checkpointed (interrupt + re-run resumes), and partial fits are usable. The
full walkthrough — corpus choice, convergence flags, loading, and reading the
plots — is in [docs/lens-tutorial.md](docs/lens-tutorial.md).

In the **Lens tab** you can select layer ranges and token positions, browse
aggregated readouts (which tokens appear across the selected cells, and at
which layers), pin tokens to track their rank across layers, and add
**steer / swap / ablate** interventions on readout directions. Multiple
interventions can be active at once; they apply during generation and are
also reflected in the readouts. Interactive Mode has a "Layer Lens" panel
showing a per-layer readout aligned to the current token and can
apply the Lens tab's interventions to its session.

Architecture support: the Llama/Qwen/Mistral/Gemma family, **Gemma 4**
(text-only and multimodal wrappers, softcapping included), **GLM MoE-DSA**
(GLM-5/5.2 style), GPT-2/Phi/NeoX — auto-detected, with a per-architecture
test matrix at tiny scale. See the support table in
[docs/lens-tutorial.md](docs/lens-tutorial.md) for practical caveats
(model sizes, quantization, MLA numerical notes).

## Configuration

Optional `.env` file (see `.env.example`):

```bash
MIRU_DEBUG=0                 # 1/true/yes/on enables debug logging + Gradio debug
MIRU_SERVER_NAME=127.0.0.1   # bind address (0.0.0.0 to expose to network)
MIRU_SERVER_PORT=7860        # bind port
MIRU_LENS_DIR=               # fitted-lens cache (default ~/.cache/miru-tracer/lenses)
HF_TOKEN=your_token_here     # only needed for gated models
```

### Generation parameters

- **Temperature**: Lower = more deterministic, higher = more random
- **Top-K**: Limit sampling to K most likely tokens
- **Top-P**: Nucleus sampling threshold
- **Strategy**: Greedy (always pick top token) or Sampling (random from
  distribution)

## Development

```bash
pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cpu

pytest                   # unit tests (fast, offline, tiny in-code model)
pytest -m integration    # end-to-end tests with Qwen/Qwen3-0.6B (~1.4GB download)
ruff check src tests     # lint
```

The unit suite runs entirely on CPU with a tiny randomly-initialized model
built in-code — no network, suitable for CI.

## Docker

```bash
docker build -t miru-tracer .                 # CUDA image (default)
docker build -t miru-tracer:cpu \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .

docker run --gpus all -p 7860:7860 miru-tracer
```

## Performance tips

- Use 4-bit or 8-bit quantization for models 7B+ (requires CUDA)
- Start with small models like `Qwen/Qwen3-0.6B`
- Don't enable "Log full probabilities" unless you need entire distributions
  (~600KB per step for a 150K vocab)

## Project structure

```
src/miru_tracer/
  app.py                    # Gradio app assembly + entry point
  config.py                 # Environment-based settings
  core/
    tracer.py               # LLMTracer generation engine (KV-cache safe)
    sampling.py             # Pure sampling/post-processing functions
    schema.py               # TokenStep + versioned export/import
    model_manager.py        # Model/tokenizer loading
    session_manager.py      # Thread-safe session isolation
    lens.py                 # Logit/Jacobian lens readout engine
    lens_fit.py             # Lens fitting (library + CLI)
    interventions.py        # Steer/swap/ablate activation edits
    _jlens/                 # Vendored Anthropic jacobian-lens (Apache-2.0)
  ui/                       # One module per tab + shared helpers/theme
  visualization/plots.py    # Plotly figures (histories + lens slices)
tests/                      # pytest suite (unit + integration)
```

## License

Miru Tracer is released into the public domain under the
[Unlicense](LICENSE), **with one exception**: the code under
`src/miru_tracer/core/_jlens/` is vendored from Anthropic's
[jacobian-lens](https://github.com/anthropics/jacobian-lens) reference
implementation and remains under the **Apache License 2.0** (see
`src/miru_tracer/core/_jlens/LICENSE` and `VENDORED.md` there for provenance
and modifications).
