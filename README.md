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

## Configuration

Optional `.env` file (see `.env.example`):

```bash
MIRU_DEBUG=0                 # 1/true/yes/on enables debug logging + Gradio debug
MIRU_SERVER_NAME=127.0.0.1   # bind address (0.0.0.0 to expose to network)
MIRU_SERVER_PORT=7860        # bind port
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
  ui/                       # One module per tab + shared helpers/theme
  visualization/plots.py    # Plotly figures from TokenStep histories
tests/                      # pytest suite (unit + integration)
```

## License

This project is released into the public domain under the [Unlicense](LICENSE).
