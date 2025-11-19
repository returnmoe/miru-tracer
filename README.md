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

- Load any HuggingFace model (with optional 4-bit/8-bit quantization)
- Tokenize text and inspect individual tokens
- Interactive mode: step through generation manually, pick specific tokens, undo
  steps
- Logging mode: generate text automatically while recording full probability
  distributions
- Analysis: visualize generation logs with heatmaps and probability curves

## Installation

Requires Python 3.10+. CUDA GPU recommended but not required.

```bash
git clone <repository-url>
cd miru-tracer
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Note: You may want to skip `flash-attn` in requirements.txt if you don't have
CUDA.

## Usage

```bash
python app.py
```

Opens at `http://127.0.0.1:7860`

### Basic workflow

1. Load a model in the "Model Loader" tab (try `gpt2` for testing)
2. Go to "Interactive Mode"
3. Enter a prompt and click "Initialize"
4. Click "Next Step" to generate tokens one at a time
5. See the probability distribution for each step
6. Optionally override token selection or undo steps

### Logging mode

Use "Logging Mode" to automatically generate text while recording probabilities.
Export the log as JSON and analyze it in the "Log Analysis" tab.

## Configuration

Optional `.env` file:

```bash
MIRU_DEBUG=0  # Set to 1 for debug output
HF_TOKEN=your_token_here  # Only needed for gated models
```

### Generation parameters

- **Temperature**: Lower = more deterministic, higher = more random
- **Top-K**: Limit sampling to K most likely tokens
- **Top-P**: Nucleus sampling threshold
- **Strategy**: Greedy (always pick top token) or Sampling (random from
  distribution)

## Performance tips

- Use 4-bit or 8-bit quantization for models 7B+
- Start with small models like `gpt2` (124M params)
- Don't enable "log all logits" unless you need full distributions (uses ~600KB
  per step)

## Project structure

```
app.py                      # Main Gradio app
src/core/tracer.py         # Core LLMTracer class
src/core/models.py         # ModelManager and data models
src/ui/                    # Gradio UI tabs
src/visualization/         # Plotly visualizations
```

## License

This project is released into the public domain under the [Unlicense](LICENSE).
