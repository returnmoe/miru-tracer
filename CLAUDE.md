# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Miru Tracer is an experimental Gradio-based web application for interactive analysis of LLM text generation. It allows stepping through token generation one token at a time, visualizing probability distributions, manually selecting tokens, and logging generation sessions for analysis.

**Target users**: Researchers, educators, and developers debugging LLM prompts and generation behavior.

**Tech stack**: Python 3.10+, PyTorch, Transformers (HuggingFace), Gradio 5.x, Plotly

## Development Commands

### Running the application
```bash
# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Run locally (opens at http://127.0.0.1:7860)
cd src && python app.py

# Enable debug mode
cd src && MIRU_DEBUG=1 python app.py

# Run with Gradio hot reload (for UI development)
cd src && gradio app.py
```

### Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Note: flash-attn is commented out by default (requires CUDA)
# Uncomment in requirements.txt if you have CUDA and want performance optimization
```

### Docker
```bash
# Build image
docker build -t miru-tracer .

# Run container (GPU required)
docker run --gpus all -p 7860:7860 miru-tracer
```

## Architecture Overview

### Core Components

**ModelManager (src/core/models.py)**
- Singleton pattern for model/tokenizer loading
- Handles quantization (4-bit/8-bit via bitsandbytes)
- Device detection (CUDA/CPU) and memory management
- Shared across all UI tabs

**LLMTracer (src/core/tracer.py)**
- Main generation engine with step-by-step token control
- Supports two modes: completion (direct text) and chat (applies tokenizer chat template)
- KV cache optimization: only processes last token when past_key_values exists
- Probability caching: `cache_next_probabilities()` avoids redundant forward passes
- Key methods:
  - `reset()`: Initialize with prompt or messages
  - `step()`: Generate one token (greedy or sampling), returns TokenStep
  - `generate()` / `generate_stream()`: Auto-generate multiple tokens
  - `undo_step()`: Remove last token and rebuild state
  - `export_to_dict()`: Export history to JSON

**SessionManager (src/core/session_manager.py)**
- Thread-safe session isolation for Interactive Mode
- Each session has unique UUID, LLMTracer instance, and thread lock
- Prevents race conditions from rapid Gradio interactions
- Auto-cleanup of inactive sessions (30min timeout)

### UI Structure (src/ui/)

Each tab is a separate module returning a `gr.Tab`:

1. **model_loader.py**: Load HuggingFace models with quantization options
2. **tokenize_text.py**: Tokenize input and inspect individual tokens
3. **token_lookup.py**: Lookup tokens by ID or text
4. **interactive_mode.py**: Step-through generation with manual token override
5. **logging_mode.py**: Auto-generate with full probability logging (export as JSON)
6. **analysis.py**: Visualize logged generations (heatmaps, probability curves)

### Visualization (src/visualization/plots.py)

Plotly-based charts for log analysis:
- **Heatmap**: Token probabilities over generation steps
- **Probability curve**: Selected token probability over time
- **Top-K evolution**: How top candidates change each step

### State Management Pattern

**Interactive Mode uses session-based state**:
- Gradio state only stores `session_id` (string)
- Actual LLMTracer lives in SessionManager
- All operations acquire session lock before modifying tracer
- This prevents Gradio's serialization issues with complex objects

**Logging Mode uses direct state**:
- Stores LLMTracer in Gradio state (simpler, no multi-step interaction)

## Key Implementation Details

### Temperature Handling
- Guard against temperature <= 0 (use 1e-10 minimum) to avoid division by zero in softmax
- Applied in `get_next_token_probabilities()` and `step()`

### Token Decoding
- Use `safe_decode_token()` (src/core/tokenizer_utils.py) which handles byte fallback tokens
- Returns tuple: (decoded_string, raw_fallback) for display

### Probability Cache Optimization
- Interactive mode caches next token probabilities at end of step
- Reused at start of next step to avoid redundant forward pass
- Invalidated on state change (undo, manual override)

### Warning System
- `suppress_warnings=True` in `step()` collects warnings instead of logging immediately
- Retrieved via `get_warnings()` - useful for batch generation
- Warns once per session about: memory usage (log_all_logits), deterministic sampling

### Chat vs Completion Mode
- Auto-detected based on input type (messages vs prompt)
- Chat mode applies tokenizer.chat_template if available, else concatenates messages
- Both modes share same generation logic after tokenization

## Configuration

### Environment Variables (.env)
```bash
MIRU_DEBUG=0        # Set to 1 for debug logging
HF_TOKEN=...        # Only needed for gated models
```

### Generation Parameters
- **Temperature**: Controls randomness (lower = deterministic, higher = creative)
- **Top-K**: Limits sampling to K most likely tokens
- **Top-P**: Nucleus sampling - cumulative probability threshold
- **Strategy**: "greedy" (always top-1) or "sampling" (random from distribution)

### Memory Considerations
- `log_all_logits=True` uses ~600KB per step for 150K vocab (avoid unless needed)
- Use quantization (4-bit/8-bit) for models 7B+ parameters
- Test with small models first (e.g., Qwen/Qwen3-1.7B)

## Common Gotchas

1. **KV Cache Invalidation**: Always reset `past_key_values = None` and call `invalidate_probability_cache()` when modifying input_ids (undo, reset)

2. **Thread Safety**: In Interactive Mode, always acquire session lock before tracer operations

3. **Gradio State**: Don't store complex objects (model, tokenizer, tracer) directly in Gradio state for multi-step interactions - use SessionManager pattern instead

4. **Tokenizer Padding**: ModelManager sets `pad_token = eos_token` if missing (common issue)

5. **Chat Template Fallback**: If tokenizer lacks chat_template, concatenate messages manually

## Testing Workflow

1. Load a small model (Qwen/Qwen3-1.7B recommended for testing)
2. Try "Tokenize Text" tab to verify model loaded correctly
3. Use "Interactive Mode" to step through generation manually
4. Try "Logging Mode" to record full generation trace
5. Analyze logged trace in "Log Analysis" tab

## File Organization

```
src/
  app.py                        # Main Gradio app entry point
  core/
    models.py                   # ModelManager, TokenStep dataclass
    tracer.py                   # LLMTracer generation engine
    session_manager.py          # Thread-safe session isolation
    tokenizer_utils.py          # Token decoding utilities
    logging_config.py           # Python logging setup
  ui/
    model_loader.py             # Tab: Load models
    tokenize_text.py            # Tab: Inspect tokenization
    token_lookup.py             # Tab: Token ID/text lookup
    interactive_mode.py         # Tab: Step-through generation
    logging_mode.py             # Tab: Auto-generate with logging
    analysis.py                 # Tab: Visualize generation logs
  visualization/
    plots.py                    # Plotly visualization functions
```
