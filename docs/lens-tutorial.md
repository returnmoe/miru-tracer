# Lens tutorial: fitting, loading, and using the Jacobian lens

> **⚠️ Experimental**: the lens and intervention features were implemented
> very recently and are still being tested. Readouts and intervention effects
> from *our implementation* may be wrong or nonsensical in ways we haven't
> caught yet — don't treat them as ground truth about the model without
> cross-checking (e.g. against the final layer, which must always match the
> model's real output distribution).

Miru Tracer ships two "lenses" for looking inside a model while it generates:

- The **logit lens** projects each layer's residual stream through the
  model's own unembedding matrix. It answers *"if the model had to speak from
  this layer, what would it say?"* It works out of the box for **any** loaded
  model — no preparation needed.
- The **Jacobian lens** (J-lens, from Anthropic's
  ["Verbalizable Representations Form a Global Workspace in Language Models"](https://transformer-circuits.pub/2026/workspace/index.html))
  first transports the residual through a fitted per-layer matrix
  `J_ℓ = E[∂h_final/∂h_ℓ]` before unembedding. This corrects for how
  representations change across layers, recovering interpretable readouts in
  early and middle layers where the logit lens produces noise. **It needs a
  per-model fit file.**

This document covers producing that fit file, loading it, and what to do with
it once you have it.

## 1. What a fit file is

A fit file (`lens.pt`) contains one `d_model × d_model` matrix per layer —
the model's input–output Jacobian averaged over a text corpus — plus a little
metadata (layer indices, prompt count, `d_model`). For a 0.6B model that's
~50MB (fp16); for larger models it scales with `n_layers × d_model²`, not
with parameter count, so even a 27B-class model's fit file stays in the
single-digit GB range.

Fit files are **per model** (same architecture, same weights). A lens fitted
for `Qwen/Qwen3-0.6B` is meaningless for any other checkpoint, and Miru
refuses files whose `d_model` doesn't match the loaded model.

## 2. Fitting (do this on a GPU)

Fitting runs one forward pass plus `d_model / dim_batch` **backward** passes
per prompt, over 100–1,000 prompts. On a CPU that is minutes per prompt —
hours to days in total. On a single modern GPU it is minutes to an hour.
Fit where the compute is, then copy the file to wherever you run Miru.

On the GPU instance:

```bash
pip install miru-tracer  # or: pip install -e . from a checkout
miru-tracer-fit-lens Qwen/Qwen3-0.6B --dim-batch 32
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--num-prompts N` | Corpus size (default 100). More prompts = a better-converged average; the paper used ~1,000. Quality degrades gracefully below that. |
| `--prompts-file f.txt` | Use your own corpus (one prompt per line) instead of wikitext-103. Prompts should look like the model's pretraining distribution and be at least a few hundred characters (the fitter skips the first 16 token positions). |
| `--dim-batch K` | Jacobian rows per backward pass. Higher = fewer backward passes but more memory. 4–8 on CPU, 16–64 on GPU. |
| `--max-length N` | Truncate prompts to N tokens (default 128). Longer = more positions averaged per prompt, more memory. |
| `--device cuda --dtype bfloat16` | Defaults resolve to this automatically when CUDA is available. |
| `--out path/lens.pt` | Write somewhere specific (default: Miru's lens cache). |
| `--fresh` | Discard the checkpoint and start over. |

Practical notes:

- **Interrupt freely.** Fitting checkpoints after every prompt. Ctrl-C (or a
  spot-instance reclaim) loses nothing; re-running the same command resumes.
- **Partial fits work.** The output `lens.pt` is (re)written after every
  chunk and is a valid lens averaged over the prompts so far. You can start
  using it while the rest of the corpus finishes.
- Watch the per-prompt log line `max_d_mean` — it tracks how much each new
  prompt still moves the running average. Once it's consistently small, more
  prompts buy little.

## 3. Loading a fit file into Miru Tracer

Miru looks for fit files at:

```
$MIRU_LENS_DIR/<model-name-sanitized>/lens.pt
# default: ~/.cache/miru-tracer/lenses/Qwen--Qwen3-0.6B/lens.pt
```

Three equivalent ways to get your file there:

1. **Upload in the app**: Lens tab → *"Jacobian lens fit file"* accordion →
   drop the `lens.pt` in. Miru validates it against the loaded model
   (`d_model`, layer count) and installs it in the cache. "Check status"
   shows what's currently loaded.
2. **Copy it yourself**:
   `scp gpu-box:lens.pt ~/.cache/miru-tracer/lenses/Qwen--Qwen3-0.6B/lens.pt`
   (the directory name is the model name with `/` replaced by `--`).
3. **Fit directly into place**: if you run `miru-tracer-fit-lens` on the same
   machine without `--out`, it already writes to the cache path.

The app picks up new or updated files automatically (no restart needed — the
store checks the file's mtime).

## 4. Using the lenses

Load a model (Model Loader tab), then open the **Lens** tab:

1. **Generate & Analyze** runs the prompt (optionally with interventions —
   see below) and computes readouts for every (layer, position) cell.
2. **Selection**: click tokens in the sequence to restrict positions
   (none selected = all), set the layer range/stride, choose the lens mode —
   *Logit*, *Jacobian*, or *Diff* (what the J-lens boosts relative to the
   logit lens) — and press **Update readouts** to re-slice without
   regenerating.
3. **Readouts**: the aggregated table shows which tokens appear across the
   selected cells and at which layers (sparkline column); below it are the
   count-by-layer heatmap, the position × layer top-1 heatmap, and rank
   trajectories for any **pinned tokens** (comma-separated text or ids).
4. **Interactive Mode** has a *"Layer Lens"* accordion showing the per-layer
   readout of the current next-token position while you step.

Reading the picture: with a fitted J-lens, meaningful tokens typically appear
from the early-middle layers onward and sharpen toward the top; the logit
lens usually only becomes readable in the last few layers. Both agree at the
final layer (which is exactly the model's real next-token distribution).

## 5. Interventions (steer / swap / ablate)

The Lens tab's *Interventions* section edits residual-stream directions
during generation:

- **Steer** — add the token's lens direction, scaled by *strength* × the
  activation's norm. Positive pushes toward the concept, negative away.
- **Ablate** — remove the activation's component along the direction.
- **Swap** — transfer the coefficient from one token's direction to
  another's, preserving everything orthogonal.

Each intervention picks a **layer** and a **basis**: `jacobian` (the
direction that *reads out* as the token at that layer — needs the fit file)
or `logit` (the raw unembedding direction — always available, most meaningful
near the final layer).

Unlike Neuronpedia's single-intervention steering, **any number of
interventions can be active at once** — add several rows and they compose,
in order, inside every forward pass. They apply on the next
**Generate & Analyze** (the readouts then reflect the edited activations
too), and can be applied to an Interactive Mode session from its Layer Lens
panel. Removing all interventions restores the model exactly — nothing is
permanently modified.

Suggested first experiment: prompt `The capital of France is`, add
`steer, token = " Berlin", layer ≈ 2/3 of the way up, strength 2–3,
basis jacobian`, regenerate, and watch both the output text and where
" Berlin" enters the readout heatmap.

## 6. Architecture support

The lens/intervention stack auto-detects where the residual blocks, final
norm, embedding, and LM head live. Verified architectures (via tiny random
models of each family in the test suite — `tests/unit/test_arch_matrix.py` —
plus Qwen3-0.6B end-to-end):

| Family | Example | Notes |
|---|---|---|
| Llama / Qwen / Mistral / Gemma ≤3 / OLMo | `Qwen/Qwen3-0.6B` (integration-tested for real) | Standard `model.layers` layout. |
| **Gemma 4** | `google/gemma-4-31B` | Text-only and multimodal-wrapper classes both detected; logit softcapping handled; per-layer input embeddings are internal to blocks and don't affect readouts. Multimodal models run in text-only mode. 31B ⇒ GPU (4-bit is fine: quantization skips the LM head, so lens directions stay full-precision). |
| **GLM MoE-DSA** | `zai-org/GLM-5.2` | Standard layout; MoE and sparse attention are internal to blocks; MTP layers are not part of the traced stack. Note: MLA-style attention computes prefill and decode slightly differently, so cached logits can differ from a from-scratch forward by ~1e-3 (rankings unaffected). At 753B parameters this is multi-GPU-server territory — architectural support is verified at tiny scale. |
| GPT-2 / Phi / GPT-NeoX (Pythia) | `gpt2` | Covered by the vendored layout table. |

Unknown architectures: if the model exposes a text decoder with
`layers`/`norm`/`embed_tokens` and an `lm_head`, detection usually works; the
error message tells you when it doesn't.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Jacobian/Diff need a fitted lens" | No fit file for the loaded model — see §2–3. Logit mode always works. |
| "fit file has d_model=X, model has d_model=Y" | The file was fitted for a different model. |
| "none of the selected layers are fitted" | Your fit covers fewer layers than the range you selected (e.g. a partial fit); widen the stride or refit. |
| J-lens readouts look like noise everywhere | Too few prompts in the fit, or a corpus very unlike the model's pretraining data. |
| Fitting OOMs on GPU | Lower `--dim-batch` or `--max-length`. |
| `<12345>` shown as a readout token | The id is in the model's (padded) embedding matrix but not in the tokenizer vocab; normal for some models. |
