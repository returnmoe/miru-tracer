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

A fit file (`lens.safetensors`) contains one `d_model × d_model` matrix per
layer — the model's input–output Jacobian averaged over a text corpus — plus
metadata describing its shape, model/corpus provenance, fit settings, and
convergence history (the most recent 1,000 points for unusually long fits).
For a 0.6B model that's ~50MB (fp16); for larger models
it scales with `n_layers × d_model²`, not with parameter count, so even a
27B-class model's fit file stays in the single-digit GB range.

Fit files are **per model** (same architecture, same weights). A lens fitted
for `Qwen/Qwen3-0.6B` is meaningless for any other checkpoint, and Miru
refuses files whose `d_model` doesn't match the loaded model.

## 2. Fitting (do this on a GPU)

Fitting runs one forward pass plus `d_model / dim_batch` **backward** passes
per prompt. The default budget is up to 1,000 prompts, but fitting can stop
after convergence once at least 100 prompts have succeeded. On a CPU that is
minutes per prompt — hours to days in total. On a single modern GPU it is
minutes to an hour. Fit where the compute is, then copy the file to wherever
you run Miru.

On the GPU instance:

```bash
# Download both the wheel and constraints.txt attached to the GitHub release:
pip install ./miru_tracer-0.3.2-py3-none-any.whl -c ./constraints.txt

# Or install from a checkout:
pip install -e .
miru-tracer-fit-lens Qwen/Qwen3-0.6B --dim-batch 32
```

The project dependency ranges are bounded to the minor versions tested for
v0.3.2; the attached `constraints.txt` selects the exact release-verified
versions.

Useful flags:

| Flag | Meaning |
|---|---|
| `--revision COMMIT` | Select a Hugging Face model revision. A full commit is recommended for controlled or comparative fits; default, branch, tag, and short-hash references are resolved once, then both model and tokenizer are pinned to and record the resulting immutable commit. |
| `--num-prompts N` | Maximum prompt budget (default 1,000). The run may finish earlier when the convergence criterion below is met. |
| `--min-prompts N` | Minimum number of successful prompts before convergence can stop the run (default 100). |
| `--stop-window N` | Number of recent successful prompt updates used for the rolling convergence check (default 10). |
| `--stop-at-delta X` | Stop when the window's mean relative change is strictly below `X` (default 0.002). Set `0` to disable convergence stopping and use the full prompt budget. |
| `--prompts-file f.txt` | Use your own corpus (one prompt per line, still capped by `--num-prompts`) instead of WikiText-103. Prompts should look like the model's pretraining distribution and be at least a few hundred characters (the fitter skips the first 16 token positions). |
| `--dim-batch K` | Jacobian rows per backward pass. Higher = fewer backward passes but more memory. 4–8 on CPU, 16–64 on GPU. |
| `--max-length N` | Maximum sequence length (default 128 tokens); longer prompts are truncated, while shorter prompts remain shorter. Raising it averages more positions but uses more memory. |
| `--chunk-size N` | Prompts between full checkpoint and partial-artifact writes (default 5). Larger values reduce storage traffic, but an abrupt process or node loss can discard more completed prompts. |
| `--empty-cuda-cache` | Run `torch.cuda.empty_cache()` after each saved chunk. Off by default; use it for a labelled A/B run when fragmentation is suspected, not as a general memory-leak fix. |
| `--telemetry-interval S` | Sample process, host, allocator, and physical-GPU state while a prompt is still running (default every 30 seconds; `0` disables periodic samples but retains prompt-boundary records). |
| `--device cuda --dtype bfloat16` | Defaults resolve to this automatically when CUDA is available. |
| `--require-fast-kernels` | For Gated DeltaNet models, fail before fitting if Miru detects Transformers' PyTorch fallback instead of the optional linear-attention and causal-convolution fast paths. |
| `--out path/lens.safetensors` | Write somewhere specific (default: Miru's lens cache). A `.pt` extension writes the legacy torch.save format instead. |
| `--fresh` | Discard the checkpoint and start over. |

### Paper parity and convergence

The paper's default lens used 1,000 sequences of exactly 128 tokens sampled
from an unnamed pretraining-like distribution. Miru matches the 1,000-prompt
budget and 128-token maximum, and its default WikiText-103 corpus serves the
same broad purpose, but it is not claimed to be the paper's undisclosed corpus.
For WikiText, Miru follows Neuronpedia by concatenating source rows and
rechunking them into roughly 2,000-character contexts before tokenization.
The built-in download is pinned to Salesforce/WikiText commit
`b08601e04326c79dfdd32d625aee71d232d685c3`, and that revision plus the
ordered prompt-sequence hash are stored in fit provenance.
Sequences are truncated rather than padded, so a short custom or final partial
prompt can still contribute fewer than 128 tokens. Like the
[released reference fitter](https://github.com/anthropics/jacobian-lens/blob/main/jlens/fitting.py),
it excludes the first 16 positions and the final position from each sequence
average.

This is parity with the released open-model fitter, not every experimental
choice in the paper: the paper's primary Sonnet 4.5 lens targeted the
penultimate residual layer, while the public fitter—and Miru's Qwen fits—use
the final residual layer by default.

The paper swept from 1 to 1,000 prompts: on its Sonnet 4.5 evaluations the
J-lens beat logit- and tuned-lens baselines with as few as 10 prompts, with
modest improvements from additional data. Miru nevertheless uses a
conservative floor of 100 successful prompts, then checks the mean relative
change over a rolling 10-prompt window. The default run stops when that mean
is strictly below 0.002. For each successful prompt, the relative change is
the mean across layers of
`‖new running mean − old running mean‖_F / ‖new running mean‖_F`; the stop
statistic is the arithmetic mean of those changes over the latest window.
This convergence stop is an operational addition, not a change to the
Jacobian estimator; the resulting artifact records the actual number of
prompts averaged. The Neuronpedia Qwen3-4B fit, for example,
[met the default criterion after 479 prompts](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/qwen3-4b/jlens/Salesforce-wikitext/config.yaml).

To reproduce the paper's full prompt-count budget rather than allow an early
stop, disable the threshold explicitly:

```bash
miru-tracer-fit-lens Qwen/Qwen3-4B --dim-batch 32 --stop-at-delta 0
```

Practical notes:

- **Resume safely.** The CLI writes the full checkpoint and partial artifact
  once per chunk (five prompts by default), rather than serializing a
  multi-gigabyte accumulator after every prompt. Re-running the same command
  resumes both the Jacobian average and its rolling convergence window. An
  abrupt process or node loss can discard the unfinished chunk; use
  `--chunk-size 1` for per-prompt persistence or a larger value to reduce I/O
  further. Checkpoints record and validate the ordered prompt prefix, model
  identifier/revision when available, model-config and tokenizer fingerprints,
  dtype, and estimator-affecting settings. Checkpoints created before this
  metadata was introduced cannot be matched to the newly rechunked corpus, so
  Miru refuses to resume them and directs you to restart with `--fresh`.
  Finished lens artifacts remain compatible. Mutable local model paths still
  cannot prove that their underlying weight file contents stayed unchanged.
- **Plan large-model I/O.** A fit with `L` source layers and hidden width `d`
  has an fp32 checkpoint of approximately `4 × L × d²` bytes and an fp16
  artifact of approximately `2 × L × d²` bytes. For 63 layers at width 5,120,
  those are about 6.6 GB and 3.3 GB respectively, so the default five-prompt
  cadence generates roughly 238 GB of logical writes by prompt 120 even though
  only the latest two files remain. Miru logs
  `fit_storage_plan` with the actual estimate, atomic-rewrite headroom,
  filesystem, mount point, free space, and any orphaned `.tmp.<pid>` files.
  Every `chunk_saved` record includes the remaining output-disk space. A
  caught write error removes its own temporary file; a hard kill cannot, so
  inspect and remove a reported orphan only when no fit is still targeting
  that output. Put large outputs on fast local storage when its durability is
  acceptable, or use a larger chunk such as 20–25 and accept the wider
  preemption window.
- **Partial fits work.** The output `lens.safetensors` is (re)written after
  every chunk and is a valid lens averaged over the prompts so far. You can
  start using it while the rest of the corpus finishes.
- **Safe to share.** Fit files are safetensors — no pickle, so copying one
  from an untrusted source can't execute code. Legacy `lens.pt` artifacts
  from older versions still load, and
  `miru-tracer-convert-lens lens.pt` rewrites one as `lens.safetensors`
  (uploading a `.pt` in the app converts it too).
- The progress log reports the convergence delta and rolling mean. Early
  stopping is considered only after `--min-prompts` successful prompts and a
  complete `--stop-window`; lower the threshold for a stricter fit, or use
  `--stop-at-delta 0` to disable it.

### Model-specific notes

The fitting **method is identical for every architecture** — the fitter only
needs residual blocks, a final norm, and an unembedding, all auto-detected.
What differs is logistics:

- **Llama / Qwen through Qwen3 / Mistral / Gemma ≤3** (e.g.
  `Qwen/Qwen3-0.6B`): nothing special. One consumer GPU fits models in this
  size class.
- **Qwen3.5/Qwen3.6 Gated DeltaNet** (e.g. `Qwen/Qwen3.6-27B`): Transformers
  can use optional Flash Linear Attention and causal-convolution extensions.
  Without both, it emits a warning and uses a substantially slower,
  more-memory-hungry PyTorch path—the supplied two-H100 incident logs were
  using that fallback. Miru records each detected component as `fit_backend`;
  add `--require-fast-kernels` when a controlled benchmark must reject the
  fallback. Multi-GPU `device_map` remains sequential layer sharding.
- **Gemma 4** (e.g. `google/gemma-4-31B`): loads through the same command —
  transformers resolves the multimodal checkpoint automatically and the
  fitter locates the text decoder inside it (`model.language_model`); the
  vision tower is simply unused. 31B in bf16 needs ~62GB: use one large GPU
  (H100/A100-80GB) or shard with `--device-map auto` across smaller ones.
- **GLM-5/5.2-style MoE** (e.g. `zai-org/GLM-5.2`): same command, but at
  753B total parameters this requires a multi-GPU node —
  `--device-map auto` is mandatory, and expect a long run. The MTP
  (speculative-decoding) layers are not part of the traced stack and are
  ignored by the fit. Do not fit MoE giants in fp32; keep the default
  bf16.

```bash
# multi-GPU sharding for models that don't fit on one device
miru-tracer-fit-lens zai-org/GLM-5.2 --device-map auto --dim-batch 16
```

### Long-run and multi-GPU diagnostics

In v0.3 the fit log records the runtime environment, resolved device map,
optional Gated DeltaNet backend, CUDA allocator backend, estimated
checkpoint/artifact sizes, output mount/filesystem, and one machine-readable
`fit_telemetry key=value` stream. `prompt_start` and `prompt_complete` bracket
each prompt; `prompt_sample` records the current encode, allocation, forward,
or backward/copy phase every 30 seconds, including backward-pass progress.
When a completed prompt is at least twice the rolling process-local median and
at least 60 seconds slower, `prompt_slowdown` emits an additional warning-level
snapshot. The boundary telemetry also separates tokenization, CPU allocation,
forward submission, backward/copy, and checkpoint/artifact time. It records
process RSS/PSS/swap and I/O; host load,
dirty/writeback memory, and Linux CPU/memory/I/O pressure stalls; PyTorch
allocated/reserved/active/inactive-split/peak/free memory, allocator retries,
and OOM counts for every CUDA device; and physical-GPU utilization, P-state,
device memory, clocks, power, and temperature from `nvidia-smi`. The
`nvidiaN_` prefix is a physical index and is intentionally distinct from the
process-local `cudaN_` index when `CUDA_VISIBLE_DEVICES` remaps devices.

Keep the complete stdout and stderr logs when reporting a slowdown. Compare
`process_prompt` (prompts completed in this process, independent of resumed
corpus position), the `prompt_sample` sequence inside the last healthy and
first slow prompt, its `phase`/`backward_pass` progress,
`phase_forward_enqueue_s`/`phase_backward_and_copy_s`, `proc_write_bytes`,
host I/O PSI,
GPU utilization/clocks/P-state, and allocated versus reserved/physical device
memory. That combination distinguishes compute throttling, storage stalls,
allocator fragmentation, and a retained live tensor much better than a single
memory number.

Treat a complete fit log as potentially identifying cluster data. Before
posting one publicly, redact or replace the hostname, PID, SLURM job ID,
`CUDA_VISIBLE_DEVICES`, output mount/path, and physical GPU UUIDs. Preserve the
timings, counters, driver version, device model, and consistent pseudonyms for
hosts/GPUs so the diagnostic sequence remains useful. Private reports may keep
the original identifiers when the recipient is authorized to receive them.
On a shared bare-metal node, `nvidia-smi` may expose physical GPUs outside the
job's CUDA allocation, so review those records as well as the process-local
`cudaN_` fields before sharing.

`--device-map auto` uses Accelerate's layer-dispatch path. It lets a model that
does not fit on one GPU run across several GPUs, but it is not data parallelism:
one prompt still travels through the sharded layers in order, so two GPUs do
not imply twice the fitting throughput and uneven memory use is expected. For
Qwen-family models with Gated DeltaNet layers, heed Transformers' warning if
the optional fast linear-attention or causal-convolution kernels are missing;
the PyTorch fallback is slower and more memory hungry. Miru records the backend
it can detect as `fit_backend` and emits `kind=linear_attention_fallback` when
appropriate. The stock Miru Docker images do not bundle these separately
compiled extensions; build and hardware-test a compatible derived image when
they are required, then use `--require-fast-kernels` to make that requirement
enforceable.

This diagnostics work followed a reproducible multi-GPU slowdown reported,
with cluster logs, by
[Lucas Teske (@racerxdl)](https://github.com/racerxdl). Those logs showed a
roughly 4.7× process-lifetime slowdown that disappeared on restart. They did
not prove a single CUDA allocator defect; Miru v0.3 fixes the confirmed
full-checkpoint write amplification and large-object lifetimes, and preserves
the new intra-prompt telemetry so any remaining backend or allocator component
can be isolated in the required post-release hardware reproduction. The exact
slowdown is not considered resolved merely by publishing v0.3.0.
`torch.cuda.empty_cache()` is therefore an opt-in
`--empty-cuda-cache` experiment. It does not increase the memory available to
PyTorch, although it can reduce fragmentation in some workloads; leaving it
off preserves the allocator's normal behavior and avoids mandatory
reallocation after every chunk.

### Fitting inside Docker (GPU cloud platforms)

Some GPU platforms only run Docker images. The Miru image ships the fit CLI,
so you can run a fitting job with the same image you deploy the app with.
Mount only the artifact directory on persistent/network storage; keep the
Hugging Face cache on fast instance-local storage with the separate
`--hf-home` option:

```bash
docker run --gpus all \
  -v /path/on/host/lenses:/lenses \
  miru-tracer \
  miru-tracer-fit-lens Qwen/Qwen3-0.6B \
  --dim-batch 32 \
  --out /lenses/Qwen--Qwen3-0.6B/lens.safetensors \
  --hf-home /tmp/huggingface

# resume after an interruption/preemption: identical command, it picks up
# from the checkpoint next to the --out path
```

The bind-mounted output directory must be writable by UID 10001 (`miru`) for
direct `docker run` commands. `--out` controls only the final artifact and
its sibling `.checkpoint.pt`; Hugging Face Hub, Xet, assets, and module caches
are controlled by `--hf-home` (or the standard Hugging Face environment when
that option is omitted). Short-lived atomic-write files may appear next to
the artifact while a chunk is being saved.

On RunPod, expose `22/tcp`, set `MIRU_AUTO_START_UI=0`, and deploy with
`startSsh: true`. RunPod's console may hide **SSH Terminal Access** for custom
images; in that case use `runpodctl pod create ... --ports 22/tcp --ssh=true`.
The SSH setting makes RunPod supply the account key through `PUBLIC_KEY`;
exposing the port alone does not. The SSH-only container stays alive with
`sshd` in the foreground; after logging in, use `tmux` for a persistent
session and run the same fit command with the network volume as `--out` and
`/tmp/huggingface` as `--hf-home`. To start Miru manually without exposing it
through the platform's HTTP proxy:

```bash
# inside the Pod
MIRU_SERVER_NAME=127.0.0.1 miru-tracer

# workstation (replace host/port with RunPod's Direct TCP mapping)
ssh -L 7860:127.0.0.1:7860 -p <external-ssh-port> root@<pod-ip>
```

The default `miru-tracer` image is CUDA 13.0 and supports Blackwell on an
R580.65.06+ host. Use `miru-tracer:0.3.2-cu126` for an older driver—including
the R550.54.15 driver in the reported two-H100 run—or a
Maxwell CC 5.x (except 5.3),
Pascal, or Volta GPU. Both images contain the matching CUDA userspace libraries;
the host kernel driver still comes from the cloud platform/NVIDIA container runtime.

## 3. Loading a fit file into Miru Tracer

Miru looks for fit files at:

```
$MIRU_LENS_DIR/<model-name-sanitized>/lens.safetensors
# default: ~/.cache/miru-tracer/lenses/Qwen--Qwen3-0.6B/lens.safetensors
# a legacy lens.pt in the same directory still loads (safetensors wins if both exist)
```

Three equivalent ways to get your file there:

1. **Upload in the app**: Lens tab → *"Jacobian lens fit file"* accordion →
   drop the `lens.safetensors` (or a legacy `lens.pt`) in. Miru validates it
   against the loaded model (`d_model`, layer count, model identity, resolved
   revision, normalized config, and tokenizer fingerprint when recorded) and
   installs it in the cache — always as safetensors. Confirmed conflicts are
   rejected. A v0.2 artifact has only the legacy full-configuration
   fingerprint, which included load settings such as dtype; a mismatch in that
   fingerprint alone has been a warning since v0.3.1 because it cannot
   distinguish a changed architecture from a different load configuration.
   Older and upstream lenses without provenance remain loadable, but "Check
   status" warns that their identity cannot be verified.
2. **Copy it yourself**:
   `scp gpu-box:lens.safetensors ~/.cache/miru-tracer/lenses/Qwen--Qwen3-0.6B/lens.safetensors`
   (the directory name is the model name with `/` replaced by `--`).
3. **Fit directly into place**: if you run `miru-tracer-fit-lens` on the same
   machine without `--out`, it already writes to the cache path.

The app picks up new or updated files automatically (no restart needed — the
store checks the file's mtime). If you have independently established that a
checkpoint is correct despite an explicit provenance conflict, select **Force
lens despite provenance conflicts**. The override applies only to the exact
loaded model object and current lens file and clears when either changes. It
does not bypass residual-width or layer-range checks.

If a legacy lens records `"model_commit_hash": null`, the artifact itself
cannot identify the Hub snapshot used for fitting, and its full-config digest
cannot be reversed to recover one. Pin **Hugging Face revision / commit SHA**
only when you can recover the commit from the original fit log, retained
Hugging Face cache, or another run record. Otherwise the model name, tokenizer
fingerprint, and structural checks are the available evidence, and Miru reports
that the exact revision remains unverified.

New Hub-backed fits resolve and record this commit automatically even when
Transformers does not populate its private `config._commit_hash`. A null value
is still normal for a genuinely local checkpoint, which has no Hub revision;
its normalized configuration and local manifest provide the applicable
identity evidence. Both `miru-tracer-fit-lens` and the main application emit a
grep-friendly INFO record containing the full resolved `commit_sha`.

## 4. Using the lenses

Load a model in the Model Loader tab, optionally entering a full immutable
Hugging Face commit in **Hugging Face revision / commit SHA**. Miru passes that
same revision to the model and tokenizer and reports and logs the resolved
commit. Then open the **Lens** tab:

1. **Generate & Analyze** runs the prompt (optionally with interventions —
   see below) and computes readouts for every (layer, position) cell.
2. **Selection**: click tokens in the sequence to restrict positions
   (none selected = all), set the layer range/stride, choose the lens mode —
   *Logit*, *Jacobian*, or *Compare (Jacobian / Logit)* (the two independent
   visualizations side by side) — and press **Update readouts** to re-slice without
   regenerating.
3. **Readouts**: All Layers orders tokens by descending occurrence count, using
   reciprocal-rank relevance to break ties, and shows each vocabulary ID plus
   occurrence strips across layers. With exactly one sequence token selected,
   hover a layer to preview its exact candidates and probabilities, click to
   lock it, and use **All Layers** to clear the lock. Exact-layer rows retain
   both token IDs and probabilities. Compare shares one layer selector across
   independent Jacobian and Logit columns. Active interventions are shown above
   the inspector.
4. **Interactive Mode** has a *"Layer Lens"* accordion showing the per-layer
   residual readout after the current token while you step.

**Candidates per layer+position** controls the top-N retained in each cell
(default 8). **All Layers rows** independently caps the aggregate list (default
100), so requesting eight candidates no longer limits the whole Readouts view
to eight tokens.

Lens labels preserve the tokenizer's exact vocabulary form. When a byte-level
token decodes to useful non-ASCII text, the readable form is appended—for
example, Qwen's `æ³ķåĽ½` appears as `æ³ķåĽ½ (法国)`. Ordinary ASCII labels stay
compact, while incomplete UTF-8 fragments remain in their raw tokenizer form.

Selecting token position `p` decodes the block-output residual **at that same
position `p`**, after the displayed token has entered the causal context. This
matches Neuronpedia's position convention. Consequently, the final identity
layer is the model's true next-token distribution from that state: it predicts
the token after the selected token. Position 0 is a valid state and is included.
A fitted Jacobian readout remains broader than the final distribution: it
describes verbalizable concepts in the selected state that can influence
current or future computation.

Miru versions before v0.3.0 instead displayed token `p` while decoding residual
position `p−1`. That attached readouts to a different causal state. v0.3.0
changes the readout slicer and UI selection semantics; that alignment fix does
not change the Jacobian estimator or matrices. v0.3.0 also changes fitter
checkpoint cadence, memory cleanup, and diagnostics, but the artifact and
checkpoint schemas remain unchanged. Existing lenses created by
`miru-tracer-fit-lens` or the upstream reference implementation remain valid
and do **not** need retraining. v0.3.0 could reject v0.2 Miru lenses because
their legacy full-config hash changed with runtime settings; since v0.3.1 Miru
treats that ambiguous mismatch as a warning. Provenance-free older/upstream
artifacts also show an identity-verification warning; neither warning indicates
damaged Jacobians. Re-run the inexpensive readout step to replace screenshots,
tables, or conclusions produced with the old alignment.

The recommended Jacobian/Compare range starts after the first 29% of layers,
which are commonly degenerate, and always includes the final model-output layer
as a reference. Enter a nonnegative **From layer** explicitly to override this;
early included layers are marked with a warning. Meaningful J-lens tokens
typically appear from the early-middle layers onward and sharpen toward the
top, while the logit lens often becomes readable only in the last few layers.

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
interventions can be active at once**. Each Add action creates one table item,
even when its layer selection expands to several layer edits. Table items (and
their layer edits) compose in order inside every forward pass. They apply on
the next **Generate & Analyze** (the readouts then reflect the edited
activations too), and can be applied to an Interactive Mode session from its
Layer Lens panel. Removing all interventions restores the model exactly —
nothing is permanently modified.

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
| Llama / Qwen through Qwen3 / Mistral / Gemma ≤3 / OLMo | `Qwen/Qwen3-0.6B` (integration-tested for real) | Standard `model.layers` layout. |
| **Qwen3.5/Qwen3.6 hybrid Gated DeltaNet** | Tiny Qwen3.5 (full trace/lens/intervention matrix); `Qwen/Qwen3.6-27B` is scheduled for post-release v0.3 hardware validation | Standard residual layout, but hybrid recurrent caches cannot be cropped. Miru invalidates and rebuilds them on undo/replay. Optional fast kernels are detected separately from the PyTorch fallback. |
| **Gemma 4** | `google/gemma-4-31B` | Text-only and multimodal-wrapper classes both detected; logit softcapping handled; per-layer input embeddings are internal to blocks and don't affect readouts. Multimodal models run in text-only mode. 31B ⇒ GPU (4-bit is fine: quantization skips the LM head, so lens directions stay full-precision). |
| **GLM MoE-DSA** | `zai-org/GLM-5.2` | Standard layout; MoE and sparse attention are internal to blocks; MTP layers are not part of the traced stack. Note: MLA-style attention computes prefill and decode slightly differently, so cached logits can differ from a from-scratch forward by ~1e-3 (rankings unaffected). At 753B parameters this is multi-GPU-server territory — architectural support is verified at tiny scale. |
| GPT-2 / Phi / GPT-NeoX (Pythia) | `gpt2` | Covered by the vendored layout table. |

Unknown architectures: if the model exposes a text decoder with
`layers`/`norm`/`embed_tokens` and an `lm_head`, detection usually works; the
error message tells you when it doesn't.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Jacobian/Compare need a fitted lens" | No fit file for the loaded model — see §2–3. Logit mode always works. |
| "fit file has d_model=X, model has d_model=Y" | The file was fitted for a different model. |
| "none of the selected layers are fitted" | Your fit covers fewer layers than the range you selected (e.g. a partial fit); widen the stride or refit. |
| J-lens readouts look like noise everywhere | Too few prompts in the fit, or a corpus very unlike the model's pretraining data. |
| Fitting OOMs on GPU | Lower `--dim-batch` or `--max-length`. |
| Qwen3.5/Qwen3.6 reports that its fast path is unavailable | Install and hardware-test compatible Flash Linear Attention and causal-convolution extensions in a derived environment, or accept Transformers' slower fallback. Use `--require-fast-kernels` when fallback would invalidate the run. |
| A long fit slows down after many process-local prompts | Keep both output streams and compare the `fit_telemetry` phase, allocator, physical-GPU, PSI, and I/O fields around the transition. Retry from the durable checkpoint in a fresh process; use `--empty-cuda-cache` only as a labelled A/B test. |
| `<12345>` shown as a readout token | The id is in the model's (padded) embedding matrix but not in the tokenizer vocab; normal for some models. |
