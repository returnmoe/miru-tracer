# Changelog

## 0.3.1 — 2026-07-26

### Lens provenance compatibility hotfix

v0.3.0 could reject a valid lens produced by `miru-tracer-fit-lens` v0.2 even
when the Hugging Face model repository had not changed. The v0.2 fitter
recorded a SHA-256 fingerprint of the complete serialized Transformers
configuration. That serialization also contains load/runtime fields such as
the requested dtype and Transformers version. A lens fitted on CUDA with the
default `bfloat16`, then used with Miru's `float16` UI loader, could therefore
report a full-configuration mismatch without any change to the architecture or
weights. Such provenance was not malformed; the legacy fingerprint was too
broad for the compatibility decision v0.3.0 made from it.

Version 0.3.1:

- treats a mismatched legacy full-configuration fingerprint as an explicit
  warning when the artifact predates the normalized architecture fingerprint;
  independently verifiable model-name, revision, tokenizer, and normalized
  architecture conflicts still fail closed;
- adds an optional **Hugging Face revision / commit SHA** field to Model Loader
  and passes the same resolved immutable commit to both the Hugging Face model
  and tokenizer. Miru retains and displays that commit without depending on
  Transformers' private configuration fields;
- makes `miru-tracer-fit-lens` resolve the selected/default Hub revision before
  loading either component and record that immutable commit automatically.
  New Hub-backed fits no longer write `model_commit_hash: null` merely because
  Transformers omitted its private `config._commit_hash`. Both the fitter and
  the main application write the full resolved SHA as `commit_sha` in their
  INFO logs;
- adds an explicit **Force lens despite provenance conflicts** control for
  exceptional recovery. It bypasses only provenance conflicts, remains scoped
  to the exact loaded model object and current lens file, and is cleared when
  either changes; and
- keeps residual-width and layer-range checks mandatory even under the force
  option, so a structurally incompatible lens cannot be loaded.

The Jacobian matrices, estimator, and artifact schema are unchanged. Existing
v0.2 lenses do not need to be refitted.

## 0.3.0 — 2026-07-26

### Lens fitting: bounded checkpoint I/O and cluster diagnostics

A multi-GPU Qwen fit was observed to slow from roughly 400 seconds per prompt
to roughly 1,880 seconds after 119 prompts in one process, then return to its
original rate after restart. The same 119-prompt transition recurred across
independent processes. This issue and the supporting cluster logs were reported
by [Lucas Teske (@racerxdl)](https://github.com/racerxdl).

The logs establish a process-lifetime slowdown, but not a single CUDA root
cause. Inspection did find concrete Miru-side pressure that could produce or
amplify it: the chunked CLI wrote the entire fp32 checkpoint after every
prompt, wrote the same checkpoint again at the end of every chunk, and retained
an extra loaded checkpoint plus the previous full fp32 lens in memory. For a
63-layer, 5,120-wide fit, one checkpoint is about 6.6 GB; the old defaults
wrote about 1 TB of checkpoint and artifact data before prompt 120.

The exact process-lifetime slowdown has not yet been reproduced with v0.3.0
on the reported two-H100 setup. This release deliberately ships the fixes and
instrumentation needed for that post-release reproduction; it should not be
read as a claim that the slowdown's root cause is known or eliminated.
Evidence from the released artifacts may lead to focused v0.3.x follow-ups.

Version 0.3.0 therefore:

- writes one resumable checkpoint and one partial artifact per chunk (five
  prompts by default) instead of one full checkpoint per prompt;
- suppresses duplicate writes of unchanged checkpoint state;
- reads the initial resume checkpoint once (rather than loading the full state
  once to find its index and again to resume), releases that probe, the
  per-prompt Jacobian, captured autograd tensors, and prior CLI progress lens
  before the next chunk, and forms the final mean in place;
- runs garbage collection at chunk boundaries, with an opt-in
  `--empty-cuda-cache` A/B mode. Cache emptying is not enabled by default:
  PyTorch documents it as a fragmentation intervention rather than a way to
  increase memory available to PyTorch, and forcing reallocations would make
  allocator diagnosis harder;
- adds `--chunk-size` so operators can trade write frequency against the
  amount of work an abrupt process/node loss may discard;
- adds `--revision` so controlled fits can pin an immutable Hugging Face model
  commit instead of relying on a moving repository default;
- pins the built-in WikiText-103 corpus to dataset commit
  `b08601e04326c79dfdd32d625aee71d232d685c3` and records that revision in fit
  provenance;
- emits stable `fit_environment`, `fit_backend`, `fit_checkpoint`, `fit_io`,
  and `fit_telemetry key=value` records. Prompt start/completion records are
  supplemented every 30 seconds by an intra-prompt sample naming the active
  encode/allocation/forward/backward-copy phase and backward-pass progress.
  A prompt exceeding both 2× the rolling process-local median and 60 seconds
  of absolute excess emits a warning-level diagnostic snapshot. Telemetry
  includes phase timings, RSS/PSS/swap, page faults, context switches, process I/O, host
  dirty/writeback memory, Linux pressure-stall counters, filesystem type,
  file descriptors, allocator backend, and allocated/reserved/peak/free
  memory plus allocator retries and OOMs for every CUDA device. It also records
  physical-GPU utilization, P-state, clocks, power, temperature, and device
  memory through `nvidia-smi`. OOMs dump PyTorch allocator summaries; and
- warns when a Gated DeltaNet fallback backend is detected and when
  Accelerate's inference-oriented `device_map` dispatch is being used. A
  device map is layer sharding, not data parallelism, so two GPUs should not be
  expected to halve fitting time. `--require-fast-kernels` turns the fallback
  warning into a pre-fit error for controlled cluster runs.

The default interruption guarantee changes accordingly: a normal chunk
completion is fully resumable, while an abrupt kill can lose the unfinished
chunk (at most five prompts with the default). `fit()` callers that explicitly
use `checkpoint_every=1` retain per-prompt checkpointing, without the duplicate
final write.

Required checkpoint and artifact payloads and their schema versions remain
backward compatible; v0.3 adds optional provenance and diagnostic metadata.
Atomic checkpoint and artifact writes now remove their temporary file after a
caught failure, warn about hard-kill leftovers on the next startup, and record
remaining output-disk space per saved chunk. The supplied slow-run Jacobians
remained finite, and no traceback, CUDA error, OOM, or artifact corruption was
present in those runs. A slowdown alone is not a reason to retrain a completed
lens; retraining is warranted only if a run logged non-finite/skipped prompts
unexpectedly, failed validation, or produced an incomplete lens.

### Correctness and release hardening

The v0.3 audit found and fixed several additional correctness and release
hardening problems:

- The fitter used to catch every `ValueError` raised while processing a
  prompt and describe it as a short prompt. Backend/model `ValueError`s now
  propagate; only the dedicated `PromptTooShortError` is skipped. A corpus
  whose first chunk is entirely short continues to later prompts, while an
  entirely unusable corpus fails with a clear error.
- Transformers' hybrid Qwen3.5/Qwen3.6 cache exposes a top-level `crop()`, but
  its recurrent linear-attention layers intentionally do not rewind. Miru's
  undo/replay path could therefore reuse state containing undone tokens. Miru
  now detects non-rewindable layer caches and performs a clean re-forward;
  ordinary attention-only caches retain the fast crop path.
- Lens artifacts are now checked against the loaded model before Jacobian
  readout, Jacobian-basis intervention, or UI installation. Matching dimensions
  are no longer treated as sufficient: v0.3 artifacts record and compare the
  Hub model identity, resolved revision, a normalized model-config fingerprint,
  and tokenizer fingerprint. Confirmed conflicts fail closed. The normalized
  architecture fingerprint takes precedence over the legacy full Transformers
  config hash when only runtime/load fields such as dtype differ, so valid
  lenses fitted from local checkpoints are not rejected merely because no Hub
  commit is available. Older and upstream artifacts without provenance remain
  usable, with an explicit identity-verification warning rather than a false
  claim of compatibility.
- Runtime, GPU, and development dependencies are bounded to the minor release
  lines exercised by v0.3 CI. Each GitHub Release attaches the exact tested
  `constraints.txt` beside the exact wheel installed and smoke-tested by the
  release gate (not a later rebuild), and `SHA256SUMS` covers both.
- The development test runner is updated to pytest 9.0.3. Torch remains on the
  release-tested 2.12.1 CUDA line; Miru does not invoke the affected local
  [`torch.jit.script` path](https://github.com/advisories/GHSA-rrmf-rvhw-rf47).
  Torch 2.12.1 requires `setuptools<82`, so the constrained setuptools 81
  [macOS-sdist advisory](https://github.com/advisories/GHSA-h35f-9h28-mq5c)
  is accepted for v0.3, which publishes a wheel and source archive but no
  Python sdist.

CI now tests the same-position Logit-lens invariant against a real Qwen model,
builds and installs the wheel, and exercises a tiny hybrid Qwen3.5 model across
tracing, undo/replay, lens readout, and interventions. The gated release
workflow accepts either a manual dispatch or a validated
`release/v<version>` branch push and requires an explicit acknowledgement that
the multi-GPU slowdown remains unresolved. For the raw-Git path, deliberately
pushing that branch is the acknowledgement; it does not publish the version
tag early. Both CUDA images are built and service/hardening smoke-tested under
immutable commit tags, then promoted to exact version tags before the Git tag,
curated changelog notes, and wheel are published. Mutable minor and `latest`
aliases are promoted last. A release candidate that was current `master` when
dispatched remains the candidate throughout the workflow even if `master`
advances during the image builds, preventing a published tag and wheel from
being stranded without their exact version image aliases. A process-local,
two-H100 Qwen3.6-27B run extending past the previously observed slowdown
boundary is post-release validation for v0.3.0 and a source of telemetry for
any necessary v0.3.x patch.

### Breaking: Lens positions now identify residual states

The Lens tab and Interactive Mode now decode residual position `p` when token
position `p` is displayed or selected. A causal transformer's block output at
`p` includes token `p` in its context, so the final Logit-lens row is the
model's true distribution for token `p + 1`.

Miru 0.2.x instead decoded residual position `p - 1` while labeling the result
with token `p`. That convention showed the state that predicted a generated
token, but it did not match Neuronpedia's position semantics and attached the
visible token to a different activation vector. Position 0 was also
unnecessarily hidden. The 0.3.0 convention makes every visible position
state-aligned, includes position 0, and keeps Jacobian and Logit readouts on the
same residual tensor.

This changes the meaning of existing token-position selections:

- The activation Miru 0.2.x showed under token `p` is the activation that
  Miru 0.3.0 shows at position `p - 1`, under the preceding token.
- Miru 0.3.0 token `p` shows the state after token `p` and predicts the token
  that follows it.
- Previously saved screenshots, tables, or interpretations should be
  recomputed before comparison with 0.3.0 results.

The obsolete `token_aligned` argument to `compute_lens_slice` and the redundant
`LensSlice.source_positions` field have been removed. `LensSlice.positions` now
always names both the displayed-token position and the decoded residual
position.

### Existing fitted lenses remain valid

No Jacobian-lens retraining is required. `miru-tracer-fit-lens` estimates a
position-agnostic per-layer transport matrix; it does not encode the viewer's
displayed-token offset. The estimator and required matrix/checkpoint payloads
are unchanged. v0.3 adds optional provenance fields without changing the
artifact or checkpoint schema versions, so existing Miru and upstream
Jacobian-lens artifacts can be loaded and used directly.
