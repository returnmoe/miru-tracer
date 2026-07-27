# Miru Tracer release checklist

Unless a section is explicitly marked post-release, this checklist is a
release gate rather than a claim that every item has already passed. The
Qwen3.6-27B reproduction is deliberately post-release: the reporter needs the
published v0.3 instrumentation to run it on the affected hardware. Shipping
v0.3.1 therefore accepts a known diagnostic uncertainty; it does not establish
that the reported slowdown is fixed.

## 1. Source and metadata

- Release from the current `master` commit with no uncommitted release edits.
- Confirm `pyproject.toml`, the Docker examples, and the requested tag contain
  the same version.
- Confirm every direct dependency remains bounded to the tested minor line and
  `constraints.txt` contains the exact candidate environment.
- Record the accepted dependency exceptions: Torch 2.12.1 remains on the
  release-tested CUDA line and Miru does not invoke the affected local
  [`torch.jit.script` path](https://github.com/advisories/GHSA-rrmf-rvhw-rf47);
  Torch requires `setuptools<82`, so the constrained setuptools 81
  [macOS-sdist advisory](https://github.com/advisories/GHSA-h35f-9h28-mq5c)
  is accepted because v0.3 publishes no sdist. Pytest is pinned to the fixed
  9.0.3 release.
- Confirm the dated `0.3.1` changelog entry matches the publication date.
  Keep the Lucas Teske credit in the published section.
- Confirm the release notes explain the v0.2 full-config compatibility fix,
  automatic immutable-commit capture in the fitter, revision-pinned UI
  loading, full `commit_sha` logging in both executables, the provenance-only
  force option, and why structural lens checks remain mandatory. Keep the
  v0.3.0 notes for the same-position migration, chunk persistence, diagnostics,
  hybrid-cache replay fix, and optional-kernel policy.
- Confirm no `v0.3.1` tag exists at another commit.

## 2. Automated gates

The CI and manual release workflows must pass:

- Python 3.12 and 3.13 unit/integration suites and Ruff lint/format checks.
- The real `Qwen/Qwen3-0.6B` external-model suite, including the exact
  same-position final Logit-lens comparison.
- The tiny architecture matrix, including Qwen3.5 hybrid-cache undo/replay,
  lens readout, and interventions.
- Wheel build, installation, dependency check, version assertion, and both
  lens CLI `--help` entry points. The release job must publish that exact
  preserved wheel, not an untested rebuild.
- CUDA 12.6 and 13.0 Docker builds and service/hardening smoke tests against
  the exact pushed immutable tags. The image label
  `io.returnmoe.miru-tracer.qwen-fast-kernels=not-bundled` must be present.

The release workflow reruns the source, real-Qwen, and wheel gates. It builds
both images under immutable `sha-<full-commit>-cu*` tags, smoke-tests those
registry images, and promotes exact version aliases before publishing release
metadata. Mutable minor and `latest` aliases are promoted afterward.

## 3. Publication

After the source metadata is ready, CI has passed, and the known risk is
accepted:

1. Either dispatch the `Release` workflow on `master`, or push the current
   `master` commit to the versioned release branch:
   `git push origin HEAD:refs/heads/release/v0.3.1`. A branch-triggered release
   rejects version mismatches and release branches that do not point at current
   `master`; it does not create the public version tag before the image gates.
2. For a manual dispatch, enter `0.3.1` and acknowledge that the multi-GPU
   slowdown's root cause remains unresolved. For the raw-Git path, deliberately
   pushing the versioned release branch is that acknowledgement. Neither path
   is a hardware-test result; the workflow reruns the release-specific
   automated gates before publishing.
3. Confirm both immutable and promoted GHCR tags resolve, the GitHub Release
   uses the curated changelog section, and the wheel, exact `constraints.txt`,
   plus `SHA256SUMS` are attached.
4. Verify both checksums, install the attached wheel in a clean environment
   using the attached constraints file, and pull both exact version image tags
   once more.

Miru currently publishes a GitHub Release wheel and GHCR images. Do not
advertise a PyPI release unless a separate, verified PyPI publication step is
added.

## 4. Post-release two-H100 Qwen3.6-27B validation

This validation is required before describing the slowdown as resolved, but
it does not block v0.3.1. Use a fresh output directory, the exact published
v0.3.1 wheel or image, the same pinned model revision and prompt corpus for
comparisons, two H100s, bf16, and the `device_map=balanced` path from the
reported incident. Preserve stdout, stderr, scheduler accounting, and
periodic system monitoring. The supplied incident environment used
Transformers' PyTorch Gated DeltaNet fallback; the reproduction must exercise
the stock Miru environment and record its `fit_backend` lines rather than
silently changing that variable.

The supplied run used NVIDIA driver 550.54.15. On that host, use
`ghcr.io/returnmoe/miru-tracer:0.3.1-cu126`; the unqualified CUDA 13.0 image
requires R580.65.06 or newer. Record the exact image digest in the log.

Run at least 160 successful prompts in one process—resuming at corpus prompt
137 and running only a few more does not cross the observed process-lifetime
boundary:

```bash
set -o pipefail
miru-tracer-fit-lens Qwen/Qwen3.6-27B \
  --revision FULL_40_HEX_MODEL_COMMIT \
  --device-map balanced \
  --dtype bfloat16 \
  --dim-batch 32 \
  --num-prompts 160 \
  --stop-at-delta 0 \
  --chunk-size 5 \
  --telemetry-interval 30 \
  --out /fast-local/miru-v0.3-soak/lens.safetensors \
  2>&1 | tee /durable-logs/miru-v0.3-soak.log
```

Do not enable `--empty-cuda-cache` in the primary run. It is an allocator
intervention and the default behavior needs validation. If a transition is
observed, run a fresh, otherwise identical A/B job with
`--empty-cuda-cache`; do not overwrite the primary checkpoint or log.

Acceptance criteria:

- Prompts 121–160 do not show a sustained return of the reported roughly
  4.7× slowdown. Compare medians for process-local prompts 20–100 and 121–160,
  and investigate any late median above 1.5× or three consecutive prompts
  above 2×.
- `prompt_sample` records cover every long prompt and identify its live phase
  and backward-pass progress. `phase_forward_enqueue_s` and
  `phase_backward_and_copy_s`, host I/O PSI, GPU utilization/P-state/clocks/power, PyTorch
  allocated/reserved/active/inactive-split memory, and physical device memory
  explain any isolated outlier. There are no OOMs, allocator retries that
  coincide with a permanent transition, non-finite Jacobians, or unexpected
  skipped prompts.
- Process and GPU memory reach a stable band rather than growing to device
  capacity. The output filesystem keeps adequate free space and chunk writes
  do not dominate prompt time.
- The partial artifact loads successfully, contains the expected finite
  matrices and fit metadata, and reports 160 fitted prompts.
- A fresh process resumes the retained checkpoint for at least one additional
  chunk without changing provenance or replaying completed prompts.

If validation fails, retain the complete telemetry and compare the primary and
cache-emptying A/B runs before changing allocator settings. Before attaching a
log to a public issue, redact the hostname, PID, SLURM job ID,
`CUDA_VISIBLE_DEVICES`, output paths/mount, and physical GPU UUIDs while
preserving consistent pseudonyms and all timings/counters. Do not describe the
slowdown as fixed; prepare a focused v0.3.x patch once the evidence identifies
a safe change. Optional Flash Linear Attention and causal-convolution kernels
may be tested in a separate compatible environment with
`--require-fast-kernels`, but that result does not replace validation of the
stock image's documented fallback.
