# Miru Tracer on RunPod

Miru Tracer is an experimental Gradio app for exploring LLM generation one
token at a time. Inspect probabilities, choose alternative tokens, record
generation logs, visualize layer readouts, and experiment with interventions.

> **Security:** This template is configured for access through SSH forwarding.
> Do not expose Miru's Gradio port through RunPod's HTTP proxy or a public TCP
> mapping. The UI is unauthenticated and its model loader can load and execute
> code. The recommended access method is the SSH tunnel described below.

> **Known RunPod web-launch limitation (verified July 16, 2026):** The RunPod
> web interface can expose `22/tcp` for this custom/community image while
> launching it with `startSsh: false`. In that state, the Pod has a mapped SSH
> port but RunPod supplies no account public key, so the image cannot start
> secure SSH. The same template launched through `runpodctl` with `--ssh=true`
> received the account key and SSH worked.

## Template behavior

- Provides CUDA, PyTorch, quantization support, SSH, and `tmux`.
- Exposes port 22 and starts key-only SSH **only when RunPod actually injects**
  a public key registered to your account.
- Keeps the Miru UI private on `127.0.0.1:7860`.

## Before deploying

1. Register an SSH public key in your RunPod account **before** deployment.
2. Select a host supporting CUDA 13.0. If available, set the **CUDA Versions**
   filter to `13.0`.
3. Allocate at least 20 GB of container disk, and more for large models.
4. Optionally attach a RunPod network volume at `/workspace` if files must
   survive Pod replacement.

Do not use the presence of port 22, an `ssh` port label, or
`RUNPOD_TCP_PORT_22` as evidence that the key was injected. They describe the
network mapping only.

For gated Hugging Face models, provide `HF_TOKEN` using a
[RunPod secret](https://docs.runpod.io/pods/templates/environment-variables#using-secrets).
Never store tokens or private keys directly in a public template.

## Deploy with SSH key injection

The reliable path observed during testing is a per-Pod GPU launch through
[`runpodctl`](https://github.com/runpod/runpodctl/releases/tag/v2.7.1):

```bash
runpodctl pod create \
  --template-id TEMPLATE_ID \
  --gpu-id "GPU_ID_FROM_RUNPODCTL_GPU_LIST" \
  --ports 22/tcp \
  --ssh=true \
  --min-cuda-version 13.0
```

The account key must exist before this command runs. The `--ssh=true` option
sets `startSsh: true` for that Pod launch; it is not just a port switch.

This cannot currently be made persistent with `runpodctl template update`:
version 2.7.1 exposes no `--ssh` or `startSsh` template-update option. Adding
`22/tcp` or an `ssh` port label therefore does not repair web launches. Until
RunPod fixes its web/template control path, a launch from the web interface
may still omit the account key.

If RunPod instructions point to `cli.runpod.io` and it does not resolve, use
the [official `runpodctl` GitHub releases](https://github.com/runpod/runpodctl/releases)
instead. At the time this was verified, the maintained
[`runpodctl` repository](https://github.com/runpod/runpodctl) referenced
`cli.runpod.net`, not `cli.runpod.io`.

## Connect with SSH forwarding

Wait for the Pod to become healthy, then open its **Connect** panel to find the
SSH host and mapped port. RunPod maps container port 22 to a dynamic public
port, so always use the current values shown there.

Before connecting, open the Pod's **Logs** view. At startup, the container
prints SHA-256 fingerprints for its SSH host keys, including:

```text
miru-entrypoint: SSH host key ssh_host_ed25519_key.pub: 256 SHA256:<fingerprint> ...
```

Connect from your computer while forwarding local port 7860 to Miru:

```bash
ssh -L 7860:127.0.0.1:7860 \
  root@<pod-public-ip> -p <mapped-port> -i ~/.ssh/id_ed25519
```

On the first connection, accept the host key only if the `SHA256:...` value
shown by your SSH client exactly matches the same key type in the trusted
RunPod container logs. `ssh-keyscan` over the same network path is not an
independent verification source.

Inside the Pod, start Miru in `tmux`:

```bash
tmux new -s miru
miru-tracer
```

Keep the SSH connection open and visit
[http://127.0.0.1:7860](http://127.0.0.1:7860). Detach from `tmux` with
`Ctrl-b`, then `d`; reattach with `tmux attach -t miru`. If SSH disconnects,
Miru continues in `tmux`; reconnect with the same forwarding command.

## First steps

1. Open **Model Loader** and load `Qwen/Qwen3-0.6B` as a small first test.
2. Open **Interactive Mode**, enter a prompt, and click **Initialize**.
3. Use **Next Step** to inspect or replace each generated token.

Use **Logging Mode** to generate automatically while recording probabilities,
then inspect exported JSON in **Log Analysis**. For out-of-memory errors, use a
smaller model, enable 4-bit/8-bit quantization, or select a GPU with more VRAM.
The first model load may take several minutes while files download.

## Storage and lens fitting

The Hugging Face cache is kept on the fast, instance-local container disk at
`/tmp/huggingface`. Downloads disappear when the Pod is deleted. If you attach
a RunPod network volume at `/workspace`, use it for artifacts that must survive
Pod replacement.

The logit lens works without a fit. To fit a model-specific Jacobian lens:

```bash
tmux new -s lens-fit
miru-tracer-fit-lens Qwen/Qwen3-0.6B \
  --out /workspace/lenses/Qwen--Qwen3-0.6B/lens.safetensors \
  --hf-home /tmp/huggingface --dim-batch 32
```

The fitter checkpoints beside the output. Reduce `--dim-batch` if it runs out
of VRAM. See the
[lens tutorial](https://github.com/returnmoe/miru-tracer/blob/master/docs/lens-tutorial.md)
for fitting controls and supported architectures.

## Troubleshooting

- **Pod repeatedly exits with “supplied no account SSH key”:** With
  `MIRU_SSH_ENABLE=1`, a missing key is deliberately fatal and RunPod may
  restart the container repeatedly. Confirm the account key existed before
  deployment and launch the Pod with `runpodctl ... --ssh=true`. Setting
  `MIRU_SSH_ENABLE=1` does not itself request key injection.
- **Port 22 exists but SSH does not:** The mapping and account-key injection
  are separate. `RUNPOD_TCP_PORT_22` does not mean `startSsh` was enabled.
  This is the known web-launch failure described above.
- **SSH still fails after a CLI launch:** Copy the latest host and mapped port
  from **Connect**; they can change after a reset. Check that your identity
  file matches your account key.
- **`PUBLIC_KEY` is absent inside a working SSH session:** This is expected for
  images that consume the bootstrap variable and omit it from later login
  environments. The durable result is `/root/.ssh/authorized_keys`.
- **Browser fails:** Confirm `miru-tracer` and the forwarding SSH connection are
  running. With the recommended setup, use `http://127.0.0.1:7860`, not a
  RunPod proxy URL.
- **Model fails:** Check the `tmux` output, available VRAM, disk space, and gated
  model access. Keep `MIRU_ALLOW_REMOTE_CODE=0` unless you reviewed and pinned
  the model repository.

[Source and documentation](https://github.com/returnmoe/miru-tracer) ·
[Report an issue](https://github.com/returnmoe/miru-tracer/issues) ·
[RunPod SSH guide](https://docs.runpod.io/pods/configuration/use-ssh)

Miru Tracer is community software, not maintained or supported by RunPod.
