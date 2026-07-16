#!/bin/sh
set -eu

image="${1:?usage: smoke.sh IMAGE EXPECTED_CUDA}"
expected_cuda="${2:?usage: smoke.sh IMAGE EXPECTED_CUDA}"
expected_torch="2.12.1+cu$(printf '%s' "$expected_cuda" | tr -d '.')"
case "$expected_cuda" in
    12.6) required_arches='sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90' ;;
    13.0) required_arches='sm_75 sm_80 sm_86 sm_90 sm_100 sm_120' ;;
    *)
        echo "unsupported smoke-test CUDA version: $expected_cuda" >&2
        exit 1
        ;;
esac
tmp="$(mktemp -d)"
containers=""

cleanup() {
    for container in $containers; do
        docker rm -f "$container" >/dev/null 2>&1 || true
    done
    rm -rf "$tmp"
}
trap cleanup EXIT INT TERM

# torch.cuda.get_arch_list() is empty on the CPU-only CI runner, while this
# compile-time API still exposes the targets embedded in the wheel.
docker run --rm --entrypoint /opt/miru/bin/python "$image" -c \
    "import accelerate, bitsandbytes, pyarrow, torch, transformers, triton; assert str(torch.__version__) == '$expected_torch', torch.__version__; assert torch.version.cuda == '$expected_cuda', torch.version.cuda; flags = set(torch._C._cuda_getArchFlags().split()); missing = set('$required_arches'.split()) - flags; assert not missing, (missing, flags); assert not torch.cuda.is_available()"

image_entrypoint="$(docker image inspect --format '{{json .Config.Entrypoint}}' "$image")"
[ "$image_entrypoint" = \
    '["/usr/bin/tini","-s","--","/usr/local/bin/miru-entrypoint"]' ] || {
    echo "unexpected image entrypoint: $image_entrypoint" >&2
    exit 1
}
image_user="$(docker image inspect --format '{{.Config.User}}' "$image")"
[ "$image_user" = root ] || {
    echo "unexpected image bootstrap user: $image_user" >&2
    exit 1
}
docker run --rm --entrypoint sh "$image" -c \
    'test "$(id -u)" = 0 && test "$HOME" = /root && tmux -V >/dev/null'
docker run --rm --entrypoint sh "$image" -c \
    '/usr/bin/tini -s -- true; status=$?; :; exit "$status"' \
    > "$tmp/nested-tini-log" 2>&1
if grep -F 'Tini is not running as PID 1' "$tmp/nested-tini-log"; then
    echo "nested Tini did not register as a child subreaper" >&2
    exit 1
fi

docker run --rm -e MIRU_SSH_ENABLE=0 "$image" python -c \
    'from miru_tracer.config import Settings; assert Settings.from_env().server_name == "127.0.0.1"'
docker run --rm "$image" sh -c \
    'test "$(id -u)" = 10001 && test "$HOME" = /home/miru && tmux -V >/dev/null && command -v miru-tracer && command -v miru-tracer-fit-lens && command -v miru-tracer-convert-lens'
docker run --rm "$image" miru-tracer-fit-lens --help >/dev/null

docker run --rm "$image" true > "$tmp/no-key-log" 2>&1
grep -Fqx \
    'miru-entrypoint: SSH disabled: auto mode found no public key; configure a key or set MIRU_SSH_ENABLE=1 to make this fatal' \
    "$tmp/no-key-log"
if docker run --rm -e RUNPOD_TCP_PORT_22=32022 "$image" \
    > "$tmp/runpod-no-key-log" 2>&1; then
    echo "container accepted a RunPod SSH mapping without a public key" >&2
    exit 1
fi
grep -Fqx \
    'miru-entrypoint: RunPod exposed TCP port 22 but supplied no account SSH key; enable SSH Terminal Access when deploying and redeploy the Pod (the account key must exist before startup), or set MIRU_SSH_ENABLE=0' \
    "$tmp/runpod-no-key-log"
docker run --rm -e RUNPOD_TCP_PORT_22=32022 "$image" true \
    > "$tmp/runpod-explicit-command-log" 2>&1
grep -Fqx \
    'miru-entrypoint: SSH disabled: RunPod supplied no account SSH key; enable SSH Terminal Access when deploying and redeploy the Pod (the account key must exist before startup), or set MIRU_SSH_ENABLE=0 if SSH is intentionally disabled' \
    "$tmp/runpod-explicit-command-log"
if docker run --rm -e RUNPOD_POD_ID=test -e MIRU_SSH_ENABLE=1 "$image" true \
    > "$tmp/runpod-required-key-log" 2>&1; then
    echo "container accepted required RunPod SSH without an account key" >&2
    exit 1
fi
grep -Fqx \
    'miru-entrypoint: RunPod supplied no account SSH key; enable SSH Terminal Access when deploying and redeploy the Pod (the account key must exist before startup)' \
    "$tmp/runpod-required-key-log"

docker run --rm --entrypoint sh "$image" -c \
    'ssh-keygen -A >/dev/null && exec /usr/sbin/sshd -T' > "$tmp/sshd-effective"
grep -qx 'permitrootlogin without-password' "$tmp/sshd-effective"
grep -qx 'authenticationmethods publickey' "$tmp/sshd-effective"
grep -qx 'passwordauthentication no' "$tmp/sshd-effective"
grep -qx 'kbdinteractiveauthentication no' "$tmp/sshd-effective"
grep -qx 'allowtcpforwarding local' "$tmp/sshd-effective"
grep -qx 'allowagentforwarding no' "$tmp/sshd-effective"
grep -qx 'x11forwarding no' "$tmp/sshd-effective"
grep -qx 'allowusers root' "$tmp/sshd-effective"

ssh-keygen -q -t ed25519 -N '' -f "$tmp/id_ed25519"
docker run --rm \
    --mount "type=bind,source=$tmp/id_ed25519.pub,target=/root/.ssh/authorized_keys,readonly" \
    "$image" true > "$tmp/account-key-file-log" 2>&1
grep -Fqx 'miru-entrypoint: SSH enabled (key source: /root/.ssh/authorized_keys)' \
    "$tmp/account-key-file-log"

ssh_container="$(docker run -d \
    -e MIRU_AUTO_START_UI=0 \
    -e PUBLIC_KEY="$(cat "$tmp/id_ed25519.pub")" \
    -p 127.0.0.1::22 \
    "$image")"
containers="$containers $ssh_container"

i=0
until docker exec "$ssh_container" /usr/local/bin/miru-healthcheck; do
    i=$((i + 1))
    [ "$i" -lt 30 ] || {
        docker logs "$ssh_container"
        exit 1
    }
    sleep 1
done

host_fingerprint="$(docker exec "$ssh_container" ssh-keygen -l -E sha256 \
    -f /etc/ssh/ssh_host_ed25519_key.pub)"
docker logs "$ssh_container" 2>&1 | grep -Fqx \
    "miru-entrypoint: SSH host key ssh_host_ed25519_key.pub: $host_fingerprint"

mapping="$(docker port "$ssh_container" 22/tcp)"
ssh_port="${mapping##*:}"
ssh_opts="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes"
# shellcheck disable=SC2086
ssh $ssh_opts -i "$tmp/id_ed25519" -p "$ssh_port" root@127.0.0.1 \
    'test "$HOME" = /root && tmux -V >/dev/null && command -v miru-tracer-fit-lens'
if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o BatchMode=yes -o PubkeyAuthentication=no \
    -o PreferredAuthentications=password \
    -p "$ssh_port" root@127.0.0.1 true 2>/dev/null; then
    echo "password-only SSH unexpectedly succeeded" >&2
    exit 1
fi

if docker run --rm -e MIRU_AUTO_START_UI=0 -e MIRU_SSH_ENABLE=0 "$image"; then
    echo "service-less container unexpectedly succeeded" >&2
    exit 1
fi
if docker run --rm -e MIRU_AUTO_START_UI=0 -e PUBLIC_KEY=invalid "$image"; then
    echo "container accepted an invalid SSH public key" >&2
    exit 1
fi
if docker run --rm -e MIRU_AUTO_START_UI=0 -e MIRU_SSH_ENABLE=1 \
    -e PUBLIC_KEY="$(cat "$tmp/id_ed25519.pub")" -e MIRU_SSH_PORT=70000 "$image"; then
    echo "container accepted an invalid SSH port" >&2
    exit 1
fi

ui_container="$(docker run -d \
    -e SSH_PUBLIC_KEY="$(cat "$tmp/id_ed25519.pub")" \
    -p 127.0.0.1::22 \
    "$image")"
containers="$containers $ui_container"
i=0
until docker exec "$ui_container" /usr/local/bin/miru-healthcheck; do
    i=$((i + 1))
    [ "$i" -lt 60 ] || {
        docker logs "$ui_container"
        exit 1
    }
    sleep 1
done
test "$(docker exec "$ui_container" cat /run/miru/mode)" = ui+ssh
docker logs "$ui_container" > "$tmp/ui-log" 2>&1
grep -Fqx 'miru-entrypoint: SSH enabled (key source: SSH_PUBLIC_KEY)' \
    "$tmp/ui-log"
grep -F 'Server configuration: host=127.0.0.1, port=7860' "$tmp/ui-log" >/dev/null

ui_ssh_mapping="$(docker port "$ui_container" 22/tcp)"
ui_ssh_port="${ui_ssh_mapping##*:}"
# shellcheck disable=SC2086
ssh $ssh_opts -i "$tmp/id_ed25519" -p "$ui_ssh_port" root@127.0.0.1 \
    'test "$HOME" = /root && command -v miru-tracer'
