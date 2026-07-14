#!/bin/sh
set -eu

image="${1:?usage: smoke.sh IMAGE EXPECTED_CUDA}"
expected_cuda="${2:?usage: smoke.sh IMAGE EXPECTED_CUDA}"
tmp="$(mktemp -d)"
containers=""

cleanup() {
    for container in $containers; do
        docker rm -f "$container" >/dev/null 2>&1 || true
    done
    rm -rf "$tmp"
}
trap cleanup EXIT INT TERM

docker run --rm --entrypoint /opt/miru/bin/python "$image" -c \
    "import accelerate, bitsandbytes, pyarrow, torch, transformers, triton; assert torch.version.cuda == '$expected_cuda'; assert not torch.cuda.is_available()"
docker run --rm "$image" sh -c \
    'test "$(id -u)" = 10001 && command -v miru-tracer && command -v miru-tracer-fit-lens && command -v miru-tracer-convert-lens'
docker run --rm "$image" miru-tracer-fit-lens --help >/dev/null

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

mapping="$(docker port "$ssh_container" 22/tcp)"
ssh_port="${mapping##*:}"
ssh_opts="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes"
# shellcheck disable=SC2086
ssh $ssh_opts -i "$tmp/id_ed25519" -p "$ssh_port" root@127.0.0.1 \
    'test "$HOME" = /root && command -v miru-tracer-fit-lens'
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

ui_container="$(docker run -d -e MIRU_SSH_ENABLE=0 -p 127.0.0.1::7860 "$image")"
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
