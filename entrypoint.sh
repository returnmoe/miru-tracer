#!/bin/sh
set -eu

run_dir=/run/miru
mode_file="$run_dir/mode"
authorized_keys_source=/root/.ssh/authorized_keys
authorized_keys_runtime="$run_dir/authorized_keys"
mkdir -p "$run_dir"

die() {
    echo "miru-entrypoint: $*" >&2
    exit 1
}

detect_ssh_key_source() {
    if [ -n "${MIRU_SSH_AUTHORIZED_KEYS:-}" ]; then
        printf '%s\n' MIRU_SSH_AUTHORIZED_KEYS
    elif [ -n "${SSH_PUBLIC_KEY:-}" ]; then
        # RunPod documents this as its per-Pod key override.
        printf '%s\n' SSH_PUBLIC_KEY
    elif [ -s "$authorized_keys_source" ]; then
        printf '%s\n' /root/.ssh/authorized_keys
    elif [ -n "${PUBLIC_KEY:-}" ]; then
        printf '%s\n' PUBLIC_KEY
    else
        printf '%s\n' none
    fi
}

install_authorized_keys() {
    mkdir -p /root/.ssh
    chmod 0700 /root/.ssh
    case "$ssh_key_source" in
        MIRU_SSH_AUTHORIZED_KEYS)
            printf '%s\n' "$MIRU_SSH_AUTHORIZED_KEYS" > "$authorized_keys_runtime"
            ;;
        SSH_PUBLIC_KEY)
            printf '%s\n' "$SSH_PUBLIC_KEY" > "$authorized_keys_runtime"
            ;;
        /root/.ssh/authorized_keys)
            # Copy out of a possibly read-only bind mount so StrictModes can
            # validate predictable root ownership and permissions.
            cp "$authorized_keys_source" "$authorized_keys_runtime"
            ;;
        PUBLIC_KEY)
            printf '%s\n' "$PUBLIC_KEY" > "$authorized_keys_runtime"
            ;;
    esac
    [ -s "$authorized_keys_runtime" ] || \
        die "SSH requires a public key; configure MIRU_SSH_AUTHORIZED_KEYS, SSH_PUBLIC_KEY, PUBLIC_KEY, or /root/.ssh/authorized_keys"
    chown root:root /root/.ssh "$authorized_keys_runtime"
    chmod 0600 "$authorized_keys_runtime"
    ssh-keygen -l -f "$authorized_keys_runtime" >/dev/null 2>&1 || \
        die "root authorized_keys contains no valid public key"
}

log_ssh_host_key_fingerprints() {
    found=0
    for public_key in /etc/ssh/ssh_host_*_key.pub; do
        [ -f "$public_key" ] || continue
        fingerprint="$(ssh-keygen -l -E sha256 -f "$public_key")" || \
            die "could not read SSH host key fingerprint: $public_key"
        printf 'miru-entrypoint: SSH host key %s: %s\n' \
            "$(basename "$public_key")" "$fingerprint"
        found=1
    done
    [ "$found" -eq 1 ] || die "SSH host key generation produced no public keys"
}

configure_ssh() {
    [ "$(id -u)" -eq 0 ] || die "SSH requires the container to start as root"
    install_authorized_keys
    ssh_port="${MIRU_SSH_PORT:-22}"
    case "$ssh_port" in
        ''|*[!0-9]*) die "MIRU_SSH_PORT must be an integer from 1 to 65535" ;;
    esac
    if [ "$ssh_port" -lt 1 ] || [ "$ssh_port" -gt 65535 ]; then
        die "MIRU_SSH_PORT must be an integer from 1 to 65535"
    fi
    printf 'Port %s\n' "$ssh_port" > /etc/ssh/sshd_config.d/99-miru-port.conf
    ssh-keygen -A
    log_ssh_host_key_fingerprints
    /usr/sbin/sshd -t
}

start_ssh_background() {
    rm -f /run/miru/sshd.pid
    /usr/sbin/sshd
    attempt=0
    while [ "$attempt" -lt 50 ]; do
        if [ -s /run/miru/sshd.pid ]; then
            ssh_pid="$(cat /run/miru/sshd.pid)"
            case "$ssh_pid" in
                ''|*[!0-9]*) ;;
                *)
                    if kill -0 "$ssh_pid" 2>/dev/null; then
                        printf 'miru-entrypoint: SSH daemon started on port %s (pid %s)\n' \
                            "$ssh_port" "$ssh_pid"
                        return
                    fi
                    ;;
            esac
        fi
        attempt=$((attempt + 1))
        sleep 0.1
    done
    die "SSH daemon did not create a live PID within 5 seconds"
}

exec_as_miru() {
    HOME=/home/miru
    export HOME
    if [ "$(id -u)" -eq 0 ]; then
        exec setpriv --reuid=miru --regid=miru --init-groups -- "$@"
    fi
    exec "$@"
}

automatic=0
if [ "$#" -eq 1 ] && [ "$1" = "miru-auto" ]; then
    automatic=1
fi

ssh_enabled=0
ssh_mode="${MIRU_SSH_ENABLE:-auto}"
ssh_key_source="$(detect_ssh_key_source)"
runpod=0
if [ -n "${RUNPOD_POD_ID:-}" ] || [ -n "${RUNPOD_TCP_PORT_22:-}" ]; then
    runpod=1
fi
case "$ssh_mode" in
    1)
        if [ "$ssh_key_source" = none ] && [ "$runpod" -eq 1 ]; then
            die "RunPod supplied no account SSH key; enable SSH Terminal Access when deploying and redeploy the Pod (the account key must exist before startup)"
        fi
        ssh_enabled=1
        ;;
    0) ssh_enabled=0 ;;
    auto)
        if [ "$ssh_key_source" != none ]; then
            ssh_enabled=1
        elif [ -n "${RUNPOD_TCP_PORT_22:-}" ] && [ "$automatic" -eq 1 ]; then
            die "RunPod exposed TCP port 22 but supplied no account SSH key; enable SSH Terminal Access when deploying and redeploy the Pod (the account key must exist before startup), or set MIRU_SSH_ENABLE=0"
        fi
        ;;
    *) die "MIRU_SSH_ENABLE must be auto, 1, or 0" ;;
esac

if [ "$ssh_enabled" -eq 1 ]; then
    printf 'miru-entrypoint: SSH enabled (key source: %s)\n' "$ssh_key_source"
    configure_ssh
elif [ "$ssh_mode" = auto ] && [ "$runpod" -eq 1 ]; then
    printf '%s\n' \
        'miru-entrypoint: SSH disabled: RunPod supplied no account SSH key; enable SSH Terminal Access when deploying and redeploy the Pod (the account key must exist before startup), or set MIRU_SSH_ENABLE=0 if SSH is intentionally disabled' >&2
elif [ "$ssh_mode" = auto ]; then
    printf '%s\n' \
        'miru-entrypoint: SSH disabled: auto mode found no public key; configure a key or set MIRU_SSH_ENABLE=1 to make this fatal' >&2
else
    printf '%s\n' 'miru-entrypoint: SSH disabled by MIRU_SSH_ENABLE=0'
fi

if [ "$automatic" -eq 0 ]; then
    if [ "$ssh_enabled" -eq 1 ]; then
        start_ssh_background
        printf '%s\n' command+ssh > "$mode_file"
    else
        printf '%s\n' command > "$mode_file"
    fi
    exec_as_miru "$@"
fi

case "${MIRU_AUTO_START_UI:-1}" in
    1)
        if [ "$ssh_enabled" -eq 1 ]; then
            start_ssh_background
            printf '%s\n' ui+ssh > "$mode_file"
        else
            printf '%s\n' ui > "$mode_file"
        fi
        exec_as_miru miru-tracer
        ;;
    0)
        [ "$ssh_enabled" -eq 1 ] || \
            die "MIRU_AUTO_START_UI=0 requires SSH or an explicit command"
        printf '%s\n' ssh > "$mode_file"
        # Keep SSH as the foreground service so SSH-only cloud Pods stay alive.
        printf 'miru-entrypoint: SSH daemon starting in foreground on port %s\n' \
            "$ssh_port"
        exec /usr/sbin/sshd -D -e
        ;;
    *) die "MIRU_AUTO_START_UI must be 1 or 0" ;;
esac
