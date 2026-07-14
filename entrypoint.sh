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

has_key_material() {
    [ -n "${MIRU_SSH_AUTHORIZED_KEYS:-}" ] || \
        [ -s "$authorized_keys_source" ] || \
        [ -n "${PUBLIC_KEY:-}" ]
}

install_authorized_keys() {
    mkdir -p /root/.ssh
    chmod 0700 /root/.ssh
    if [ -n "${MIRU_SSH_AUTHORIZED_KEYS:-}" ]; then
        printf '%s\n' "$MIRU_SSH_AUTHORIZED_KEYS" > "$authorized_keys_runtime"
    elif [ -s "$authorized_keys_source" ]; then
        # Copy out of a possibly read-only bind mount so StrictModes can
        # validate predictable root ownership and permissions.
        cp "$authorized_keys_source" "$authorized_keys_runtime"
    elif [ -n "${PUBLIC_KEY:-}" ]; then
        printf '%s\n' "$PUBLIC_KEY" > "$authorized_keys_runtime"
    fi
    [ -s "$authorized_keys_runtime" ] || die "SSH requires a root public key"
    chown root:root /root/.ssh "$authorized_keys_runtime"
    chmod 0600 "$authorized_keys_runtime"
    ssh-keygen -l -f "$authorized_keys_runtime" >/dev/null 2>&1 || \
        die "root authorized_keys contains no valid public key"
}

configure_ssh() {
    [ "$(id -u)" -eq 0 ] || die "SSH requires the container to start as root"
    install_authorized_keys
    port="${MIRU_SSH_PORT:-22}"
    case "$port" in
        ''|*[!0-9]*) die "MIRU_SSH_PORT must be an integer from 1 to 65535" ;;
    esac
    if [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
        die "MIRU_SSH_PORT must be an integer from 1 to 65535"
    fi
    printf 'Port %s\n' "$port" > /etc/ssh/sshd_config.d/99-miru-port.conf
    ssh-keygen -A
    /usr/sbin/sshd -t
}

start_ssh_background() {
    /usr/sbin/sshd
}

exec_as_miru() {
    if [ "$(id -u)" -eq 0 ]; then
        exec setpriv --reuid=miru --regid=miru --init-groups -- "$@"
    fi
    exec "$@"
}

ssh_enabled=0
case "${MIRU_SSH_ENABLE:-auto}" in
    1) ssh_enabled=1 ;;
    0) ssh_enabled=0 ;;
    auto)
        if has_key_material; then
            ssh_enabled=1
        fi
        ;;
    *) die "MIRU_SSH_ENABLE must be auto, 1, or 0" ;;
esac

if [ "$ssh_enabled" -eq 1 ]; then
    configure_ssh
fi

automatic=0
if [ "$#" -eq 1 ] && [ "$1" = "miru-auto" ]; then
    automatic=1
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
        exec /usr/sbin/sshd -D -e
        ;;
    *) die "MIRU_AUTO_START_UI must be 1 or 0" ;;
esac
