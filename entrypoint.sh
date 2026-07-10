#!/bin/sh
set -eu

if [ "${MIRU_SSH_ENABLE:-0}" = "1" ]; then
    if [ "$(id -u)" -ne 0 ]; then
        echo "MIRU_SSH_ENABLE=1 requires the container to start as root" >&2
        exit 1
    fi
    ssh-keygen -A
    if [ -n "${MIRU_SSH_AUTHORIZED_KEYS:-}" ]; then
        printf '%s\n' "$MIRU_SSH_AUTHORIZED_KEYS" > /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
    fi
    if [ ! -s /root/.ssh/authorized_keys ]; then
        echo "Refusing to start SSH without MIRU_SSH_AUTHORIZED_KEYS" >&2
        exit 1
    fi
    port="${MIRU_SSH_PORT:-22}"
    case "$port" in *[!0-9]*|'') echo "Invalid MIRU_SSH_PORT" >&2; exit 1;; esac
    printf 'Port %s\n' "$port" > /etc/ssh/sshd_config.d/miru-port.conf
    /usr/sbin/sshd
fi

if [ "$(id -u)" -eq 0 ]; then
    exec setpriv --reuid=miru --regid=miru --init-groups -- "$@"
fi
exec "$@"
