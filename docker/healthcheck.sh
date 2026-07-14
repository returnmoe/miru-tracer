#!/bin/sh
set -eu

mode_file=/run/miru/mode
pid_file=/run/miru/sshd.pid

check_ui() {
    port="${MIRU_SERVER_PORT:-7860}"
    curl --fail --silent --show-error --max-time 4 \
        "http://127.0.0.1:${port}/" >/dev/null
}

check_ssh() {
    [ -s "$pid_file" ]
    pid="$(cat "$pid_file")"
    case "$pid" in
        ''|*[!0-9]*) return 1 ;;
    esac
    kill -0 "$pid"
    /usr/sbin/sshd -t
}

[ -r "$mode_file" ] || exit 1
mode="$(cat "$mode_file")"
case "$mode" in
    ui) check_ui ;;
    ui+ssh) check_ui && check_ssh ;;
    ssh|command+ssh) check_ssh ;;
    command) exit 0 ;;
    *) exit 1 ;;
esac
