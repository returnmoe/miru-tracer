#!/bin/bash
set -e

echo "=== Miru Tracer Container Starting ==="

# Check if SSH should be enabled
if [ "$MIRU_ENABLE_SSH" = "1" ]; then
    echo "SSH server is ENABLED (MIRU_ENABLE_SSH=1)"

    # Generate SSH host keys if they don't exist
    echo "Generating SSH host keys..."
    ssh-keygen -A

    # Ensure authorized_keys exists and has proper permissions
    if [ ! -f /root/.ssh/authorized_keys ]; then
        echo "WARNING: /root/.ssh/authorized_keys not found!"
        echo "Please mount your SSH public key with: -v ~/.ssh/authorized_keys:/root/.ssh/authorized_keys:ro"
        echo "SSH server will start, but you won't be able to login without authorized keys."
    else
        chmod 600 /root/.ssh/authorized_keys
        echo "Authorized keys loaded from /root/.ssh/authorized_keys"
    fi

    # Start SSH daemon
    echo "Starting SSH daemon..."
    /usr/sbin/sshd

    # Print SSH key fingerprints for client verification
    echo ""
    echo "=== SSH Host Key Fingerprints ==="
    for keyfile in /etc/ssh/ssh_host_*_key.pub; do
        if [ -f "$keyfile" ]; then
            ssh-keygen -lf "$keyfile"
        fi
    done
    echo "================================="
    echo ""
else
    echo "SSH server is DISABLED (set MIRU_ENABLE_SSH=1 to enable)"
fi

# Start the application
echo "Starting Miru Tracer application..."
cd /app
exec python3 app.py
