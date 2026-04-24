#!/bin/bash
set -e

MILVUS_VERSION="2.6.9"

echo "--- Downloading Milvus ${MILVUS_VERSION} .deb ---"
wget "https://github.com/milvus-io/milvus/releases/download/v${MILVUS_VERSION}/milvus_${MILVUS_VERSION}-1_amd64.deb"

echo "--- Installing ---"
apt-get update
dpkg -i "milvus_${MILVUS_VERSION}-1_amd64.deb" || apt-get -f install -y

echo "--- Starting Milvus ---"
# Try systemd first, fall back to direct binary
if systemctl is-system-running 2>/dev/null; then
    systemctl enable milvus
    systemctl start milvus
    systemctl status milvus
else
    # RunPod containers often don't have systemd
    export LD_LIBRARY_PATH=/usr/lib/milvus:$LD_LIBRARY_PATH
    nohup /usr/bin/milvus run standalone > /var/log/milvus.log 2>&1 &
    echo "Milvus starting (PID $!), check /var/log/milvus.log"
    sleep 8
    curl -sf http://localhost:9091/healthz && echo "Milvus is healthy!" || echo "Still starting, check logs"
fi
