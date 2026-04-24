#!/bin/bash
set -e

# Use a patched version, not v2.5.0 (has known CVEs)
MILVUS_VERSION="2.5.10"

echo "--- Downloading Milvus .deb package ---"
wget "https://github.com/milvus-io/milvus/releases/download/v${MILVUS_VERSION}/milvus_${MILVUS_VERSION}-1_amd64.deb"

echo "--- Installing ---"
apt-get update
dpkg -i "milvus_${MILVUS_VERSION}-1_amd64.deb"
apt-get -f install -y   # resolves any missing deps

echo "--- Starting Milvus ---"
systemctl enable milvus
systemctl start milvus
systemctl status milvus
