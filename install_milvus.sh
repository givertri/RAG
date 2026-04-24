#!/bin/bash
set -e  # Stop script on any error

# 1. Setup Directories
export MILVUS_DIR="/workspace/milvus"
export MILVUS_VERSION="v2.5.0"  # Or v2.4.4
mkdir -p $MILVUS_DIR/bin $MILVUS_DIR/data $MILVUS_DIR/configs
cd $MILVUS_DIR

# 2. Download Milvus Binary (The asset name is usually milvus-linux-amd64.tar.gz)
echo "--- Downloading Milvus Binary ($MILVUS_VERSION) ---"
# Note the updated filename here:
WGET_URL="https://github.com/milvus-io/milvus/releases/download/${MILVUS_VERSION}/milvus-linux-amd64.tar.gz"

if ! wget $WGET_URL; then
    echo "ERROR: Download failed. Please verify the filename at: https://github.com/milvus-io/milvus/releases/tag/${MILVUS_VERSION}"
    exit 1
fi

tar -zxvf milvus-linux-amd64.tar.gz -C bin/
rm milvus-linux-amd64.tar.gz

# 3. Download and Modify Config
echo "--- Configuring Milvus ---"
wget https://raw.githubusercontent.com/milvus-io/milvus/master/configs/milvus.yaml -O configs/milvus.yaml

# Ensure local persistence for data and embedded etcd
sed -i "s|rootPath:.*|rootPath: $MILVUS_DIR/data/milvus|g" configs/milvus.yaml
sed -i "s|path:.*etcd|path: $MILVUS_DIR/data/etcd|g" configs/milvus.yaml

# 4. Create Start/Stop helpers (fixing the binary path)
cat <<EOF > start_milvus.sh
#!/bin/bash
export LD_LIBRARY_PATH=$MILVUS_DIR/bin/lib:\$LD_LIBRARY_PATH
# In v2.x, the binary is just 'milvus', but we tell it to 'run standalone'
nohup $MILVUS_DIR/bin/milvus run standalone --config $MILVUS_DIR/configs/milvus.yaml > $MILVUS_DIR/milvus.log 2>&1 &
echo "Milvus starting... check milvus.log for status."
EOF

chmod +x start_milvus.sh
echo "------------------------------------------------"
echo "Installation complete for real this time!"
