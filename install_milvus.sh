#!/bin/bash

# 1. Setup Directories in /workspace for persistence
export MILVUS_DIR="/workspace/milvus"
mkdir -p $MILVUS_DIR/bin
mkdir -p $MILVUS_DIR/data
mkdir -p $MILVUS_DIR/configs
cd $MILVUS_DIR

# 2. Download Milvus Standalone Binary (v2.5.x for stability)
# 2. Download Milvus Binary
echo "--- Downloading Milvus Binary ---"
# Note: Ensure the version and filename match an actual release asset on GitHub
# As of v2.5.x, the naming usually follows this pattern:
export MILVUS_VERSION="v2.5.0"
wget https://github.com/milvus-io/milvus/releases/download/${MILVUS_VERSION}/milvus-linux-amd64.tar.gz

if [ $? -ne 0 ]; then
    echo "Error: Download failed. Please check if version $MILVUS_VERSION exists at the URL."
    exit 1
fi

tar -zxvf milvus-linux-amd64.tar.gz -C bin/
rm milvus-linux-amd64.tar.gz

# 3. Download Default Config
echo "--- Configuring Milvus ---"
wget https://raw.githubusercontent.com/milvus-io/milvus/master/configs/milvus.yaml -O configs/milvus.yaml

# 4. Modify config for local persistence and embedded ETCD
# We point everything to /workspace so data isn't lost on restart
sed -i "s|rootPath:.*|rootPath: /workspace/milvus/data/milvus|g" configs/milvus.yaml
sed -i "s|path:.*etcd|path: /workspace/milvus/data/etcd|g" configs/milvus.yaml

# 5. Create Start and Stop helpers
cat <<EOF > start_milvus.sh
#!/bin/bash
export LD_LIBRARY_PATH=$MILVUS_DIR/bin/lib:\$LD_LIBRARY_PATH
nohup $MILVUS_DIR/bin/milvus run standalone --config $MILVUS_DIR/configs/milvus.yaml > $MILVUS_DIR/milvus.log 2>&1 &
echo "Milvus starting in background... check milvus.log for status."
EOF

cat <<EOF > stop_milvus.sh
#!/bin/bash
pkill -f milvus
echo "Milvus stopped."
EOF

chmod +x start_milvus.sh stop_milvus.sh

echo "------------------------------------------------"
echo "Installation complete!"
echo "To start Milvus: ./start_milvus.sh"
echo "To stop Milvus:  ./stop_milvus.sh"
echo "Logs are located at: $MILVUS_DIR/milvus.log"
echo "------------------------------------------------"
