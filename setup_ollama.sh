#!/usr/bin/env bash

set -e  # Exit on error

echo "Updating package list and installing zstd..."
apt update && apt install -y zstd curl

echo "Installing Ollama..."
if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

echo "Setting environment variables..."
export OLLAMA_MODELS=/workspace/ollama_models
export OLLAMA_HOST=0.0.0.0:11434

mkdir -p "$OLLAMA_MODELS"

echo "Starting Ollama server..."
ollama serve > /tmp/ollama.log 2>&1 &

echo "Waiting for Ollama to start..."
until curl -s http://127.0.0.1:11434/api/tags >/dev/null; do
    sleep 2
done

pull_model_if_missing() {
    local model="$1"
    
    if ollama list | awk '{print $1}' | grep -qx "$model"; then
        echo "Model '$model' already installed. Skipping..."
    else
        echo "Pulling model '$model'..."
        ollama pull "$model"
    fi
}

echo "Checking and pulling required models..."
pull_model_if_missing "qwen3:14b"
pull_model_if_missing "llama3.2:3b"
pull_model_if_missing "nomic-embed-text"

echo "All models are ready."

wait