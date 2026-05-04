#!/bin/bash

echo "--- Downloading Ollama ---"
curl -fsSL https://ollama.com/install.sh | sh

echo "--- Setting Persistence in .bashrc ---"
# This ensures the variable survives across all future terminal sessions
echo 'export OLLAMA_MODELS=/workspace/ollama_models' >> ~/.bashrc
export OLLAMA_MODELS=/workspace/ollama_models
mkdir -p /workspace/ollama_models

echo "--- Starting Ollama Server ---"
# Start Ollama and redirect logs so they don't clutter your terminal
nohup ollama serve > /workspace/ollama.log 2>&1 &

echo "--- Waiting for Ollama to wake up ---"
# This loop checks if the server is responding before trying to pull
while ! ollama list >/dev/null 2>&1; do
    echo "Still waiting for server..."
    sleep 2
done

echo "--- Pulling models (this may take a while) ---"
ollama pull qwen3:14b
ollama pull llama3.2:3b
ollama pull nomic-embed-text

echo "--- Setup Complete! ---"
