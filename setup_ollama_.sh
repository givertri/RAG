#!/usr/bin/env bash

set -e  # Exit on error

echo "Updating package list and installing zstd..."
apt update && apt install -y zstd

echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo "Setting environment variables..."
export OLLAMA_MODELS=/workspace/ollama_models
export OLLAMA_HOST=0.0.0.0:11434

echo "Starting Ollama server..."
ollama serve
