#!/bin/bash
# Build script for Render deployment

echo "🔧 Installing system dependencies..."
apt-get update
apt-get install -y build-essential

echo "🦀 Installing Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements-deploy.txt

echo "✅ Build completed successfully!"
