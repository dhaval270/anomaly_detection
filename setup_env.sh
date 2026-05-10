#!/usr/bin/env bash
# Create virtual environment and install dependencies (Linux/Mac).
# Usage: bash setup_env.sh

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Run experiments with: python run_experiments.py"
