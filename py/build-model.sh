#!/bin/bash

MODEL_OUTPUT_FOLDER=".."

echo "Building model in folder: $MODEL_OUTPUT_FOLDER"
echo "Setting up Python 3.11 virtual environment..."

if [[ ! -d ".venv" ]]; then
  python3.11 -m venv .venv
  if [[ $? -ne 0 ]]; then
    echo "Failed to create virtual environment."
    exit 1
  fi
  echo "Virtual environment created. Installing dependencies..."
  source .venv/bin/activate
  pip install -r requirements.txt
else
  echo "Virtual environment already exists. Activating..."
  source .venv/bin/activate
fi

set -x
python train.py