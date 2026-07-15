#!/bin/bash
set -e

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Training model on real Open-Meteo data..."
python train_model.py

echo "Build complete. Model trained and saved."
