#!/bin/bash

echo "Starting Streamlit Docker App..."

if [ ! -d "Model" ]; then
  echo "[ERROR] 'Model' folder not found! Please download the Model folder and place it in the current directory."
  exit 1
fi

if [ ! -d "Output" ]; then
  echo "[INFO] 'Output' folder not found. Creating it..."
  mkdir -p "Output"
fi

docker run -p 8501:8501 \
  -v "$(pwd)/Model:/app/Model" \
  -v "$(pwd)/Output:/app/Output" \
  evlp-dt-streamlit-app
