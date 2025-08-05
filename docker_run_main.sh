#!/bin/bash

echo "Starting Streamlit Docker App..."

if [ ! -d "Output" ]; then
  echo "[INFO] 'Output' folder not found. Creating it..."
  mkdir -p "Output"
fi

docker run -it \
  -v "$(pwd)/Output:/app/Output" \
  dt-lung python main.py
