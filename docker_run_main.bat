@echo off
SETLOCAL

echo Starting Streamlit Docker App...

IF NOT EXIST "Output" (
    echo [INFO] 'Output' folder not found. Creating it...
    mkdir "Output"
)

docker run -it ^
  -v "%cd%\Output:/app/Output" ^
  dt-lung python main.py

ENDLOCAL
