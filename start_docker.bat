@echo off
SETLOCAL

echo Starting Streamlit Docker App...

IF NOT EXIST "Model" (
    echo [ERROR] 'Model' folder not found! Please download the Model folder and place it in the current directory.
    pause
    exit /b 1
)

IF NOT EXIST "Output" (
    echo [INFO] 'Output' folder not found. Creating it...
    mkdir "Output"
)

docker run -p 8501:8501 ^
  -v "%cd%\Model:/app/Model" ^
  -v "%cd%\Output:/app/Output" ^
  evlp-dt-streamlit-app

ENDLOCAL
