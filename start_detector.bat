@echo off
REM Start the YOLO detector server using the project's .venv.
REM Forwards any extra args, e.g.:
REM   start_detector.bat --model yolo11n.pt
REM   start_detector.bat --engines yolo,yolopv2 --yolopv2-model models\yolopv2.pt
set "VENVPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENVPY%" (
    echo .venv not found - run install.bat first.
    pause
    exit /b 1
)
"%VENVPY%" "%~dp0detector\yolo_server.py" %*
pause
