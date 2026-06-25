@echo off
REM Double-click launcher for install.ps1 (bypasses PowerShell execution policy).
REM Pass-through args, e.g.:  install.bat -Cpu   /   install.bat -NoServer
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause
