@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
py -3.13 "%~dp0scripts\install_windows.py" %*
if errorlevel 1 (
  echo Installation failed. Python 3.13 is required. See the error above.
  pause
  exit /b 1
)
echo Installation complete. Run the server launcher.
pause
