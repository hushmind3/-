@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if exist "%~dp0artifacts\experts\venv\Scripts\python.exe" (
  "%~dp0artifacts\experts\venv\Scripts\python.exe" "%~dp0start_stockrl.py" %*
) else (
  py -3 "%~dp0start_stockrl.py" %*
)
if errorlevel 1 (
  echo Server startup failed. See runtime\markets\korea\web.stderr.log
  pause
  exit /b 1
)
