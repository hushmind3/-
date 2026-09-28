@echo off
setlocal
cd /d "%~dp0\.."
python -m stockrl web %*
if errorlevel 1 pause
