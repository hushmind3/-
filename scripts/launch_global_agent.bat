@echo off
setlocal
cd /d "%~dp0\.."
python -m stockrl desktop %*
if errorlevel 1 pause
