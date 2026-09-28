chcp 65001 >nul
@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
set "STOCKRL_RUNTIME_DIR=%USERPROFILE%\Desktop\모델\runtime-global-korea-live"
set "STOCKRL_WEB_PORT=8767"
set "PY_CMD=python"
python --version >nul 2>&1
if errorlevel 1 (
  py -3 --version >nul 2>&1
  if errorlevel 1 goto no_python
  set "PY_CMD=py -3"
)
%PY_CMD% -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1
if errorlevel 1 goto no_python
%PY_CMD% -c "import torch,numpy,pandas,requests,psutil,keyring,websockets" >nul 2>&1
if errorlevel 1 (
  echo Installing the required Python packages for StockRL...
  %PY_CMD% -m pip install -e .
  if errorlevel 1 goto install_failed
)
%PY_CMD% start_stockrl.py
if errorlevel 1 goto run_failed
goto end
:no_python
echo Python 3.10 or newer is required. Install Python, then double-click this launcher again.
pause
exit /b 1
:install_failed
echo Package installation failed. Check the network and Python installation.
pause
exit /b 1
:run_failed
echo StockRL stopped or failed to start. See the message above.
pause
exit /b 1
:end
endlocal
