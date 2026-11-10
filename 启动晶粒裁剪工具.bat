@echo off
setlocal
cd /d "%~dp0"
python "%~dp0grain_crop_tool.py" %*
if errorlevel 1 pause
endlocal
