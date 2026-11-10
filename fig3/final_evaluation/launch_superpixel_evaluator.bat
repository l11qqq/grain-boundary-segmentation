@echo off
setlocal
cd /d "%~dp0"
python "superpixel_manual_evaluator.py" %*
if errorlevel 1 pause
endlocal
