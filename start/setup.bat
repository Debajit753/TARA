@echo off
REM Rebuild the TARA environment on a Windows laptop. Needs internet once.
REM Usage:  setup.bat
REM This script lives in start\, but requirements.txt and the app are one level
REM up — cd to the project root, not to start\.
cd /d "%~dp0.."
echo Creating virtual environment (.venv) ...
python -m venv .venv
if errorlevel 1 (echo SETUP FAILED: could not create .venv & exit /b 1)
.venv\Scripts\python -m pip install --upgrade pip
echo Installing dependencies from requirements.txt (this can take a few minutes) ...
.venv\Scripts\pip install -r requirements.txt
if errorlevel 1 (echo SETUP FAILED: dependency install did not complete & exit /b 1)
echo.
echo Done. Test it with:   .venv\Scripts\python quickstart.py
echo Activate manually:    .venv\Scripts\activate
