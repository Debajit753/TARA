@echo off
REM Rebuild the TARA environment on a Windows laptop. Needs internet once.
REM Usage:  setup.bat
cd /d "%~dp0"
echo Creating virtual environment (.venv) ...
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
echo Installing dependencies from requirements.txt (this can take a few minutes) ...
.venv\Scripts\pip install -r requirements.txt
echo.
echo Done. Test it with:   .venv\Scripts\python quickstart.py
echo Activate manually:    .venv\Scripts\activate
