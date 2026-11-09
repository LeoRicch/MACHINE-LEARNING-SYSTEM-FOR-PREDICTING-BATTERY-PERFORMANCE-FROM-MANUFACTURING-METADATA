@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto setupfailed
)
".venv\Scripts\python.exe" -c "import numpy, pandas, sklearn, joblib" >nul 2>&1
if errorlevel 1 (
  echo Installing Casimir dependencies into the app environment.
  if exist "wheelhouse" (
    ".venv\Scripts\python.exe" -m pip install --no-index --find-links wheelhouse -r requirements.txt
  ) else (
    ".venv\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements.txt
  )
  if errorlevel 1 goto setupfailed
)
".venv\Scripts\python.exe" server.py
if errorlevel 1 pause
exit /b
:setupfailed
echo Setup could not complete. Check Python and your internet connection, then try again.
pause
exit /b 1
