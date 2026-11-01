@echo off
rem Starts the audiospace localization app (headphones required).
rem Install first: python -m pip install -e ".[app]"  (see manual chapter 3)
cd /d "%~dp0"
rem Clear the silent switch used by the automated tests, for this run only
set AUDIOSPACE_SILENT=
rem Use the same Python that the install step used: python first, then py.
rem (The Microsoft Store placeholder named python fails this check.)
python -c "import sys" >nul 2>nul
if %errorlevel%==0 (
    python -X utf8 demo\localization_app.py
) else (
    py -X utf8 demo\localization_app.py
)
rem Keep the window open only when something went wrong
if errorlevel 1 pause
