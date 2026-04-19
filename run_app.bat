@echo off
REM ==========================================================
REM  Inspiration Illustration Finder
REM  ------------------------------------------------
REM  Streamlit app: paste an idea, get 10 ranked illustration
REM  suggestions from the Notion archive, one per visual type.
REM  Uses metaphor enrichment (claude -p) + bge-large embeddings.
REM ==========================================================
title Inspiration Illustration Finder - app
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo   py -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Inspiration Illustration Finder
echo   http://localhost:8501
echo ============================================================
echo.

".venv\Scripts\python.exe" -m streamlit run "app\app.py" --browser.gatherUsageStats=false
pause
