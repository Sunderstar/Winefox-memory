@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py main.py ui
) else (
  python main.py ui
)
echo.
pause
