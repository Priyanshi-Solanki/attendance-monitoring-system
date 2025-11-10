@echo off
echo ========================================
echo Face Recognition Attendance System
echo ========================================
echo.
echo Starting application...
echo.

cd /d "%~dp0"
python train.py

pause

