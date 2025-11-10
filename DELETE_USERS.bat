@echo off
echo ========================================
echo Delete Users from Face Recognition System
echo ========================================
echo.
echo WARNING: This will delete all users and training data!
echo.
pause

cd /d "%~dp0"

echo Deleting training images...
del /Q "TrainingImage\*.jpg" 2>nul

echo Clearing employee database...
echo Id,Name > "EmployeeDetails\EmployeeDetails.csv"

echo Deleting trained model...
del /Q "TrainingImageLabel\Trainer.yml" 2>nul

echo.
echo ========================================
echo All users deleted successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Run the application: python train.py
echo 2. Add new users
echo 3. Train the model
echo.
pause

