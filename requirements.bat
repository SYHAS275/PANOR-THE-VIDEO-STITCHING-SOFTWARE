@echo off
echo ============================================
echo  PANOR - Installing Dependencies
echo ============================================
echo.

echo [1/2] Installing Backend Dependencies...
echo --------------------------------------------
cd backend
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Backend dependency installation failed.
    pause
    exit /b 1
)
cd ..
echo Backend dependencies installed.
echo.

echo [2/2] Installing Frontend Dependencies...
echo --------------------------------------------
cd frontend
npm install
if %errorlevel% neq 0 (
    echo ERROR: Frontend dependency installation failed.
    pause
    exit /b 1
)
cd ..
echo Frontend dependencies installed.
echo.

echo ============================================
echo  All dependencies installed successfully!
echo  Run 'run.bat' to start the application.
echo ============================================
pause
