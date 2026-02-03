@echo off
setlocal
call :kill_port 8000
call :kill_port 5173
call :kill_port 5174
call :kill_port 5175
echo Servers stopped
exit /b 0

:kill_port
set PORT=%1
for /f "tokens=5" %%P in ('netstat -ano ^| findstr :%PORT% ^| findstr LISTENING') do (
  taskkill /F /PID %%P >nul 2>&1
)
exit /b 0
