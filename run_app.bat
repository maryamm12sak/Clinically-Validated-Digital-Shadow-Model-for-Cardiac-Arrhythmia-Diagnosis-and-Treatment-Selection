@echo off
setlocal EnableDelayedExpansion
rem ECG decision-support app: one-click start on Windows.
rem First run: creates a Python environment in .venv and installs packages (needs internet, several minutes).
rem Starts the backend (http://127.0.0.1:8000) and the website (http://localhost:5173) in two windows.
rem To stop the app, close both windows.

cd /d "%~dp0"
set "REPO=%~dp0"
set "VENV=%REPO%.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\.installed"

rem ---- 1. Python ----
set "BASEPY="
py -3 -c "import sys" >nul 2>nul && set "BASEPY=py -3"
if not defined BASEPY python -c "import sys" >nul 2>nul && set "BASEPY=python"
if not defined BASEPY (
  echo Python was not found. Install Python 3.13 from https://www.python.org/downloads/
  echo ^(tick "Add python.exe to PATH" in the installer^), then run this file again.
  pause & exit /b 1
)
for /f "delims=" %%v in ('%BASEPY% -c "import sys; print('%%d.%%d' %% sys.version_info[:2])"') do set "PYVER=%%v"
echo Python !PYVER! found ^(%BASEPY%^).
if not "!PYVER!"=="3.13" echo NOTE: only Python 3.13 has been tested with this app; continuing anyway.

rem ---- 2. Node.js ----
set "NODEMAJOR="
for /f "tokens=1 delims=." %%n in ('node -v 2^>nul') do set "NODEMAJOR=%%n"
if not defined NODEMAJOR (
  echo Node.js was not found. Install the LTS version from https://nodejs.org and run this file again.
  pause & exit /b 1
)
set "NODEMAJOR=!NODEMAJOR:v=!"
if !NODEMAJOR! LSS 18 (
  echo Node.js 18 or newer is required ^(found v!NODEMAJOR!^). Install the LTS version from https://nodejs.org
  pause & exit /b 1
)
echo Node.js v!NODEMAJOR! found.

rem ---- 3. Python packages (first run only) ----
if not exist "%STAMP%" (
  echo Setting up the Python environment in .venv - first run only, this can take several minutes ...
  if not exist "%VPY%" %BASEPY% -m venv "%VENV%"
  if not exist "%VPY%" (
    echo Could not create the Python environment in .venv
    pause & exit /b 1
  )
  "%VPY%" -m pip install --upgrade pip
  "%VPY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Installing the Python packages failed ^(see the messages above^).
    echo Check your internet connection, delete the .venv folder and run this file again.
    pause & exit /b 1
  )
  echo installed> "%STAMP%"
)

rem ---- 4. Website packages (first run only) ----
if not exist "ecg-frontend\node_modules" (
  echo Installing the website packages - first run only ...
  pushd ecg-frontend
  call npm ci
  if errorlevel 1 call npm install
  if errorlevel 1 (
    popd
    echo Installing the website packages failed. Check your internet connection and run this file again.
    pause & exit /b 1
  )
  popd
)

rem ---- 5. Ports ----
for %%p in (8000 5173) do (
  netstat -ano | findstr /r /c:":%%p .*LISTENING" >nul && (
    echo Port %%p is already in use - is the app already running? Close the other window or program and try again.
    pause & exit /b 1
  )
)

rem ---- 6. Start ----
echo Starting the backend and the website ...
start "ECG backend (port 8000) - close to stop" /D "%REPO%" cmd /k ""%VPY%" -m uvicorn main:app --port 8000"
start "ECG website (port 5173) - close to stop" /D "%REPO%ecg-frontend" cmd /k "set VITE_SHOW_TEST_TOOLS=false&& npm run dev -- --port 5173"

echo Waiting for the backend and the model to load ...
set "HEALTH="
for /l %%i in (1,1,60) do (
  if not defined HEALTH (
    for /f "delims=" %%h in ('curl -s http://127.0.0.1:8000/health 2^>nul') do set "HEALTH=%%h"
    if not defined HEALTH ping -n 3 127.0.0.1 >nul
  )
)
if not defined HEALTH (
  echo The backend did not start within 2 minutes. Look at the "ECG backend" window for the error.
  pause & exit /b 1
)
echo Backend status: !HEALTH!
echo !HEALTH! | findstr /c:"\"model_loaded\":true" >nul && (echo Model loaded.) || (echo WARNING: the model is NOT loaded - see the "ECG backend" window. Analyses will fail.)

start "" http://localhost:5173
echo.
echo The app is open at http://localhost:5173   ^(backend API: http://127.0.0.1:8000/docs^)
echo To stop the app, close the two windows "ECG backend" and "ECG website".
pause
endlocal
