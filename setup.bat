@echo off
echo ========================================
echo Tea Yield Prediction System - Setup
echo ========================================
echo.

echo Step 1: Setting up Backend...
cd backend
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate
echo Installing Python dependencies...
pip install -r requirements.txt
echo.
echo Generating dataset...
python -m ml.data_generation
echo.
echo Training XGBoost model...
python -m ml.train
echo Backend setup complete!
echo.
cd ..

echo Step 2: Setting up Frontend...
cd frontend
echo Installing Node.js dependencies...
call npm install
echo Frontend setup complete!
echo.
cd ..

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To run the application:
echo.
echo 1. Start Backend (in terminal 1):
echo    cd backend
echo    venv\Scripts\activate
echo    uvicorn app.main:app --reload
echo.
echo 2. Start Frontend (in terminal 2):
echo    cd frontend
echo    npm run dev
echo.
echo Backend will run on: http://localhost:8000
echo Frontend will run on: http://localhost:3000
echo.
pause
