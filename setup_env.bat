@echo off
REM Create and activate a virtual environment, then install all dependencies.
REM Run this once from the project root: setup_env.bat

echo Creating virtual environment...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing PyTorch (CPU)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo Installing other dependencies...
pip install -r requirements.txt

echo.
echo Setup complete. To activate in future sessions:
echo   .venv\Scripts\activate.bat
echo.
echo To run experiments:
echo   python run_experiments.py
