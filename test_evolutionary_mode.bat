@echo off
REM Test script for main-thesis.py evolutionary mode (Windows)

echo ==========================================
echo Testing main-thesis.py Evolutionary Mode
echo ==========================================
echo.

REM Check if .env exists
if not exist .env (
    echo WARNING: .env file not found
    echo Please create .env with your API key:
    echo   OPENAI_API_KEY=your_key_here
    echo   BASE_URL=https://api.your-provider.com/v1
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

echo Test 1: Minimal Evolutionary Run (2 parents, 4 offspring, budget 10)
echo This should take ~5-10 minutes
echo.
python main-thesis.py --evolutionary-mode --n-parents 2 --n-offspring 4 --budget 10 --eval-budget 1000

if %errorlevel% neq 0 (
    echo.
    echo Test 1 FAILED
    exit /b 1
)

echo.
echo Test 1 PASSED
echo.

echo ==========================================
echo Test 2: Iterative Mode (legacy, budget 5)
echo This should take ~3-5 minutes
echo ==========================================
echo.
python main-thesis.py --budget 5 --eval-budget 1000

if %errorlevel% neq 0 (
    echo.
    echo Test 2 FAILED
    exit /b 1
)

echo.
echo Test 2 PASSED
echo.

echo ==========================================
echo ALL TESTS PASSED!
echo ==========================================
echo.
echo You can now run full experiments:
echo.
echo Evolutionary mode:
echo   python main-thesis.py --evolutionary-mode --elitism --budget 100
echo.
echo Iterative mode:
echo   python main-thesis.py --elitism --budget 50
echo.
pause

