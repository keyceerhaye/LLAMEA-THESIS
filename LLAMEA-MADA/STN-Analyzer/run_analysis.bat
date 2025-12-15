@echo off
REM Quick batch script to run STN analyzer on a specific experiment

echo ========================================
echo STN Analyzer - Quick Run
echo ========================================
echo.

REM Get the experiment directory from command line or use default
if "%1"=="" (
    echo Usage: run_analysis.bat "path\to\experiment\directory"
    echo.
    echo Example:
    echo   run_analysis.bat "..\exp-12-13_091444-google-gemini-2.5-flash-mada-v2-experiment-evolutionary-elitism(4,16(4+16, 0.9discount, 0.8-0.4cosine))"
    echo.
    pause
    exit /b 1
)

set EXP_DIR=%~1

echo Experiment: %EXP_DIR%
echo.

REM Check if experiment directory exists
if not exist "%EXP_DIR%" (
    echo ERROR: Experiment directory not found: %EXP_DIR%
    echo.
    pause
    exit /b 1
)

echo Running analyzer with detailed debugging...
echo.

python analyze_experiment.py "%EXP_DIR%" --output-dir stn_outputs

echo.
echo ========================================
echo Analysis complete!
echo Check the output above for any errors.
echo ========================================
pause
