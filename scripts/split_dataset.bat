@echo off
REM Generic Dataset Split Script - Windows Batch Wrapper
REM This batch file simplifies running the generic split script on Windows

setlocal enabledelayedexpansion

echo.
echo ================================================================
echo   GENERIC DATASET SPLIT (70/20/10 Stratified)
echo ================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python or add it to your PATH environment variable
    exit /b 1
)

REM Check number of arguments
if "%1"=="" (
    echo USAGE:
    echo   split_dataset.bat INPUT_CSV OUTPUT_DIR LABEL_ROW [RANDOM_SEED]
    echo.
    echo EXAMPLE:
    echo   split_dataset.bat data\Finalized_GSE90073.csv data\splits_gse90073 classification 12345
    echo.
    exit /b 1
)

REM Parse arguments
set INPUT_CSV=%1
set OUTPUT_DIR=%2
set LABEL_ROW=%3
set RANDOM_SEED=%4

REM Default seed if not provided
if "%RANDOM_SEED%"=="" (
    set RANDOM_SEED=12345
)

REM Check if input file exists
if not exist "%INPUT_CSV%" (
    echo ERROR: Input file not found: %INPUT_CSV%
    exit /b 1
)

REM Display parameters
echo Parameters:
echo   Input CSV:     %INPUT_CSV%
echo   Output Dir:    %OUTPUT_DIR%
echo   Label Row:     %LABEL_ROW%
echo   Random Seed:   %RANDOM_SEED%
echo.
echo Starting split process...
echo ================================================================
echo.

REM Run the Python script
python scripts/generic_split_dataset.py "%INPUT_CSV%" "%OUTPUT_DIR%" "%LABEL_ROW%" %RANDOM_SEED%

REM Check exit status
if %errorlevel% equ 0 (
    echo.
    echo ================================================================
    echo SUCCESS: Dataset split completed!
    echo Output files saved to: %OUTPUT_DIR%
    echo ================================================================
) else (
    echo.
    echo ================================================================
    echo ERROR: Split process failed with exit code %errorlevel%
    echo ================================================================
    exit /b %errorlevel%
)

endlocal
