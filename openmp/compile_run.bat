@echo off
:: =========================================================
:: compile_run.bat
:: Compile and run OpenMP Traffic Density Simulation
:: Group 10
::
:: Requirements: MinGW-w64 gcc with OpenMP support
::   (typically at: C:\mingw64\bin\gcc.exe)
::
:: Usage:
::   compile_run.bat          -> compile + run (default)
::   compile_run.bat compile  -> compile only
::   compile_run.bat run      -> run only (requires existing binary)
:: =========================================================

setlocal

set SRC=traffic_openmp.c
set EXE=traffic_openmp.exe
set CC=gcc
set FLAGS=-fopenmp -O2
set LIBS=-lm

echo.
echo ==========================================================
echo   OpenMP Traffic Simulation - Build and Run
echo   Group 10
echo ==========================================================

:: --- Compile ---
if /I "%1"=="run" goto RUN

echo.
echo [Compiling] %CC% %FLAGS% -o %EXE% %SRC% %LIBS%
%CC% %FLAGS% -o %EXE% %SRC% %LIBS%

if errorlevel 1 (
    echo.
    echo [ERROR] Compilation failed.
    echo   Make sure gcc with OpenMP support is installed and on PATH.
    echo   Recommended: MinGW-w64  https://winlibs.com/
    exit /b 1
)
echo [OK] Compiled successfully: %EXE%

if /I "%1"=="compile" goto END

:: --- Run ---
:RUN
if not exist %EXE% (
    echo [ERROR] %EXE% not found. Run compile_run.bat compile first.
    exit /b 1
)

echo.
echo [Running] %EXE%
echo ----------------------------------------------------------
%EXE%

echo.
echo [Output Files Generated]
if exist openmp_output.txt          echo   openmp_output.txt
if exist openmp_traffic_heatmap.ppm echo   openmp_traffic_heatmap.ppm
if exist openmp_traffic_values.txt  echo   openmp_traffic_values.txt
if exist openmp_performance.txt     echo   openmp_performance.txt

:END
echo.
echo ==========================================================
echo   Done
echo ==========================================================
endlocal
