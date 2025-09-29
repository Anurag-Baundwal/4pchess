@echo off
ECHO --- 4pChess Engine Tuner for Windows ---

REM --- IMPORTANT FILE PATHS ---
SET SCM_PATH="C:\Users\dell3\source\repos5\simplechessmatch-sprt\scm\x64\Release\scm-tony.exe"
SET ENGINE_PATH="C:\Users\dell3\source\repos5\4pchess-2\cli.exe"
SET FENS_PATH="C:\Users\dell3\source\repos5\simplechessmatch-sprt\scm\x64\Release\FENs_4PC_balanced.txt"

ECHO SCM Path: %SCM_PATH%
ECHO Engine Path: %ENGINE_PATH%
ECHO FENs Path: %FENS_PATH%
ECHO.

@REM ECHO Building the engine...
@REM bazel build -c opt //:cli
@REM IF %ERRORLEVEL% NEQ 0 (
@REM     ECHO Engine build failed. Exiting.
@REM     EXIT /B 1
@REM )
@REM ECHO Engine build successful.
@REM ECHO.

ECHO Starting the tuning process...
python tune.py ^
  -scm_path %SCM_PATH% ^
  -engine_path %ENGINE_PATH% ^
  -fens_path %FENS_PATH% ^
  -tune_params "lmr_min_moves,lmr_depth_sub,lmr_depth_div,lmr_move_count_div,lmp_q_base1,lmp_q_div_declining1,lmp_q_div_normal1,lmp_q_base2,lmp_q_div_declining2,lmp_q_div_normal2,lmp_q_improving_mult" ^
  -n_calls 250

ECHO Tuning complete.