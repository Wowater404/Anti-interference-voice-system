@echo off
REM ============================================================
REM 全量推理独立运行脚本 - 不受会话断开影响
REM ============================================================

set PYTHON=F:\work\Anaconda\envs\zhinnegjiaju\python.exe
set WORK_DIR=F:\龙虾\2026-07-18-13-57-00\voice_pipeline
set DATA_ROOT=F:\挑杯资料\datasetA
set OUTPUT=%WORK_DIR%\results\full_inference.json
set CHECKPOINT=%WORK_DIR%\results\checkpoint.json
set LOG=%WORK_DIR%\results\full_run.log

cd /d %WORK_DIR%

echo ============================================================
echo 全量推理启动: %date% %time%
echo ============================================================
echo Python: %PYTHON%
echo 数据集: %DATA_ROOT%
echo 输出: %OUTPUT%
echo 断点: %CHECKPOINT%
echo 日志: %LOG%
echo ============================================================
echo.

%PYTHON% run_inference.py ^
    --data_root "%DATA_ROOT%" ^
    --split all ^
    --output "%OUTPUT%" ^
    --checkpoint "%CHECKPOINT%" ^
    > "%LOG%" 2>&1

echo.
echo ============================================================
echo 推理完成: %date% %time%
echo ============================================================

REM 检查结果文件
if exist "%OUTPUT%" (
    echo 结果文件已生成: %OUTPUT%
) else (
    echo [错误] 结果文件未生成，请检查日志: %LOG%
)

REM 保持窗口 5 秒
timeout /t 5 /nobreak >nul
