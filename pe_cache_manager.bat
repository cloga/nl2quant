@echo off
REM PE数据缓存管理便捷脚本

echo ================================================
echo PE数据缓存管理工具
echo ================================================
echo.
echo 请选择操作:
echo 1. 启动Web管理界面（推荐）
echo 2. 从缓存快速导出CSV
echo 3. 增量更新缓存（跳过已有）
echo 4. 强制全量更新
echo 5. 测试模式（仅50只股票）
echo 6. 查看缓存状态
echo 0. 退出
echo.

set /p choice=请输入选项 (0-6): 

if "%choice%"=="1" goto web
if "%choice%"=="2" goto export
if "%choice%"=="3" goto update
if "%choice%"=="4" goto force
if "%choice%"=="5" goto test
if "%choice%"=="6" goto status
if "%choice%"=="0" goto end

echo 无效选项，请重新运行
goto end

:web
echo.
echo 启动Web管理界面...
.venv\Scripts\streamlit.exe run main.py
goto end

:export
echo.
echo 从缓存快速导出CSV...
.venv\Scripts\python.exe scripts/batch_compute_pe_v2.py --from-cache
pause
goto end

:update
echo.
echo 增量更新缓存...
.venv\Scripts\python.exe scripts/batch_compute_pe_v2.py
pause
goto end

:force
echo.
echo 强制全量更新（将重新计算所有股票）...
set /p confirm=确认执行？这可能需要1-2小时 (Y/N): 
if /i "%confirm%"=="Y" (
    .venv\Scripts\python.exe scripts/batch_compute_pe_v2.py --force-update
    pause
)
goto end

:test
echo.
echo 测试模式（仅处理50只股票）...
.venv\Scripts\python.exe scripts/batch_compute_pe_v2.py --limit 50
pause
goto end

:status
echo.
echo 查看缓存状态...
.venv\Scripts\python.exe scripts/test_pe_cache.py
pause
goto end

:end
