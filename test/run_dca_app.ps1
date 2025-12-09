#!/usr/bin/env pwsh
<#
.SYNOPSIS
    DCA 定投回测平台 - 快速启动脚本
    
.DESCRIPTION
    自动启动 Streamlit Web 应用，无需手动命令行操作
    
.EXAMPLE
    .\run_dca_app.ps1
#>

param(
    [switch]$Help,
    [int]$Port = 8501,
    [switch]$NoOpen
)

if ($Help) {
    @"
DCA 定投回测平台 - 启动脚本

用法:
  .\run_dca_app.ps1                 # 启动应用 (自动打开浏览器)
  .\run_dca_app.ps1 -Port 8502     # 指定端口 8502
  .\run_dca_app.ps1 -NoOpen        # 启动但不自动打开浏览器
  .\run_dca_app.ps1 -Help          # 显示此帮助信息

选项:
  -Port <int>      指定运行端口，默认 8501
  -NoOpen          启动后不自动打开浏览器
  -Help            显示帮助信息

"@
    exit 0
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   DCA 定投回测平台 (DCA Backtest Platform)  " -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python 环境
Write-Host "🔍 检查 Python 环境..." -ForegroundColor Yellow
$python = python --version 2>&1
if (-not $?) {
    Write-Host "❌ 错误：未找到 Python" -ForegroundColor Red
    Write-Host "请先安装 Python 3.8+ 版本" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Python: $python" -ForegroundColor Green

# 检查依赖
Write-Host ""
Write-Host "🔍 检查依赖包..." -ForegroundColor Yellow
$packages = @("streamlit", "tushare", "pandas", "plotly")
$missing = @()

foreach ($pkg in $packages) {
    try {
        python -c "import $pkg" 2>$null
        Write-Host "✅ $pkg" -ForegroundColor Green
    } catch {
        Write-Host "❌ $pkg (缺失)" -ForegroundColor Red
        $missing += $pkg
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "📦 安装缺失的包..." -ForegroundColor Yellow
    pip install -q @missing
    Write-Host "✅ 依赖安装完成" -ForegroundColor Green
}

# 检查 .env 文件
Write-Host ""
Write-Host "🔍 检查 Tushare Token 配置..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  未找到 .env 文件" -ForegroundColor Yellow
    Write-Host "   创建 .env 文件并添加: TUSHARE_TOKEN=your_token_here" -ForegroundColor Yellow
    Write-Host ""
    @"
如何获取 Token:
  1. 访问 https://www.tushare.pro/
  2. 注册并登录账户
  3. 在"个人中心"获取 API Token
  4. 在 .env 文件中配置 TUSHARE_TOKEN

"@ | Write-Host -ForegroundColor Cyan
}

# 启动应用
Write-Host ""
Write-Host "🚀 启动应用..." -ForegroundColor Green
Write-Host "   Web 应用地址: http://localhost:$Port" -ForegroundColor Cyan
Write-Host ""

# 记录启动时间
$startTime = Get-Date

# 构建启动命令
$command = "streamlit run dca_web_app.py --server.port=$Port"

# 如果不是 NoOpen，延迟后打开浏览器
if (-not $NoOpen) {
    Write-Host "💡 提示: 应用启动后浏览器会自动打开" -ForegroundColor Gray
    Write-Host ""
    
    # 在后台启动应用
    $job = Start-Job -ScriptBlock {
        param($cmd)
        Invoke-Expression $cmd
    } -ArgumentList $command
    
    # 等待服务启动，然后打开浏览器
    Start-Sleep -Seconds 3
    Start-Process "http://localhost:$Port"
    
    # 等待 Job 完成（用户关闭应用）
    Wait-Job -Job $job
    Receive-Job -Job $job
} else {
    # 直接运行
    Invoke-Expression $command
}

Write-Host ""
Write-Host "👋 应用已关闭" -ForegroundColor Yellow
