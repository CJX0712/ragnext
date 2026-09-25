# RAGNext 安装脚本（Windows PowerShell）
# 创建 venv 并安装核心依赖；可选 SOTA 依赖安装失败则忽略。
$ErrorActionPreference = "Stop"

$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$venv = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv" }

Write-Host ">> 创建虚拟环境：$venv"
& $py -m venv $venv

$pip = Join-Path $venv "Scripts/pip.exe"
$pyexe = Join-Path $venv "Scripts/python.exe"

& $pip install --upgrade pip
Write-Host ">> 安装核心依赖（requirements.txt）"
& $pip install -r requirements.txt
Write-Host ">> 尝试安装可选 SOTA 依赖（失败容忍）"
& $pip install -r requirements-sota.txt
Write-Host ">> 安装完成。运行：make demo"
