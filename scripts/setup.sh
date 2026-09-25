#!/usr/bin/env bash
# RAGNext 安装脚本（POSIX / macOS / Linux）
# 创建 venv 并安装核心依赖；可选 SOTA 依赖安装失败则忽略。
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo ">> 创建虚拟环境：$VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"

if [ "$(uname)" = "Darwin" ] || [ "$(uname)" = "Linux" ]; then
  PIP="$VENV_DIR/bin/pip"
  PY="$VENV_DIR/bin/python"
else
  PIP="$VENV_DIR/Scripts/pip"
  PY="$VENV_DIR/Scripts/python"
fi

"$PY" -m pip install --upgrade pip
echo ">> 安装核心依赖（requirements.txt）"
"$PY" -m pip install -r requirements.txt
echo ">> 尝试安装可选 SOTA 依赖（失败容忍）"
"$PY" -m pip install -r requirements-sota.txt || echo ">> 跳过可选依赖（离线兜底链路仍可用）"
echo ">> 安装完成。运行：make demo"
