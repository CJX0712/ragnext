# RAGNext Makefile —— 一键复现（venv 隔离）
# 用法：
#   make install     建 .venv 并安装核心 + 可选依赖（可选依赖失败容忍）
#   make demo        运行内置离线示例（Local + Numpy + Extractive）
#   make test        运行 pytest 单测套件
#   make benchmark   运行检索基准并生成 benchmark_report.json
#   make clean       清理 venv 与产物

PYTHON ?= python3
VENV_DIR ?= .venv

ifeq ($(OS),Windows_NT)
	PIP = $(VENV_DIR)/Scripts/pip
	PY = $(VENV_DIR)/Scripts/python
else
	PIP = $(VENV_DIR)/bin/pip
	PY = $(VENV_DIR)/bin/python
endif

.PHONY: install demo test benchmark clean

install:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	-$(PY) -m pip install -r requirements-sota.txt

demo:
	$(PY) -m ragnext.examples.run_demo

test:
	$(PY) -m pytest -q

benchmark:
	$(PY) scripts/benchmark.py

clean:
	rm -rf $(VENV_DIR) ragnext.index.npz benchmark_report.json
