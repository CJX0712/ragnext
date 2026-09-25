# RAGNext 运行镜像：默认走离线兜底链路
FROM python:3.13-slim

WORKDIR /app

# 先装核心依赖（离线兜底）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码并可编辑安装（提供 CLI / 示例入口）
COPY . .
RUN pip install --no-cache-dir -e .

# 安装 make 以执行 Makefile 目标
RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

# 构建 venv 并运行离线 demo
RUN make install
CMD ["make", "demo"]
