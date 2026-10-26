FROM python:3.12-slim

WORKDIR /app

# 安裝 uv 工具
RUN pip install uv

# 複製專案檔案
COPY . /app

# 安裝相依套件
RUN uv sync

# 執行主要程式
CMD ["uv", "run", "main.py"]
