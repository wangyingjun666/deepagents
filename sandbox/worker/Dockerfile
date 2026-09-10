# ============================================================================
#  deep-search 沙箱镜像
#  构建：python sandbox/build_image.py        （或 docker build -f sandbox/worker/Dockerfile -t deep-search-sandbox:latest .）
#
#  会话执行环境：模型触发的文件读写与 PDF 渲染都在这里发生，宿主进程不直接碰磁盘。
#
#  镜像内包含：
#    - python:3.12-slim 基础运行时
#    - LibreOffice（headless 渲染 PDF）
#    - Noto CJK + 文泉驿中文字体（否则中文 PDF 渲染成方块）
#    - markdown 库（md -> html 这一跳）
# ============================================================================
FROM python:3.12-slim

LABEL org.opencontainers.image.title="deep-search-sandbox" \
      org.opencontainers.image.description="Per-session sandboxed execution environment for deep_search agent" \
      org.opencontainers.image.version="1.0.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# --- 中文字体 + LibreOffice（只装 writer，够渲染文档就行，镜像体积可控）---
RUN apt-get update && apt-get install -y --no-install-recommends \
        libreoffice-writer \
        fonts-noto-cjk \
        fonts-wqy-zenhei \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        markdown==3.10.2

# --- 非 root 用户 ---
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin sandbox

# --- 挂载点：/workspace 会话可写，/uploads 只读 ---
RUN mkdir -p /workspace /uploads /opt/sandbox && chown -R sandbox:sandbox /workspace /opt/sandbox

# --- 操作执行器 + 容器内路径守卫 ---
COPY sandbox/worker/ops.py /opt/sandbox/ops.py
COPY security/path_guard.py /opt/sandbox/path_guard.py
RUN chmod 0555 /opt/sandbox/ops.py /opt/sandbox/path_guard.py

USER sandbox
WORKDIR /workspace

# 容器常驻，宿主通过 `docker exec` 调用 ops.py（单次调用无状态）
CMD ["sleep", "infinity"]
