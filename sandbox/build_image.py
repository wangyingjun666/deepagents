"""
构建沙箱镜像。

    python sandbox/build_image.py                # 构建 deep-search-sandbox:latest
    python sandbox/build_image.py --check        # 只检查 Docker 与镜像是否就绪
    python sandbox/build_image.py --no-cache     # 全量重建

镜像含 LibreOffice 和中文字体，首次构建约 5-10 分钟、700MB 左右。构建产物落在
Docker 数据根目录，空间不足时先改 Docker Desktop 的数据根位置。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import docker
    from docker.errors import BuildError, DockerException, ImageNotFound
except ImportError:
    print("缺少 docker SDK：pip install docker")
    raise SystemExit(1)

IMAGE = "deep-search-sandbox:latest"
DOCKERFILE = PROJECT_ROOT / "sandbox" / "worker" / "Dockerfile"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"


def preflight() -> int:
    """构建前置检查：.dockerignore 必须是纯 ASCII。

    docker-py 读取 .dockerignore 时用 `open(dockerignore)` 且没传 encoding
    （docker/api/build.py:153），按系统区域编码解码，中文 Windows 上是 GBK。
    UTF-8 中文会在这里抛 UnicodeDecodeError，报错信息看不出跟 .dockerignore
    有关，所以提前拦下来给出可读提示。
    """
    if not DOCKERIGNORE.exists():
        return 0
    try:
        DOCKERIGNORE.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        line = DOCKERIGNORE.read_text(encoding="utf-8", errors="replace").splitlines()
        lineno = DOCKERIGNORE.read_bytes()[: exc.start].count(b"\n") + 1
        print(f"[X] {DOCKERIGNORE.name} 第 {lineno} 行含非 ASCII 字符，无法构建。")
        print(f"    内容：{line[lineno - 1].strip() if lineno <= len(line) else ''}")
        print("    原因：docker-py 用区域编码（本机 GBK）解码该文件，UTF-8 中文会让")
        print("          构建在启动前就抛 UnicodeDecodeError，且报错信息与 .dockerignore 无关。")
        print("    处理：把该文件改成纯 ASCII（注释也请用英文）。")
        print("          临时绕过：set PYTHONUTF8=1 后再执行本脚本。")
        return 1
    return 0


def check() -> int:
    try:
        client = docker.from_env()
        version = client.version()
    except DockerException as exc:
        print(f"[X] Docker 守护进程不可达：{exc}")
        print("    -> 确认 Docker Desktop 已启动，且托盘图标显示 Engine running")
        return 1

    print(f"[OK] Docker {version.get('Version')}（API {version.get('ApiVersion')}）")
    try:
        image = client.images.get(IMAGE)
        print(f"[OK] 镜像已存在：{IMAGE}  id={image.id[:12]}  大小={image.attrs['Size'] / 1e6:.0f}MB")
        return 0
    except ImageNotFound:
        print(f"[!] 镜像不存在：{IMAGE}")
        print("    -> 执行 python sandbox/build_image.py 构建")
        return 2


def build(no_cache: bool = False) -> int:
    rc = preflight()
    if rc:
        return rc

    client = docker.from_env()
    if not DOCKERFILE.exists():
        print(f"[X] 找不到 Dockerfile：{DOCKERFILE}")
        return 1

    print(f"开始构建 {IMAGE}（构建上下文：{PROJECT_ROOT}）...")
    try:
        image, logs = client.images.build(
            path=str(PROJECT_ROOT),
            dockerfile=str(DOCKERFILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            tag=IMAGE,
            rm=True,
            nocache=no_cache,
            labels={"app": "deep-search-sandbox"},
        )
    except BuildError as exc:
        print("[X] 构建失败，最后 30 行日志：")
        for chunk in list(exc.build_log)[-30:]:
            line = chunk.get("stream") or chunk.get("error") or ""
            if line.strip():
                print("   ", line.rstrip())
        return 1
    except DockerException as exc:
        print(f"[X] 构建异常：{exc}")
        return 1

    for chunk in logs:
        line = chunk.get("stream") or ""
        if line.strip():
            print("   ", line.rstrip())

    print(f"\n[OK] 镜像构建完成：{IMAGE}  id={image.id[:12]}")
    print("     自检：python sandbox/build_image.py --check")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="构建/检查 deep-search 沙箱镜像")
    parser.add_argument("--check", action="store_true", help="只检查，不构建")
    parser.add_argument("--no-cache", action="store_true", help="不使用构建缓存")
    args = parser.parse_args()

    if args.check:
        return check()
    rc = build(no_cache=args.no_cache)
    if rc == 0:
        check()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
