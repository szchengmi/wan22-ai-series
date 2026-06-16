"""
模型下载脚本 — wan22-ai-series
================================
下载所有模型到 Kaggle working 目录。

Kaggle DNS 无法直连 hf.co，必须通过代理或 hf-mirror.com。
使用 huggingface_hub + hf_mirror 镜像。

模型清单:
  1. SD 1.5 (v1-5-pruned-emaonly.safetensors) ~4.27GB
  2. Wan2.2-TI2V-5B-Q4_K_M.gguf ~5.1GB (UNET GGUF)
  3. umt5-xxl-encoder-Q4_K_M.gguf ~3.6GB (T5 text encoder)
  4. Wan2.2_VAE.safetensors ~1.3GB
  5. Qwen2.5-3B-Instruct (完整目录) ~6.2GB (可选)

用法:
  python download_models.py
  python download_models.py --proxy http://127.0.0.1:7890
  python download_models.py --models sd wan22  # 只下载指定模型
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import log, HF_TOKEN, MODEL_CACHE_DIR

# 下载目标目录
DOWNLOAD_DIR = os.environ.get("MODEL_CACHE_DIR", MODEL_CACHE_DIR)


def download_with_hf_mirror(repo_id, filename, dest_dir, token=None):
    """通过 hf-mirror.com 下载"""
    from huggingface_hub import hf_hub_download

    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000_000:
        log(f"  跳过(已存在): {filename}")
        return dest

    log(f"  下载: {repo_id}/{filename}")
    os.makedirs(dest_dir, exist_ok=True)

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
        token=token,
        mirror="https://hf-mirror.com",
    )
    return path


def download_sd15():
    """下载 SD 1.5"""
    log("下载 SD 1.5...")
    dest = f"{DOWNLOAD_DIR}/stable-diffusion-v1-5"
    os.makedirs(dest, exist_ok=True)

    # 单文件版本
    download_with_hf_mirror(
        "runwayml/stable-diffusion-v1-5",
        "v1-5-pruned-emaonly.safetensors",
        dest,
        token=HF_TOKEN,
    )
    return dest


def download_wan22():
    """下载 Wan2.2 5B GGUF 模型"""
    log("下载 Wan2.2-TI2V-5B GGUF...")

    # UNET GGUF
    download_with_hf_mirror(
        "QuantStack/Wan2.2-TI2V-5B-GGUF",
        "Q4_K_M/Wan2.2-TI2V-5B-Q4_K_M.gguf",
        DOWNLOAD_DIR,
        token=HF_TOKEN,
    )

    # T5 Text Encoder GGUF
    download_with_hf_mirror(
        "city96/umt5-xxl-encoder-gguf",
        "Q4_K_M/umt5-xxl-encoder-Q4_K_M.gguf",
        DOWNLOAD_DIR,
        token=HF_TOKEN,
    )

    # VAE (GGUF 不含 VAE tensor，需要独立下载)
    download_with_hf_mirror(
        "QuantStack/Wan2.2-TI2V-5B-GGUF",
        "VAE/Wan2.2_VAE.safetensors",
        DOWNLOAD_DIR,
        token=HF_TOKEN,
    )

    return DOWNLOAD_DIR


def download_qwen():
    """下载 Qwen2.5-3B-Instruct (可选，用于本地剧本生成)"""
    log("下载 Qwen2.5-3B-Instruct (可选)...")
    from huggingface_hub import snapshot_download

    dest = f"{DOWNLOAD_DIR}/Qwen2.5-3B-Instruct"
    os.makedirs(dest, exist_ok=True)

    snapshot_download(
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        local_dir=dest,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
        mirror="https://hf-mirror.com",
    )
    return dest


def main():
    parser = argparse.ArgumentParser(description="下载 wan22-ai-series 模型")
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=["all", "sd", "wan22", "qwen"],
                        help="要下载的模型")
    parser.add_argument("--proxy", type=str, default=None,
                        help="HTTP 代理 URL (如 http://127.0.0.1:7890)")
    args = parser.parse_args()

    if args.proxy:
        os.environ["http_proxy"] = args.proxy
        os.environ["https_proxy"] = args.proxy
        log(f"代理: {args.proxy}")

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    log(f"下载目录: {DOWNLOAD_DIR}")

    models = args.models
    if "all" in models:
        models = ["sd", "wan22", "qwen"]

    if "sd" in models:
        download_sd15()
    if "wan22" in models:
        download_wan22()
    if "qwen" in models:
        download_qwen()

    log("\n✅ 模型下载完成!")
    log("模型目录:")
    for item in sorted(os.listdir(DOWNLOAD_DIR)):
        path = os.path.join(DOWNLOAD_DIR, item)
        if os.path.isdir(path):
            size = sum(os.path.getsize(os.path.join(path, f))
                       for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
            log(f"  {item}/ ({size / 1e9:.1f}GB)")
        else:
            size = os.path.getsize(path)
            log(f"  {item} ({size / 1e9:.2f}GB)")


if __name__ == "__main__":
    main()
