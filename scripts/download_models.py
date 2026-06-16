"""
模型下载脚本 — wan22-ai-series
================================
Kaggle 上获取模型：
  - SD 1.5: 从 Dataset 挂载点复制（如果已存在）
  - Wan2.2 5B GGUF: 从 HuggingFace 下载
  - Qwen2.5-3B: 从 HuggingFace 下载（可选）

用法:
  python download_models.py                    # 下载所有
  python download_models.py --models wan22    # 只下载 Wan2.2
  python download_models.py --models sd       # 只复制 SD 1.5
  python download_models.py --skip-existing   # 跳过已存在的文件
"""

import os
import sys
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import log, HF_TOKEN, MODEL_CACHE_DIR

# 下载目标目录
DOWNLOAD_DIR = os.environ.get("MODEL_CACHE_DIR", MODEL_CACHE_DIR)

# Dataset 挂载点（旧仓库的模型）
DATASET_PATHS = [
    "/kaggle/input/datasets/saysnkaggle/newdataset/kaggle-ai-series/models",
    "/kaggle/input/newdataset/kaggle-ai-series/models",
    "/kaggle/input/datasets/saysnkaggle/newdataset/models",
    "/kaggle/input/newdataset/models",
]


def find_sd_on_disk():
    """在 Dataset 挂载点查找 SD 1.5"""
    for base in DATASET_PATHS:
        if not os.path.isdir(base):
            continue
        # 查找 v1-5-pruned-emaonly.safetensors
        for root, dirs, files in os.walk(base):
            for f in files:
                if f == "v1-5-pruned-emaonly.safetensors":
                    return os.path.join(root, f)
    return None


def copy_sd15():
    """从 Dataset 挂载点复制 SD 1.5 到 DOWNLOAD_DIR"""
    log("查找 SD 1.5...")

    sd_file = find_sd_on_disk()
    if not sd_file:
        log("  ❌ Dataset 中未找到 SD 1.5 (v1-5-pruned-emaonly.safetensors)")
        log("  请确保 Dataset 'newdataset' 已挂载")
        return None

    dest = f"{DOWNLOAD_DIR}/stable-diffusion-v1-5"
    os.makedirs(dest, exist_ok=True)
    dest_file = f"{dest}/v1-5-pruned-emaonly.safetensors"

    if os.path.exists(dest_file) and os.path.getsize(dest_file) > 1_000_000_000:
        log(f"  跳过(已存在): {dest_file}")
        return dest

    log(f"  复制: {sd_file}")
    log(f"  目标: {dest_file}")
    shutil.copy2(sd_file, dest_file)
    size_mb = os.path.getsize(dest_file) / 1e6
    log(f"  ✅ 复制完成 ({size_mb:.0f}MB)")
    return dest


def download_with_hf(repo_id, filename, dest_dir, token=None):
    """从 HuggingFace 下载（通过 hf-mirror.com）"""
    from huggingface_hub import hf_hub_download

    # 目标路径：保持原始文件名（不替换 /）
    dest = os.path.join(dest_dir, os.path.basename(filename))

    if os.path.exists(dest) and os.path.getsize(dest) > 100_000_000:
        log(f"  跳过(已存在): {os.path.basename(filename)}")
        return dest

    log(f"  下载: {filename} ({repo_id})")
    os.makedirs(dest_dir, exist_ok=True)

    # 使用 hf-mirror.com
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
        token=token,
    )
    size_mb = os.path.getsize(dest) / 1e6
    log(f"  ✅ {os.path.basename(filename)} ({size_mb:.0f}MB)")
    return path


def download_wan22():
    """下载 Wan2.2-TI2V-5B GGUF 模型"""
    log("下载 Wan2.2-TI2V-5B GGUF...")

    # 1. UNET GGUF (Q4_K_M ~5.1GB)
    download_with_hf(
        "QuantStack/Wan2.2-TI2V-5B-GGUF",
        "Q4_K_M/Wan2.2-TI2V-5B-Q4_K_M.gguf",
        DOWNLOAD_DIR,
        token=HF_TOKEN,
    )

    # 2. T5 Text Encoder GGUF (Q4_K_M ~3.6GB)
    download_with_hf(
        "city96/umt5-xxl-encoder-gguf",
        "Q4_K_M/umt5-xxl-encoder-Q4_K_M.gguf",
        DOWNLOAD_DIR,
        token=HF_TOKEN,
    )

    # 3. VAE (GGUF 不含 VAE tensor，需要独立下载, ~1.3GB)
    download_with_hf(
        "QuantStack/Wan2.2-TI2V-5B-GGUF",
        "VAE/Wan2.2_VAE.safetensors",
        DOWNLOAD_DIR,
        token=HF_TOKEN,
    )

    log("  Wan2.2 模型下载完成")
    return DOWNLOAD_DIR


def download_qwen():
    """下载 Qwen2.5-3B-Instruct (可选，用于本地剧本生成)"""
    log("下载 Qwen2.5-3B-Instruct (可选)...")
    from huggingface_hub import snapshot_download

    dest = f"{DOWNLOAD_DIR}/Qwen2.5-3B-Instruct"
    os.makedirs(dest, exist_ok=True)

    # 检查是否已有部分文件
    if os.path.isdir(dest) and os.listdir(dest):
        log(f"  目录非空，尝试增量下载")

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    snapshot_download(
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        local_dir=dest,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )
    log("  Qwen2.5-3B 下载完成")
    return dest


def main():
    parser = argparse.ArgumentParser(description="下载 wan22-ai-series 模型")
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=["all", "wan22", "qwen"],
                        help="要下载的模型")
    parser.add_argument("--proxy", type=str, default=None,
                        help="HTTP 代理 URL (如 http://127.0.0.1:7890)")
    args = parser.parse_args()

    if args.proxy:
        os.environ["http_proxy"] = args.proxy
        os.environ["https_proxy"] = args.proxy
        log(f"代理: {args.proxy}")

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    log(f"模型目录: {DOWNLOAD_DIR}")

    models = args.models
    if "all" in models:
        models = ["wan22", "qwen"]

    if "wan22" in models:
        download_wan22()

    if "qwen" in models:
        download_qwen()

    # 汇总
    log("\n" + "=" * 50)
    log("模型目录:")
    total_size = 0
    for item in sorted(os.listdir(DOWNLOAD_DIR)):
        path = os.path.join(DOWNLOAD_DIR, item)
        if os.path.isdir(path):
            size = sum(os.path.getsize(os.path.join(path, f))
                       for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
            log(f"  {item}/ ({size / 1e9:.1f}GB)")
            total_size += size
        else:
            size = os.path.getsize(path)
            log(f"  {item} ({size / 1e9:.2f}GB)")
            total_size += size
    log(f"总计: {total_size / 1e9:.1f}GB")
    log("✅ 完成!")


if __name__ == "__main__":
    main()
