# wan22-ai-series

AI短剧自动生成流水线 — Wan2.2-TI2V-5B 视频生成版。

## 流程

```
剧本生成 → 分镜生成 → 画面生成(SD 1.5) → 视频生成(Wan2.2) → 配音 → 剪辑合成
```

## Kaggle 快速开始

```bash
# 克隆仓库
!rm -rf /kaggle/working/wan22-ai-series && cd /kaggle/working && git clone https://github.com/szchengmi/wan22-ai-series.git

# 安装依赖
!pip install -q -r /kaggle/working/wan22-ai-series/scripts/requirements.txt

# 下载模型 (需要 HF Token)
!python /kaggle/working/wan22-ai-series/scripts/download_models.py

# 运行流水线
!python /kaggle/working/wan22-ai-series/scripts/kaggle_pipeline.py

# 强制重新生成
!python /kaggle/working/wan22-ai-series/scripts/kaggle_pipeline.py --force
```

## Kaggle Secrets

在 Notebook → Add-ons → Secrets 中添加：
- `secret_value_0`: Google API Key
- `secret_value_1`: HuggingFace Token

## 模型 (~15GB)

| 模型 | 文件 | 大小 |
|------|------|------|
| SD 1.5 | v1-5-pruned-emaonly.safetensors | 4.27GB |
| Wan2.2 UNET | Wan2.2-TI2V-5B-Q4_K_M.gguf | 5.1GB |
| Wan2.2 T5 | umt5-xxl-encoder-Q4_K_M.gguf | 3.6GB |
| Wan2.2 VAE | Wan2.2_VAE.safetensors | 1.3GB |
| Qwen2.5-3B (可选) | 完整目录 | 6.2GB |

## 输出

最终视频: `/kaggle/working/ai-series/episode_01/final/episode_01_final.mp4`
