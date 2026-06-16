# wan22-ai-series

AI短剧自动生成流水线 — Wan2.2-TI2V-5B 视频生成版。

## 流程

```
剧本生成 → 分镜生成 → 视频生成(Wan2.2 T2V) → 配音 → 剪辑合成
```

## Kaggle 快速开始

```bash
# 1. 克隆仓库
!rm -rf /kaggle/working/wan22-ai-series && cd /kaggle/working && git clone https://github.com/szchengmi/wan22-ai-series.git

# 2. 安装依赖
!pip install -q -r /kaggle/working/wan22-ai-series/scripts/requirements.txt

# 3. 下载模型 (~8.2GB)
!python /kaggle/working/wan22-ai-series/scripts/download_models.py

# 4. 运行流水线
!python /kaggle/working/wan22-ai-series/scripts/kaggle_pipeline.py --force

# 5. 查看输出
!ls -lh /kaggle/working/ai-series/episode_01/final/
```

## Kaggle Secrets

在 Notebook → Add-ons → Secrets 中添加：
- `secret_value_0`: Google API Key (剧本生成)
- `secret_value_1`: HuggingFace Token (模型下载)

## 模型 (~8.2GB)

| 模型 | 文件 | 大小 |
|------|------|------|
| Wan2.2 UNET | Wan2.2-TI2V-5B-Q4_K_M.gguf | 3.27GB |
| Wan2.2 T5 | umt5-xxl-encoder-Q4_K_M.gguf | 3.65GB |
| Wan2.2 VAE | Wan2.2_VAE.safetensors | 1.34GB |
| Qwen2.5-3B (可选) | 完整目录 | 6.2GB |

## 输出

最终视频: `/kaggle/working/ai-series/episode_01/final/episode_01_final.mp4`

## 参数调整

编辑 `scripts/common.py` 中的 `WAN22_DEFAULTS` 可调整视频参数：
- `width/height`: 分辨率 (默认 846x480)
- `frames_per_shot`: 每镜头帧数 (默认 25, ~3秒@8fps)
- `steps`: 去噪步数 (默认 20)
- `cfg`: 分类器自由引导 (默认 5.0)
