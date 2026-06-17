"""
Step 4: 视频生成 (Wan2.2-TI2V-5B GGUF)
========================================
通过 ComfyUI API 调用 Wan2.2 生成视频。
使用 comfy-core 原生节点: UnetLoaderGGUF + CLIPLoaderGGUF + WanImageToVideo + KSampler

模型需求:
  - UNET: Wan2.2-TI2V-5B-Q4_K_M.gguf (5.1GB)
  - CLIP: umt5-xxl-encoder-Q4_K_M.gguf (3.6GB)
  - VAE: Wan2.2_VAE.safetensors (1.3GB)
  总计约10GB，T4 16GB 可用

工作流节点连接:
  UnetLoaderGGUF → KSampler.model
  CLIPLoaderGGUF → CLIPTextEncode.clip → KSampler.conditioning
  VAELoader → WanImageToVideo.vae
  EmptyLatentVideo → WanImageToVideo.latent
  WanImageToVideo → KSampler.latent_image
  CLIPTextEncode (positive/negative) → KSampler
  KSampler → VAEDecode → VHS_VideoCombine
"""

import os
import sys
import json
import time
import shutil
import urllib.request
import urllib.error
import subprocess
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    log, save_json, load_json, run_cmd,
    EPISODE_NUM, get_dirs, MODEL_CACHE_DIR, DTYPE,
    WAN22_DEFAULTS,
)


# ============================================================
# ComfyUI 管理
# ============================================================

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_DIR = "/kaggle/working/ComfyUI"


def _find_comfyui():
    """查找 ComfyUI 目录（验证 main.py + 关键依赖存在）"""
    for candidate in ["/kaggle/working/ComfyUI", "/kaggle/working/ComfyUI-master"]:
        if os.path.isdir(candidate) and os.path.isfile(f"{candidate}/main.py"):
            # 检查 requirements 是否已安装（看 server.py 和 comfy 包）
            comfy_pkg = os.path.join(candidate, "comfy")
            server_py = os.path.join(candidate, "server.py")
            if os.path.isdir(comfy_pkg) and os.path.isfile(server_py):
                return candidate
            # 目录存在但不完整，删掉重新安装
            log(f"  ComfyUI 目录不完整，将重新安装")
            shutil.rmtree(candidate, ignore_errors=True)
    return None


def _install_comfyui():
    """安装 ComfyUI + GGUF 插件 (zip 方式)"""
    log("安装 ComfyUI...")
    parent = os.path.dirname(COMFYUI_DIR)
    os.chdir(parent)
    run_cmd("curl -sL https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.zip -o /tmp/comfyui.zip")
    run_cmd("unzip -qo /tmp/comfyui.zip -d /kaggle/working/")
    run_cmd("rm -f /tmp/comfyui.zip")
    if os.path.isdir("/kaggle/working/ComfyUI-master"):
        os.rename("/kaggle/working/ComfyUI-master", "/kaggle/working/ComfyUI")
    run_cmd("pip install -q -r requirements.txt", timeout=300)
    run_cmd("pip install -q xformers gguf accelerate", timeout=120)
    # GGUF 插件
    run_cmd("curl -sL https://github.com/city96/ComfyUI-GGUF/archive/refs/heads/main.zip -o /tmp/gguf.zip")
    run_cmd("unzip -qo /tmp/gguf.zip -d custom_nodes/")
    run_cmd("mv custom_nodes/ComfyUI-GGUF-main custom_nodes/ComfyUI-GGUF 2>/dev/null; true")
    run_cmd("rm -f /tmp/gguf.zip")
    log("ComfyUI 安装完成")


def _start_comfyui():
    """启动 ComfyUI 服务器（后台）"""
    # 已在运行？
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=2)
        log("ComfyUI 已在运行")
        return True
    except:
        pass

    # 查找或安装
    cwd = _find_comfyui()
    if not cwd:
        _install_comfyui()
        cwd = COMFYUI_DIR

    log("启动 ComfyUI...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = subprocess.Popen(
        ["python", "main.py", "--dont-print-server", "--highvram",
         "--preview-method", "none", "--port", "8188",
         "--cuda-device", "0"],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    # 快速轮询（3s 间隔），最多 10 分钟
    for i in range(200):
        time.sleep(3)
        try:
            urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=2)
            log(f"  ComfyUI 就绪 ({(i + 1) * 3}s)")
            return True
        except:
            if i < 5 or i % 20 == 0:
                log(f"  ⏳ 等待... ({(i + 1) * 3}s)")
    return False


def _queue_prompt(workflow):
    """提交工作流到 ComfyUI API"""
    payload = json.dumps({"prompt": workflow, "client_id": "wan22-ai"}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def _get_history(prompt_id):
    """获取工作流执行结果"""
    req = urllib.request.Request(f"{COMFYUI_URL}/history/{prompt_id}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _wait_for_completion(prompt_id, timeout=1800):
    """等待工作流执行完成"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            history = _get_history(prompt_id)
            if prompt_id in history:
                entry = history[prompt_id]
                if entry.get("status", {}).get("status_str") == "success":
                    return entry
                if entry.get("status", {}).get("status_str") == "error":
                    raise RuntimeError(f"工作流执行失败: {entry}")
        except urllib.error.HTTPError:
            pass
        time.sleep(5)
    raise TimeoutError(f"工作流 {prompt_id} 超时 ({timeout}s)")


# ============================================================
# 工作流构建
# ============================================================

def _build_wan22_workflow(positive_prompt, negative_prompt, model_path,
                           clip_path, vae_path, width, height, frames,
                           steps, cfg, sampler, scheduler, shift, seed):
    """
    构建 Wan2.2 T2V ComfyUI 工作流（API 格式）。

    节点:
      1: UnetLoaderGGUF — 加载 GGUF UNET
      2: CLIPLoaderGGUF — 加载 text encoder
      3: VAELoader — 加载 VAE
      4: EmptyLatentVideo — 创建空 latent
      5: CLIPTextEncode (positive) — 正面提示词
      6: CLIPTextEncode (negative) — 负面提示词
      7: WanImageToVideo — T2V latent 准备
      8: KSampler — 去噪采样
      9: VAEDecode — 解码 latent 为像素
      10: VHS_VideoCombine — 保存视频
    """
    # 计算 latent 尺寸
    lat_w = width // 8
    lat_h = height // 8
    lat_frames = (frames - 1) // 4 + 1

    workflow = {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {},
            "widgets_values": [model_path],
        },
        "2": {
            "class_type": "CLIPLoaderGGUF",
            "inputs": {},
            "widgets_values": [clip_path, "wan"],  # type 必须是 "wan"
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {},
            "widgets_values": [vae_path],
        },
        "4": {
            "class_type": "EmptyLatentVideo",
            "inputs": {},
            "widgets_values": [width, height, lat_frames, 1],  # width, height, batch, frames_div4+1 的变体
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0]},
            "widgets_values": [positive_prompt],
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0]},
            "widgets_values": [negative_prompt],
        },
        "7": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["5", 0],
                "negative": ["6", 0],
                "vae": ["3", 0],
                "latent_image": ["4", 0],
            },
            "widgets_values": [width, height, lat_frames],
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["7", 0],
                "negative": ["7", 1],
                "latent_image": ["7", 2],
            },
            "widgets_values": [seed, "fixed", steps, cfg, sampler, scheduler, 1.0],
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["3", 0],
            },
        },
        "10": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["9", 0],
            },
            "widgets_values": [8, 0, "output", "h264-mp4", False, True],
        },
    }
    return workflow


# ============================================================
# 主函数
# ============================================================

def main(storyboard=None):
    log("=" * 50)
    log("Step 4: 视频生成 (Wan2.2-TI2V-5B)")
    log("=" * 50)

    dirs = get_dirs(EPISODE_NUM)
    total = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))

    # 模型路径
    model_path = f"{dirs['models']}/Wan2.2-TI2V-5B-Q4_K_M.gguf"
    clip_path = f"{dirs['models']}/umt5-xxl-encoder-Q4_K_M.gguf"
    vae_path = f"{dirs['models']}/Wan2.2_VAE.safetensors"

    # 检查模型
    for name, path in [("UNET", model_path), ("CLIP", clip_path), ("VAE", vae_path)]:
        if not os.path.isfile(path):
            log(f"  ❌ {name} 模型不存在: {path}")
            log(f"  请运行: python download_models.py")
            return
        size_mb = os.path.getsize(path) / 1e6
        log(f"  ✅ {name}: {size_mb:.0f}MB")

    # 启动 ComfyUI
    if not _start_comfyui():
        log("❌ ComfyUI 启动失败")
        return

    # Wan2.2 默认参数
    w = WAN22_DEFAULTS["width"]
    h = WAN22_DEFAULTS["height"]
    fps = WAN22_DEFAULTS["fps"]
    steps = WAN22_DEFAULTS["steps"]
    cfg = WAN22_DEFAULTS["cfg"]
    sampler = WAN22_DEFAULTS["sampler"]
    scheduler = WAN22_DEFAULTS["scheduler"]
    shift = WAN22_DEFAULTS["shift"]
    num_frames = WAN22_DEFAULTS["frames_per_shot"]

    log(f"参数: {w}x{h} | {num_frames}f | {steps}步 | CFG={cfg} | {sampler}/{scheduler}")

    count = 0
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            ep = storyboard.get("episode", 1)
            out = f"{dirs['videos']}/ep{ep:02d}_{scene['scene_id']}_{sid}.mp4"

            if os.path.exists(out) and os.path.getsize(out) > 100000:
                log(f"  [{count}/{total}] {sid} 跳过(已存在)")
                continue

            # 构建视频 prompt（从分镜 prompt + 动作描述）
            video_prompt = f"{shot.get('prompt', '')}, smooth motion, cinematic, high quality"
            neg_prompt = shot.get("negative_prompt", "blurry, distorted, low quality, static, motionless")

            seed = shot.get("seed", -1)
            if seed < 0:
                seed = WAN22_DEFAULTS["seed"]

            workflow = _build_wan22_workflow(
                positive_prompt=video_prompt,
                negative_prompt=neg_prompt,
                model_path=model_path,
                clip_path=clip_path,
                vae_path=vae_path,
                width=w, height=h,
                frames=num_frames,
                steps=steps, cfg=cfg,
                sampler=sampler, scheduler=scheduler,
                shift=shift, seed=seed,
            )

            try:
                log(f"  [{count}/{total}] {sid} 提交中...")
                result = _queue_prompt(workflow)
                prompt_id = result.get("prompt_id")
                if not prompt_id:
                    raise RuntimeError(f"API 返回无 prompt_id: {result}")

                log(f"  [{count}/{total}] {sid} 等待完成 (id={prompt_id[:8]}...)")
                completion = _wait_for_completion(prompt_id, timeout=1800)

                # 提取输出视频路径
                outputs = completion.get("outputs", {})
                video_output = None
                for node_id, node_output in outputs.items():
                    if "vhs_filenames" in node_output:
                        video_output = node_output["vhs_filenames"][0]
                        break

                if video_output and os.path.isfile(video_output):
                    # 复制到目标路径
                    import shutil
                    shutil.copy2(video_output, out)
                    dur = shot.get("duration_seconds", 3)
                    log(f"  [{count}/{total}] {sid} ✓ ({num_frames}f, {dur}s)")
                else:
                    log(f"  [{count}/{total}] {sid} 无输出视频")
                    _save_placeholder_video(shot, out, num_frames)

            except Exception as e:
                log(f"  [{count}/{total}] {sid} 失败: {e}")
                _save_placeholder_video(shot, out, num_frames)

    log("视频生成完成")


def _save_placeholder_video(shot, output_path, num_frames):
    from PIL import Image, ImageDraw
    import shutil
    res = WAN22_DEFAULTS["width"]
    img = Image.new("RGB", (res, res), (20, 20, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle([5, 5, res - 5, res - 5], outline=(100, 100, 200), width=2)
    draw.text((20, 30), "[VIDEO]", fill=(200, 200, 255))
    draw.text((20, 70), shot.get("shot_id", ""), fill=(200, 255, 200))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_png = output_path.replace(".mp4", "_tmp.png")
    img.save(tmp_png)
    dur = max(num_frames / WAN22_DEFAULTS["fps"], 1)
    run_cmd(
        f'ffmpeg -y -loop 1 -i "{tmp_png}" -t {dur} '
        f'-c:v libx264 -pix_fmt yuv420p -movflags +faststart "{output_path}" 2>/dev/null'
    )
    if os.path.exists(tmp_png):
        os.remove(tmp_png)


if __name__ == "__main__":
    dirs = get_dirs(EPISODE_NUM)
    storyboard = load_json(f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_storyboard.json")
    main(storyboard)
