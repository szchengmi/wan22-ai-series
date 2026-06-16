"""
Step 3: 画面生成 (SD 1.5)
===========================
用 Stable Diffusion 1.5 为每个分镜生成图片。
支持断点续传（跳过已生成的图片）。
"""

import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    log, save_json, load_json,
    EPISODE_NUM, IMAGE_STEPS, IMAGE_GUIDANCE, get_dirs,
    MODEL_CACHE_DIR, DTYPE,
)


def main(storyboard=None):
    log("=" * 50)
    log("Step 3: 画面生成 (SD 1.5)")
    log("=" * 50)

    has_gpu = torch.cuda.is_available()
    device = "cuda" if has_gpu else "cpu"
    dirs = get_dirs(EPISODE_NUM)
    total = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))

    from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

    mp = f"{dirs['models']}/stable-diffusion-v1-5"
    if not os.path.exists(mp) or not os.listdir(mp):
        mp = "runwayml/stable-diffusion-v1-5"

    sd_file = f"{mp}/v1-5-pruned-emaonly.safetensors"
    if not os.path.isfile(sd_file):
        sd_file = "runwayml/stable-diffusion-v1-5"
        log(f"加载 SD (HF): {sd_file}")
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_file, torch_dtype=DTYPE,
            safety_checker=None, requires_safety_checker=False,
        )
    else:
        log(f"加载 SD (单文件): {sd_file}")
        sd_cache = f"{dirs['base']}/cache/stable-diffusion-v1-5"
        os.makedirs(sd_cache, exist_ok=True)
        os.environ["HF_HOME"] = sd_cache
        pipe = StableDiffusionPipeline.from_single_file(
            sd_file, torch_dtype=DTYPE,
            safety_checker=None, requires_safety_checker=False,
            cache_dir=sd_cache,
        )

    # VAE: from_single_file 已包含，只在为 None 时外部加载
    if pipe.vae is None:
        try:
            from diffusers import AutoencoderKL
            vp = f"{dirs['models']}/vae-ft-mse"
            if not os.path.exists(vp):
                vp = "stabilityai/sd-vae-ft-mse"
            pipe.vae = AutoencoderKL.from_pretrained(vp, torch_dtype=DTYPE)
        except Exception as e:
            log(f"VAE: {e}")

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if has_gpu:
        try:
            pipe.enable_attention_slicing()
        except:
            pass
        try:
            pipe.enable_vae_slicing()
        except:
            pass
        pipe.to(device)
    else:
        pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        log("xformers ✓")
    except:
        pass

    log(f"生成 {total} 张...")
    count = 0
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            ep = storyboard.get("episode", 1)
            out = f"{dirs['images']}/ep{ep:02d}_{scene['scene_id']}_{sid}.png"
            if os.path.exists(out):
                log(f"  [{count}/{total}] {sid} 跳过(已存在)")
                continue

            w = max((shot["width"] // 8) * 8, 512)
            h = max((shot["height"] // 8) * 8, 512)
            gen = None
            if shot.get("seed", -1) > 0:
                gen = torch.Generator(device).manual_seed(shot["seed"])

            try:
                result = pipe(
                    prompt=shot["prompt"],
                    negative_prompt=shot.get("negative_prompt", ""),
                    width=w, height=h,
                    num_inference_steps=shot.get("steps", IMAGE_STEPS),
                    guidance_scale=shot.get("guidance", IMAGE_GUIDANCE),
                    generator=gen,
                )
                result.images[0].save(out)
                log(f"  [{count}/{total}] {sid} ({w}x{h}) ✓")
            except Exception as e:
                log(f"  [{count}/{total}] {sid} 失败: {e}")
                _save_placeholder_image(shot, out)

            if has_gpu and count % 5 == 0:
                torch.cuda.empty_cache()

    log("画面生成完成")


def _save_placeholder_image(shot, output_path):
    from PIL import Image, ImageDraw
    w = max((shot.get("width", 768) // 8) * 8, 512)
    h = max((shot.get("height", 768) // 8) * 8, 512)
    img = Image.new("RGB", (w, h), (20, 20, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, w - 10, h - 10], outline=(100, 100, 200), width=2)
    draw.text((30, 40), f"[PLACEHOLDER] {shot.get('shot_id', '')}", fill=(200, 200, 255))
    draw.text((30, 80), f"Char: {shot.get('character', '')}", fill=(200, 255, 200))
    draw.text((30, 120), f"Desc: {shot.get('description', '')[:50]}", fill=(255, 255, 200))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)


if __name__ == "__main__":
    dirs = get_dirs(EPISODE_NUM)
    storyboard = load_json(f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_storyboard.json")
    main(storyboard)
