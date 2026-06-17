#!/usr/bin/env python3
"""
wan22-ai-series — AI短剧自动生成端到端流水线
================================================
Kaggle Notebook 单文件执行版本。

流程:
  Step 1: 剧本生成 (Gemini API / Qwen本地 / 预置)
  Step 2: 分镜生成 (剧本→视频 prompt)
  Step 3: 视频生成 (Wan2.2-TI2V-5B GGUF via ComfyUI API)
  Step 4: 配音生成 (ChatTTS / edge-tts)
  Step 5: 剪辑合成 (FFmpeg + MoviePy)

用法:
  python kaggle_pipeline.py          # 正常运行
  python kaggle_pipeline.py --force  # 清除所有输出重新生成
"""

import os
import sys
import json
import re
import time
import argparse
import shutil

# ============================================================
# 确保 import path
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ============================================================
# 从 common 导入
# ============================================================
from common import (
    log, save_json, load_json, run_cmd,
    EPISODE_NUM, GENRE, NUM_SCENES, SHOTS_PER_SCENE,
    IMAGE_STEPS, IMAGE_GUIDANCE,
    CHARACTER_PROMPTS, SCENE_PROMPTS, WAN22_DEFAULTS,
    MODEL_CACHE_DIR, DTYPE,
    get_dirs, get_fallback_script,
    GOOGLE_API_KEY, HF_TOKEN,
)

# ============================================================
# 环境检测
# ============================================================

def detect_environment():
    """检测运行环境"""
    env = {
        "is_kaggle": os.path.isdir("/kaggle/input"),
        "is_mac": sys.platform == "darwin",
        "has_gpu": False,
        "gpu_name": "none",
        "gpu_mem_gb": 0,
        "cpu_count": os.cpu_count(),
        "total_memory_gb": 0,
    }

    try:
        import torch
        env["has_gpu"] = torch.cuda.is_available()
        if env["has_gpu"]:
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_mem_gb"] = torch.cuda.get_device_properties(0).total_mem / 1e9
    except:
        pass

    try:
        import psutil
        env["total_memory_gb"] = psutil.virtual_memory().total / 1e9
    except:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    env["total_memory_gb"] = int(line.split()[1]) / 1e6
                    break

    return env


def find_models():
    """查找模型文件（Dataset 挂载 或 本地目录）"""
    search_paths = []

    # Kaggle Dataset 挂载点
    for dataset_dir in ["/kaggle/input/datasets/saysnkaggle/wan2-2-5b-q4-gguf",
                        "/kaggle/input/datasets/saysnkaggle/newdataset",
                        "/kaggle/input/newdataset"]:
        if os.path.isdir(dataset_dir):
            search_paths.append(f"{dataset_dir}/model")
            search_paths.append(f"{dataset_dir}/models")
            search_paths.append(f"{dataset_dir}/kaggle-ai-series/models")

    # 本地目录（新仓库）
    search_paths.extend([
        MODEL_CACHE_DIR,  # /kaggle/working/wan22-ai-series/models
        "/kaggle/working/wan22-ai-series/models",
        "/kaggle/working/models",
        f"{SCRIPT_DIR}/../models",
    ])

    model_dirs = {
        "wan22_unet": None,
        "wan22_clip": None,
        "wan22_vae": None,
        "qwen": None,
    }

    for base in search_paths:
        if not os.path.isdir(base):
            continue

        # Wan22 UNET GGUF
        for name in ["Wan2.2-TI2V-5B-Q4_K_M.gguf", "Wan2.2-TI2V-5B-Q3_K_S.gguf",
                     "Wan2.2-TI2V-5B.gguf"]:
            if os.path.isfile(f"{base}/{name}"):
                model_dirs["wan22_unet"] = f"{base}/{name}"
                break

        # Wan22 CLIP (T5 GGUF)
        for name in ["umt5-xxl-encoder-Q4_K_M.gguf", "umt5-xxl-encoder-Q3_K_S.gguf",
                     "umt5-xxl-encoder.gguf"]:
            if os.path.isfile(f"{base}/{name}"):
                model_dirs["wan22_clip"] = f"{base}/{name}"
                break

        # Wan22 VAE
        for name in ["Wan2.2_VAE.safetensors", "Wan2.2_VAE.pth"]:
            if os.path.isfile(f"{base}/{name}"):
                model_dirs["wan22_vae"] = f"{base}/{name}"
                break

        # Qwen2.5-3B
        qw_path = f"{base}/Qwen2.5-3B-Instruct"
        if os.path.isdir(qw_path) and os.path.isfile(f"{qw_path}/config.json"):
            model_dirs["qwen"] = qw_path

    return model_dirs


# ============================================================
# Step 1: 剧本生成
# ============================================================

def step1_generate_script(force=False):
    """生成剧本 JSON"""
    dirs = get_dirs(EPISODE_NUM)
    out_path = f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_script.json"

    if not force and os.path.exists(out_path):
        log(f"  跳过(已存在): {out_path}")
        return load_json(out_path)

    prompt = _build_story_prompt()
    text = None

    # 优先：本地 Qwen LLM
    try:
        text = _generate_with_local_llm(prompt)
        log("  本地 LLM ✓")
    except Exception as e:
        log(f"  本地 LLM 失败: {e}")

    # 降级：Gemini API
    if text is None and GOOGLE_API_KEY:
        log("  尝试 Gemini API...")
        try:
            text = _generate_with_gemini(prompt)
            log("  Gemini API ✓")
        except Exception as e:
            log(f"  Gemini API 失败: {e}")

    # 最终降级：预置剧本
    if text is None:
        log("  使用预置剧本")
        script_data = get_fallback_script(EPISODE_NUM, NUM_SCENES, SHOTS_PER_SCENE)
    else:
        script_data = _parse_script_response(text)

    save_json(script_data, out_path)
    return script_data


def _build_story_prompt():
    return f"""你是一个专业的中文短剧编剧。请为一部{GENRE}题材的AI短剧写第{EPISODE_NUM}集的完整剧本。

【角色设定】
- 小明(xiaoming): 28岁程序员, 内向但善良, 戴眼镜, 短发, 常穿深色卫衣
- 小丽(xiaoli): 26岁平面设计师, 活泼开朗, 长发, 穿浅色连衣裙
- 王总(boss_wang): 45岁公司总监, 严厉但公正, 西装革履

【可用场景】
- office: 现代办公室, 落地窗, 简约风格
- cafe: 温馨咖啡馆, 木质桌椅, 暖黄灯光
- park: 城市公园, 绿树成荫
- apartment: 温馨公寓, 北欧风格
- street: 城市街道, 傍晚

【要求】
1. 时长3-5分钟（约800-1200字）
2. 包含{NUM_SCENES}个场景，每个场景{SHOTS_PER_SCENE}个镜头
3. 完整故事线：开头→发展→高潮→结尾（留悬念）
4. 对话口语化，符合角色性格
5. 结尾留悬念

输出纯JSON（不要markdown标记）：
{{"episode": {EPISODE_NUM}, "title": "标题", "duration_estimate": "3-5分钟",
"scenes": [{{"scene_id": "scene_1", "location": "office", "time_of_day": "morning",
"lighting": "自然光", "mood": "描述氛围",
"shots": [{{"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
"duration_seconds": 3, "description": "画面描述", "character": "xiaoming",
"action": "动作描述", "dialogue": "对话", "narration": "旁白",
"emotion": "情绪", "subtitle": "字幕"}}]}}],
"characters_used": ["xiaoming", "xiaoli"], "next_episode_hook": "下集预告"}}"""


def _generate_with_gemini(prompt):
    from google import genai
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config={"temperature": 0.9, "top_p": 0.95, "top_k": 40, "max_output_tokens": 8192}
    )
    return response.text


def _generate_with_local_llm(prompt):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # 找 Qwen 模型
    model_path = None
    for candidate in [
        f"{MODEL_CACHE_DIR}/Qwen2.5-3B-Instruct",
        "/kaggle/working/models/Qwen2.5-3B-Instruct",
    ]:
        if os.path.isdir(candidate) and os.path.isfile(f"{candidate}/config.json"):
            model_path = candidate
            break

    if not model_path:
        model_path = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True, use_safetensors=True,
    )
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=4096, temperature=0.9, top_p=0.95,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _parse_script_response(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    # 修复常见 JSON 问题：模型可能返回字面 \n 作为换行
    text = text.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # 调试：输出前200字符
        log(f"  ⚠️ JSON 解析失败: {e}")
        log(f"  输出前200字符: {text[:200]!r}")
        # 尝试提取第一个完整 JSON 对象
        depth = 0
        start = -1
        for i, c in enumerate(text):
            if c == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    return json.loads(text[start:i+1])
        # 尝试用 yaml 解析（更宽松）
        try:
            import yaml
            result = yaml.safe_load(text)
            if isinstance(result, dict):
                log(f"  ✅ YAML 解析成功")
                return result
        except:
            pass
        raise


# ============================================================
# Step 2: 分镜生成
# ============================================================

def step2_generate_storyboard(script_data, force=False):
    """将剧本转换为视频分镜（含 Wan2.2 视频 prompt）"""
    dirs = get_dirs(EPISODE_NUM)
    out_path = f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_storyboard.json"

    if not force and os.path.exists(out_path):
        log(f"  跳过(已存在): {out_path}")
        return load_json(out_path)

    from step2_generate_storyboard import main as step2_main
    storyboard = step2_main(script_data)
    return storyboard


# ============================================================
# Step 3: 视频生成 (Wan2.2 T2V via ComfyUI API)
# ============================================================

def step3_generate_videos(storyboard, force=False):
    """通过 ComfyUI API 调用 Wan2.2 文本生成视频"""
    dirs = get_dirs(EPISODE_NUM)
    total = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))
    log(f"  镜头数: {total}")

    from step4_generate_videos_wan22 import main as wan22_main
    wan22_main(storyboard)


# ============================================================
# Step 4: 配音生成
# ============================================================

def step4_generate_audio(storyboard, force=False):
    """生成配音"""
    from step5_generate_audio import main as tts_main
    tts_main(storyboard)


# ============================================================
# Step 5: 剪辑合成
# ============================================================

def step5_compose(storyboard, script_data=None):
    """最终合成"""
    from step6_compose import main as compose_main
    return compose_main(storyboard, script_data)


# ============================================================
# 运行时依赖安装
# ============================================================

def _install_runtime_deps():
    """安装运行时依赖（幂等，已安装则跳过）"""
    deps = [
        "edge-tts>=6.1.0",
        "soundfile>=0.12.0",
        "moviepy>=1.0.3",
        "scipy>=1.10.0",
        "psutil>=5.9.0",
    ]
    for dep in deps:
        pkg = dep.split(">=")[0].split("==")[0]
        try:
            import importlib
            importlib.import_module(pkg.replace("-", "_"))
        except ImportError:
            log(f"  安装 {dep}...")
            run_cmd(f"pip install -q {dep}", timeout=120)

    # 确保 ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
    except:
        log("  安装 ffmpeg...")
        run_cmd("apt-get update -qq && apt-get install -y -qq ffmpeg 2>/dev/null", timeout=120)


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="wan22-ai-series AI短剧生成")
    parser.add_argument("--force", action="store_true", help="清除所有输出重新生成")
    parser.add_argument("--episode", type=int, default=None, help="集数")
    args = parser.parse_args()

    # 设置全局变量
    import common
    if args.episode is not None:
        common.EPISODE_NUM = args.episode
        global EPISODE_NUM
        EPISODE_NUM = args.episode

    log("╔══════════════════════════════════════════╗")
    log("║   AI短剧自动生成 — Wan2.2 版             ║")
    log("╚══════════════════════════════════════════╝")

    # 环境检测
    env = detect_environment()
    log(f"环境: {'Kaggle' if env['is_kaggle'] else '本地'} | "
        f"GPU: {env['gpu_name']} ({env['gpu_mem_gb']:.1f}GB) | "
        f"CPU: {env['cpu_count']}核 | RAM: {env['total_memory_gb']:.1f}GB")

    # 查找模型
    models = find_models()
    log("模型检测:")
    for name, path in models.items():
        if path:
            size_mb = os.path.getsize(path) / 1e6 if os.path.isfile(path) else 0
            log(f"  ✓ {name}: {os.path.basename(path)} ({size_mb:.0f}MB)")
        else:
            log(f"  ✗ {name}: 未找到")

    # 关键模型检查
    if not models["wan22_unet"] or not models["wan22_vae"]:
        log("❌ Wan2.2 模型未找到! 请运行: python download_models.py --models wan22")
        return
    if not models["wan22_clip"]:
        log("⚠️ Wan2.2 T5 CLIP 未找到，视频质量可能受影响")

    # 安装运行时依赖
    _install_runtime_deps()

    # 清除旧输出
    if args.force:
        log("清除旧输出...")
        dirs = get_dirs(EPISODE_NUM)
        for key in ["images", "videos", "audio", "final"]:
            if os.path.isdir(dirs[key]):
                shutil.rmtree(dirs[key])
                os.makedirs(dirs[key])

    # 执行流水线
    start_time = time.time()

    log("\n" + "=" * 50)
    log("Step 1: 剧本生成")
    log("=" * 50)
    t = time.time()
    script_data = step1_generate_script(force=args.force)
    log(f"  耗时: {time.time() - t:.1f}s | 场景: {len(script_data.get('scenes', []))}")
    total_shots = sum(len(s.get("shots", [])) for s in script_data.get("scenes", []))
    log(f"  总镜头: {total_shots}")

    log("\n" + "=" * 50)
    log("Step 2: 分镜生成")
    log("=" * 50)
    t = time.time()
    storyboard = step2_generate_storyboard(script_data, force=args.force)
    log(f"  耗时: {time.time() - t:.1f}s")

    log("\n" + "=" * 50)
    log("Step 3: 视频生成 (Wan2.2 T2V)")
    log("=" * 50)
    t = time.time()
    step3_generate_videos(storyboard, force=args.force)
    log(f"  耗时: {time.time() - t:.1f}s")

    log("\n" + "=" * 50)
    log("Step 4: 配音生成")
    log("=" * 50)
    t = time.time()
    step4_generate_audio(storyboard, force=args.force)
    log(f"  耗时: {time.time() - t:.1f}s")

    log("\n" + "=" * 50)
    log("Step 5: 剪辑合成")
    log("=" * 50)
    t = time.time()
    final_path = step5_compose(storyboard, script_data)
    log(f"  耗时: {time.time() - t:.1f}s")

    # 总结
    total_time = time.time() - start_time
    log("\n" + "=" * 50)
    log("完成!")
    log("=" * 50)
    log(f"总耗时: {total_time:.0f}s ({total_time / 60:.1f}min)")
    if final_path and os.path.exists(final_path):
        size_mb = os.path.getsize(final_path) / 1e6
        log(f"输出: {final_path} ({size_mb:.1f}MB)")
    else:
        log("⚠️ 最终输出不存在，请检查各步骤日志")


if __name__ == "__main__":
    main()
