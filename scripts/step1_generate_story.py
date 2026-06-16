"""
Step 1: 剧本生成
================
优先用本地 LLM（Qwen2.5-3B），失败则用 Gemini API，最终降级预置剧本。
输出 JSON 剧本到 storyboard 目录。
"""

import os
import sys
import json

# 确保 import common 能找到模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    log, save_json, load_json, get_fallback_script,
    EPISODE_NUM, GENRE, NUM_SCENES, SHOTS_PER_SCENE,
    CHARACTER_PROMPTS, GOOGLE_API_KEY, HF_TOKEN, MODEL_CACHE_DIR,
)


def build_story_prompt():
    """构建剧本生成的 prompt"""
    return f"""你是一个专业的中文短剧编剧。请为一部{GENRE}题材的AI短剧写第{EPISODE_NUM}集的完整剧本。

【角色设定】
- 小明(xiaoming): 28岁, 程序员, 内向但善良, 戴眼镜, 短发, 常穿深色卫衣
- 小丽(xiaoli): 26岁, 平面设计师, 活泼开朗, 长发, 穿浅色连衣裙
- 王总(boss_wang): 45岁, 公司总监, 严厉但公正, 西装革履

【可用场景】
- office: 现代办公室, 落地窗, 简约风格, 暖色调
- cafe: 温馨咖啡馆, 木质桌椅, 暖黄灯光
- park: 城市公园, 绿树成荫, 长椅和喷泉
- apartment: 温馨公寓, 北欧风格
- street: 城市街道, 傍晚, 路灯

【要求】
1. 时长3-5分钟（约800-1200字）
2. 包含{NUM_SCENES}个场景，每个场景{SHOTS_PER_SCENE}个镜头
3. 完整故事线：开头→发展→高潮→结尾（留悬念）
4. 对话口语化，符合角色性格
5. 包含场景描述、角色动作、对话、旁白
6. 结尾留悬念

输出纯JSON（不要markdown标记，不要```json```）：
{{"episode": {EPISODE_NUM}, "title": "标题", "duration_estimate": "3-5分钟",
"scenes": [{{"scene_id": "scene_1", "location": "office", "time_of_day": "morning",
"lighting": "自然光", "mood": "描述氛围",
"shots": [{{"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
"duration_seconds": 3, "description": "画面描述", "character": "xiaoming",
"action": "动作描述", "dialogue": "对话", "narration": "旁白",
"emotion": "情绪", "subtitle": "字幕"}}]}}],
"characters_used": ["xiaoming", "xiaoli"], "next_episode_hook": "下集预告"}}"""


def parse_script_response(text):
    """解析 LLM 返回的 JSON 剧本"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    return json.loads(text)


def generate_with_gemini(prompt):
    """用 Gemini API 生成（支持代理）"""
    from google import genai
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config={
            "temperature": 0.9, "top_p": 0.95, "top_k": 40, "max_output_tokens": 8192,
        }
    )
    return response.text


def generate_with_local_llm(prompt):
    """用本地 Qwen2.5-3B 生成（Kaggle T4 GPU）"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    local_path = f"{MODEL_CACHE_DIR}/Qwen2.5-3B-Instruct"
    if os.path.isdir(local_path) and os.path.isfile(f"{local_path}/config.json"):
        model_path = local_path
        log(f"从本地加载: {model_path}")
    else:
        model_path = "Qwen/Qwen2.5-3B-Instruct"
        log(f"从 HF 下载: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True, local_files_only=True, use_safetensors=True,
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


def main():
    log("=" * 50)
    log("Step 1: 剧本生成")
    log("=" * 50)

    prompt = build_story_prompt()
    text = None

    # 优先用本地 LLM
    try:
        text = generate_with_local_llm(prompt)
        log("本地 LLM 成功")
    except Exception as e:
        log(f"本地 LLM 失败: {e}")

    # 降级：Gemini API
    if text is None and GOOGLE_API_KEY:
        log("尝试 Gemini API...")
        try:
            text = generate_with_gemini(prompt)
            log("Gemini API 成功")
        except Exception as e:
            log(f"Gemini API 失败: {e}")

    # 最终降级：预置剧本
    if text is None:
        log("使用预置剧本")
        script_data = get_fallback_script(EPISODE_NUM, NUM_SCENES, SHOTS_PER_SCENE)
    else:
        script_data = parse_script_response(text)

    # 保存
    from common import get_dirs
    dirs = get_dirs(EPISODE_NUM)
    save_json(script_data, f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_script.json")

    log(f"剧本: {script_data.get('title')}")
    total_shots = sum(len(s.get("shots", [])) for s in script_data.get("scenes", []))
    log(f"场景: {len(script_data.get('scenes', []))} | 镜头: {total_shots}")
    return script_data


if __name__ == "__main__":
    main()
