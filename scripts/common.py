"""
wan22-ai-series 共享配置和工具
=================================
所有 step 脚本和主 pipeline 共享的配置、角色定义、工具函数。
"""

import os
import json
import time
import subprocess
import shutil

# ============================================================
# 环境变量
# ============================================================
EPISODE_NUM = int(os.environ.get("EPISODE_NUM", "1"))
GENRE = os.environ.get("GENRE", "都市情感")
NUM_SCENES = int(os.environ.get("NUM_SCENES", "3"))
SHOTS_PER_SCENE = int(os.environ.get("SHOTS_PER_SCENE", "2"))
IMAGE_STEPS = int(os.environ.get("IMAGE_STEPS", "20"))
IMAGE_GUIDANCE = float(os.environ.get("IMAGE_GUIDANCE", "7.5"))

# HuggingFace — 每次使用时动态读取，避免模块加载时 Kaggle 未初始化
def _get_hf_token():
    """动态获取 HF Token，优先环境变量 → Kaggle Secrets"""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            # 支持两种 key 名
            for key in ["HF_TOKEN", "secret_value_1"]:
                try:
                    token = UserSecretsClient().get_secret(key)
                    if token:
                        break
                except:
                    continue
        except:
            pass
    return token

# 模块级保留，但 download 时会动态刷新
HF_TOKEN = _get_hf_token()

# Google API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    try:
        from kaggle_secrets import UserSecretsClient
        GOOGLE_API_KEY = UserSecretsClient().get_secret("secret_value_0")
    except:
        pass

# 模型目录
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR",
    "/kaggle/working/wan22-ai-series/models")

# PyTorch 数据类型
import torch
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ============================================================
# 目录结构
# ============================================================

def get_dirs(episode_num=1):
    """获取所有工作目录路径"""
    base = os.environ.get("OUTPUT_BASE", "/kaggle/working/ai-series")
    ep = f"episode_{episode_num:02d}"
    dirs = {
        "base": base,
        "storyboard": f"{base}/{ep}/storyboards",
        "images": f"{base}/{ep}/images",
        "videos": f"{base}/{ep}/videos",
        "audio": f"{base}/{ep}/audio",
        "final": f"{base}/{ep}/final",
        "models": MODEL_CACHE_DIR,
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ============================================================
# 类型别名（ComfyUI 工作流节点类型）
# ============================================================

# GGUF 节点（ComfyUI-GGUF 插件）
UNET_LOADER_GGUF = "UnetLoaderGGUF"
CLIP_LOADER_GGUF = "CLIPLoaderGGUF"
VAE_LOADER = "VAELoader"

# ComfyUI 内置节点
WAN_IMAGE_TO_VIDEO = "WanImageToVideo"
KSAMPLER = "KSampler"
CLIP_TEXT_ENCODE = "CLIPTextEncode"
EMPTY_LATENT_VIDEO = "EmptyLatentVideo"
VAE_DECODE = "VAEDecode"
SAVE_VIDEO = "VHS_VideoCombine"
UNET_LOADER = "UNETLoader"
CLIP_LOADER = "CLIPLoader"


# ============================================================
# 日志
# ============================================================

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# JSON 工具
# ============================================================

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Shell 命令
# ============================================================

def run_cmd(cmd, timeout=600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


# ============================================================
# SRT 时间格式
# ============================================================

def seconds_to_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ============================================================
# 角色/场景设定
# ============================================================

CHARACTER_PROMPTS = {
    "xiaoming": {
        "base_prompt": "1boy, young Chinese man, short black hair, wearing glasses, wearing dark hoodie, anime style, high quality",
        "negative_prompt": "ugly, deformed, bad anatomy, blurry, low quality",
        "seed": 42
    },
    "xiaoli": {
        "base_prompt": "1girl, young Chinese woman, long black hair, wearing light-colored dress, anime style, high quality",
        "negative_prompt": "ugly, deformed, bad anatomy, blurry, low quality",
        "seed": 123
    },
    "boss_wang": {
        "base_prompt": "1man, middle-aged Chinese man, square face, thick eyebrows, wearing business suit, anime style, high quality",
        "negative_prompt": "ugly, deformed, bad anatomy, blurry, low quality",
        "seed": 456
    }
}

SCENE_PROMPTS = {
    "office": "modern office interior, floor-to-ceiling windows, minimalist design, warm tones, anime background",
    "cafe": "cozy cafe interior, wooden furniture, warm yellow lighting, anime background",
    "park": "city park, green trees, bench and fountain, anime background",
    "apartment": "cozy apartment, Nordic style, living room, anime background",
    "street": "city street at dusk, street lamps, anime background"
}

EMOTION_ENHANCE = {
    "happy": "smiling, bright expression", "sad": "sad expression, teary eyes",
    "angry": "angry expression, furrowed brows", "surprised": "surprised expression, wide eyes",
    "nervous": "nervous expression, sweating", "calm": "calm expression, relaxed",
    "determined": "determined expression, confident", "embarrassed": "embarrassed expression, blushing",
    "thoughtful": "thoughtful expression, contemplative"
}

SHOT_PARAMS = {
    "close_up": {"w": 768, "h": 768, "prefix": "close-up shot of"},
    "medium_shot": {"w": 768, "h": 512, "prefix": "medium shot of"},
    "wide_shot": {"w": 1024, "h": 576, "prefix": "wide shot of"},
    "extreme_close_up": {"w": 512, "h": 768, "prefix": "extreme close-up of"}
}

VOICE_PARAMS = {
    "xiaoming": {"speed": 0.9, "temp": 0.3, "top_p": 0.7, "top_k": 20},
    "xiaoli": {"speed": 1.1, "temp": 0.4, "top_p": 0.8, "top_k": 25},
    "boss_wang": {"speed": 0.85, "temp": 0.25, "top_p": 0.6, "top_k": 15},
    "narrator": {"speed": 1.0, "temp": 0.3, "top_p": 0.7, "top_k": 20}
}

EMOTION_SPEED = {
    "happy": 1.1, "sad": 0.85, "angry": 1.15, "surprised": 1.2,
    "nervous": 1.1, "calm": 1.0, "determined": 1.05,
    "embarrassed": 1.1, "thoughtful": 0.9
}

# Wan2.2 视频生成参数
WAN22_DEFAULTS = {
    "width": 846,
    "height": 480,
    "fps": 8,
    "steps": 20,
    "cfg": 5.0,
    "sampler": "dpmpp_2m",
    "scheduler": "karras",
    "shift": 8,
    "seed": 42,
    "frames_per_shot": 25,  # 每镜头帧数（约3秒 @ 8fps）
}

# ============================================================
# 预置剧本（LLM 全部失败时使用）
# ============================================================

def get_fallback_script(episode_num=1, num_scenes=6, shots_per_scene=3):
    """返回预置的都市情感剧本"""
    from common import log  # 避免循环引用时用
    
    scenes = []
    scene_templates = [
        {
            "scene_id": "scene_1", "location": "office", "time_of_day": "morning",
            "lighting": "自然光", "mood": "平静日常",
            "shots": [
                {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                 "duration_seconds": 3, "description": "小明在办公室敲代码", "character": "xiaoming",
                 "action": "专注地敲击键盘", "dialogue": "", "narration": "周一的早晨，办公室里只有键盘的声音。",
                 "emotion": "calm", "subtitle": "周一的早晨，办公室里只有键盘的声音。"},
                {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                 "duration_seconds": 2, "description": "小明表情特写", "character": "xiaoming",
                 "action": "微微皱眉", "dialogue": "这个需求又改了...", "narration": "",
                 "emotion": "thoughtful", "subtitle": "这个需求又改了..."},
                {"shot_id": "shot_3", "shot_type": "medium_shot", "camera_movement": "static",
                 "duration_seconds": 3, "description": "小丽走进办公室", "character": "xiaoli",
                 "action": "推门进来，笑着打招呼", "dialogue": "早啊小明！今天天气真好！", "narration": "",
                 "emotion": "happy", "subtitle": "早啊小明！今天天气真好！"},
            ]
        },
        {
            "scene_id": "scene_2", "location": "cafe", "time_of_day": "afternoon",
            "lighting": "暖黄灯光", "mood": "温馨浪漫",
            "shots": [
                {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                 "duration_seconds": 3, "description": "小明和小丽在咖啡馆聊天", "character": "xiaoming",
                 "action": "端着咖啡杯，认真倾听", "dialogue": "你觉得这个设计方案怎么样？", "narration": "",
                 "emotion": "calm", "subtitle": "你觉得这个设计方案怎么样？"},
                {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                 "duration_seconds": 2, "description": "小丽眼睛发亮", "character": "xiaoli",
                 "action": "眼睛发亮，兴奋地比划", "dialogue": "我觉得配色可以再大胆一些！", "narration": "",
                 "emotion": "happy", "subtitle": "我觉得配色可以再大胆一些！"},
                {"shot_id": "shot_3", "shot_type": "medium_shot", "camera_movement": "pan_right",
                 "duration_seconds": 3, "description": "两人相视而笑", "character": "xiaoming",
                 "action": "忍不住笑了", "dialogue": "你总是这么有想法。", "narration": "",
                 "emotion": "embarrassed", "subtitle": "你总是这么有想法。"},
            ]
        },
        {
            "scene_id": "scene_3", "location": "office", "time_of_day": "evening",
            "lighting": "夕阳余晖", "mood": "紧张",
            "shots": [
                {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                 "duration_seconds": 3, "description": "王总走进办公室", "character": "boss_wang",
                 "action": "严肃地推门进来", "dialogue": "小明，客户对方案很不满意！", "narration": "",
                 "emotion": "angry", "subtitle": "小明，客户对方案很不满意！"},
                {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                 "duration_seconds": 2, "description": "小明紧张的表情", "character": "xiaoming",
                 "action": "紧张地站起来", "dialogue": "什么？我明明按需求做的...", "narration": "",
                 "emotion": "nervous", "subtitle": "什么？我明明按需求做的..."},
                {"shot_id": "shot_3", "shot_type": "wide_shot", "camera_movement": "static",
                 "duration_seconds": 3, "description": "三人对峙", "character": "boss_wang",
                 "action": "将文件摔在桌上", "dialogue": "需求已经变了，你不知道吗？明天早上之前改好！", "narration": "",
                 "emotion": "angry", "subtitle": "需求已经变了，你不知道吗？明天早上之前改好！"},
            ]
        },
        {
            "scene_id": "scene_4", "location": "apartment", "time_of_day": "night",
            "lighting": "台灯光", "mood": "温馨感人",
            "shots": [
                {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                 "duration_seconds": 3, "description": "小明在公寓加班", "character": "xiaoming",
                 "action": "疲惫地盯着屏幕", "dialogue": "", "narration": "夜深了，小明还在改方案。",
                 "emotion": "sad", "subtitle": "夜深了，小明还在改方案。"},
                {"shot_id": "shot_2", "shot_type": "medium_shot", "camera_movement": "static",
                 "duration_seconds": 3, "description": "小丽端着夜宵进来", "character": "xiaoli",
                 "action": "轻轻推门，端着夜宵", "dialogue": "还没休息？我给你带了宵夜。", "narration": "",
                 "emotion": "calm", "subtitle": "还没休息？我给你带了宵夜。"},
                {"shot_id": "shot_3", "shot_type": "close_up", "camera_movement": "static",
                 "duration_seconds": 2, "description": "小明感动地看着小丽", "character": "xiaoming",
                 "action": "感动地看着小丽", "dialogue": "谢谢你，小丽。有你在真好。", "narration": "",
                 "emotion": "happy", "subtitle": "谢谢你，小丽。有你在真好。"},
            ]
        },
        {
            "scene_id": "scene_5", "location": "park", "time_of_day": "morning",
            "lighting": "阳光明媚", "mood": "充满希望",
            "shots": [
                {"shot_id": "shot_1", "shot_type": "wide_shot", "camera_movement": "pan_left",
                 "duration_seconds": 3, "description": "公园里晨跑", "character": "xiaoming",
                 "action": "在公园晨跑", "dialogue": "", "narration": "改完方案的第二天，小明决定出门透透气。",
                 "emotion": "calm", "subtitle": "改完方案的第二天，小明决定出门透透气。"},
                {"shot_id": "shot_2", "shot_type": "medium_shot", "camera_movement": "static",
                 "duration_seconds": 3, "description": "偶遇小丽", "character": "xiaoli",
                 "action": "惊喜地挥手", "dialogue": "小明！好巧啊！", "narration": "",
                 "emotion": "surprised", "subtitle": "小明！好巧啊！"},
                {"shot_id": "shot_3", "shot_type": "medium_shot", "camera_movement": "static",
                 "duration_seconds": 3, "description": "两人并肩走在公园", "character": "xiaoming",
                 "action": "并肩散步，相视而笑", "dialogue": "小丽，昨晚的方案客户通过了！", "narration": "",
                 "emotion": "happy", "subtitle": "小丽，昨晚的方案客户通过了！"},
            ]
        },
        {
            "scene_id": "scene_6", "location": "office", "time_of_day": "morning",
            "lighting": "自然光", "mood": "紧张期待",
            "shots": [
                {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                 "duration_seconds": 3, "description": "王总宣布消息", "character": "boss_wang",
                 "action": "站在会议室前方", "dialogue": "告诉大家一个好消息——", "narration": "",
                 "emotion": "calm", "subtitle": "告诉大家一个好消息——"},
                {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                 "duration_seconds": 2, "description": "小明和小丽紧张对视", "character": "xiaoming",
                 "action": "紧张地握紧拳头", "dialogue": "", "narration": "",
                 "emotion": "nervous", "subtitle": ""},
                {"shot_id": "shot_3", "shot_type": "wide_shot", "camera_movement": "dolly_in",
                 "duration_seconds": 3, "description": "王总微笑", "character": "boss_wang",
                 "action": "露出罕见的微笑", "dialogue": "客户非常满意！小明、小丽，你们做到了！", "narration": "",
                 "emotion": "happy", "subtitle": "客户非常满意！小明、小丽，你们做到了！"},
            ]
        },
    ]
    
    return {
        "episode": episode_num,
        "title": "第一集：初遇",
        "duration_estimate": "3-5分钟",
        "scenes": scene_templates[:num_scenes],
        "characters_used": ["xiaoming", "xiaoli", "boss_wang"],
        "next_episode_hook": "小明和小丽的项目获得了成功，但新的挑战正在等着他们..."
    }
