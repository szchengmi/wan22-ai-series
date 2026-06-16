"""
Step 5: 配音生成 (ChatTTS / edge-tts)
===================================
为每个镜头的对话和旁白生成语音。
优先 ChatTTS，失败降级 edge-tts。
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    log, save_json, load_json,
    EPISODE_NUM, get_dirs, VOICE_PARAMS, EMOTION_SPEED,
)


def generate_with_chattts(text, output_path, speed=1.0, temp=0.3, top_p=0.7, top_k=20):
    """用 ChatTTS 生成"""
    import ChatTTS
    import torch
    import soundfile as sf

    chat = ChatTTS.Chat()
    chat.load(compile=False)  # Kaggle 不支持 compile

    wavs = chat.infer([text], speed=speed, temperature=temp, top_p=top_p, top_k=top_k)
    sf.write(output_path, wavs[0], 24000)
    return True


def generate_with_edge_tts(text, output_path, voice="zh-CN-XiaoxiaoNeural", rate=0):
    """用 edge-tts 生成（备用）"""
    import asyncio
    import edge_tts

    async def _generate():
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_path)

    asyncio.run(_generate())
    return True


def main(storyboard=None):
    log("=" * 50)
    log("Step 5: 配音生成")
    log("=" * 50)

    dirs = get_dirs(EPISODE_NUM)
    total = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))

    count = 0
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            ep = storyboard.get("episode", 1)
            out = f"{dirs['audio']}/ep{ep:02d}_{scene['scene_id']}_{sid}.mp3"

            if os.path.exists(out) and os.path.getsize(out) > 10000:
                log(f"  [{count}/{total}] {sid} 跳过(已存在)")
                continue

            text = shot.get("dialogue") or shot.get("narration") or shot.get("subtitle", "")
            if not text:
                log(f"  [{count}/{total}] {sid} 无对话/旁白")
                continue

            char = shot.get("character", "narrator")
            emotion = shot.get("emotion", "calm")
            vp = VOICE_PARAMS.get(char, VOICE_PARAMS["narrator"])
            speed_adj = EMOTION_SPEED.get(emotion, 1.0)
            speed = vp["speed"] * speed_adj

            success = False
            try:
                generate_with_chattts(text, out, speed=speed, temp=vp["temp"],
                                       top_p=vp["top_p"], top_k=vp["top_k"])
                success = True
            except Exception as e:
                log(f"  [{count}/{total}] ChatTTS 失败: {e}")

            if not success:
                try:
                    generate_with_edge_tts(text, out)
                    success = True
                except Exception as e:
                    log(f"  [{count}/{total}] edge-tts 也失败: {e}")

            if success:
                sz = os.path.getsize(out) / 1024
                log(f"  [{count}/{total}] {sid} ✓ ({sz:.0f}KB, {char}, {emotion})")
            else:
                log(f"  [{count}/{total}] {sid} ❌ 所有 TTS 失败")

    log("配音生成完成")


if __name__ == "__main__":
    dirs = get_dirs(EPISODE_NUM)
    storyboard = load_json(f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_storyboard.json")
    main(storyboard)
