"""
Step 2: 分镜生成
================
将剧本 JSON 转换为 SD 分镜 JSON，包含每个镜头的 prompt、seed、尺寸。
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    log, save_json, load_json,
    EPISODE_NUM, CHARACTER_PROMPTS, SCENE_PROMPTS,
    EMOTION_ENHANCE, SHOT_PARAMS, get_dirs,
)


def main():
    log("=" * 50)
    log("Step 2: 分镜生成")
    log("=" * 50)

    dirs = get_dirs(EPISODE_NUM)
    script_path = f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_script.json"
    script_data = load_json(script_path)

    storyboard = {
        "episode": script_data.get("episode", 1),
        "title": script_data.get("title", ""),
        "characters": {},
        "scenes": []
    }
    characters_used = set()

    for scene in script_data.get("scenes", []):
        scene_data = {
            "scene_id": scene["scene_id"],
            "location": scene["location"],
            "time_of_day": scene["time_of_day"],
            "lighting": scene["lighting"],
            "mood": scene["mood"],
            "shots": []
        }
        for shot in scene.get("shots", []):
            char = shot.get("character", "none")
            shot_type = shot.get("shot_type", "medium_shot")
            emotion = shot.get("emotion", "calm")
            if char != "none":
                characters_used.add(char)
            params = SHOT_PARAMS.get(shot_type, SHOT_PARAMS["medium_shot"])

            # 构建 SD prompt
            pp = [params["prefix"]]
            if char != "none" and char in CHARACTER_PROMPTS:
                pp.append(CHARACTER_PROMPTS[char]["base_prompt"])
            pp.append(shot.get("action", ""))
            if emotion in EMOTION_ENHANCE:
                pp.append(EMOTION_ENHANCE[emotion])
            sp = SCENE_PROMPTS.get(scene["location"], "")
            if sp:
                pp.append(sp)
            pp.append(f"{scene['time_of_day']}, {scene['lighting']}")
            pp.append(shot.get("description", ""))
            pp.append("masterpiece, best quality, detailed, anime style")

            neg = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, cropped, worst quality, low quality, jpeg artifacts, blurry"
            if char != "none" and char in CHARACTER_PROMPTS:
                neg += ", " + CHARACTER_PROMPTS[char]["negative_prompt"]

            scene_data["shots"].append({
                "shot_id": shot["shot_id"],
                "shot_type": shot_type,
                "camera_movement": shot.get("camera_movement", "static"),
                "duration_seconds": shot.get("duration_seconds", 3),
                "width": params["w"],
                "height": params["h"],
                "prompt": ", ".join([p for p in pp if p]),
                "negative_prompt": neg,
                "seed": CHARACTER_PROMPTS.get(char, {}).get("seed", -1),
                "character": char,
                "dialogue": shot.get("dialogue", ""),
                "narration": shot.get("narration", ""),
                "subtitle": shot.get("subtitle", ""),
                "description": shot.get("description", ""),
                "action": shot.get("action", ""),
                "emotion": emotion,
                "steps": 20,  # 默认步数，pipeline 可覆盖
                "guidance": 7.5,
            })
        storyboard["scenes"].append(scene_data)

    for cid in characters_used:
        storyboard["characters"][cid] = CHARACTER_PROMPTS[cid]

    save_json(storyboard, f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_storyboard.json")
    total = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))
    log(f"分镜完成: {len(storyboard['scenes'])}场景 | {total}镜头")
    return storyboard


if __name__ == "__main__":
    main()
