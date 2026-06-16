"""
Step 6: 剪辑合成 (FFmpeg + MoviePy)
====================================
拼接所有视频片段 + 配音 + 字幕 → 最终 MP4。
"""

import os
import sys
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    log, load_json, save_json, run_cmd,
    EPISODE_NUM, get_dirs, seconds_to_srt_time,
)


def _get_video_duration(path):
    """获取视频时长（秒）"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except:
        return 3.0


def _get_audio_duration(path):
    """获取音频时长（秒）"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except:
        return 0.0


def _create_srt(subtitles, output_path):
    """创建 SRT 字幕文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(subtitles, 1):
            f.write(f"{i}\n")
            f.write(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}\n")
            f.write(f"{text}\n\n")


def _concat_videos_ffmpeg(video_list, output_path):
    """用 FFmpeg concat demuxer 拼接视频"""
    concat_file = output_path.replace(".mp4", "_concat.txt")
    with open(concat_file, "w") as f:
        for v in video_list:
            f.write(f"file '{v}'\n")

    run_cmd(
        f'ffmpeg -y -f concat -safe 0 -i "{concat_file}" '
        f'-c:v libx264 -preset fast -crf 23 '
        f'-movflags +faststart "{output_path}" 2>/dev/null',
        timeout=600
    )
    if os.path.exists(concat_file):
        os.remove(concat_file)


def main(storyboard=None, script_data=None):
    log("=" * 50)
    log("Step 6: 剪辑合成")
    log("=" * 50)

    dirs = get_dirs(EPISODE_NUM)
    ep = EPISODE_NUM

    # 收集视频和音频片段
    video_segments = []
    audio_segments = []
    subtitles = []
    current_time = 0.0

    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            sid = shot["shot_id"]
            vid = f"{dirs['videos']}/ep{ep:02d}_{scene['scene_id']}_{sid}.mp4"
            aud = f"{dirs['audio']}/ep{ep:02d}_{scene['scene_id']}_{sid}.mp3"

            if not os.path.exists(vid):
                log(f"  跳过 {sid}: 无视频")
                continue

            dur = _get_video_duration(vid)
            video_segments.append(vid)

            if os.path.exists(aud):
                aud_dur = _get_audio_duration(aud)
                audio_segments.append((aud, current_time, aud_dur))

            # 字幕
            text = shot.get("subtitle") or shot.get("dialogue") or shot.get("narration", "")
            if text:
                subtitles.append((current_time, current_time + dur, text))

            current_time += dur

    if not video_segments:
        log("❌ 没有视频片段可拼接")
        return

    log(f"视频片段: {len(video_segments)}")
    log(f"音频片段: {len(audio_segments)}")
    log(f"字幕: {len(subtitles)}")
    log(f"预计时长: {current_time:.1f}秒")

    # 1. 拼接视频
    concat_vid = f"{dirs['final']}/episode_{ep:02d}_concat.mp4"
    log("拼接视频...")
    _concat_videos_ffmpeg(video_segments, concat_vid)

    # 2. 混合音频
    final_output = f"{dirs['final']}/episode_{ep:02d}_final.mp4"

    if audio_segments:
        log("混合音频...")
        # 构建 ffmpeg filter_complex
        inputs = [f'-i "{concat_vid}"']
        filter_parts = []
        for i, (aud_path, start, dur) in enumerate(audio_segments):
            inputs.append(f'-i "{aud_path}"')
            filter_parts.append(
                f"[{i + 1}:a]adelay={int(start * 1000)}|{int(start * 1000)}[a{i}]"
            )

        mix_inputs = "".join(f"[a{i}]" for i in range(len(audio_segments)))
        n = len(audio_segments)
        filter_str = ";".join(filter_parts) + f";{mix_inputs}amix=inputs={n}:duration=first[aout]"

        cmd = f'ffmpeg -y {" ".join(inputs)} -filter_complex "{filter_str}" ' \
              f'-map 0:v -map "[aout]" -c:v copy -c:a aac -b:a 128k ' \
              f'-shortest "{final_output}" 2>/dev/null'
        run_cmd(cmd, timeout=600)
    else:
        log("无音频，直接复制视频")
        import shutil
        shutil.copy2(concat_vid, final_output)

    # 3. 烧录字幕
    if subtitles and os.path.exists(final_output):
        log("烧录字幕...")
        srt_path = f"{dirs['final']}/episode_{ep:02d}.srt"
        _create_srt(subtitles, srt_path)

        # 用 ffmpeg 烧录（需要 libass）
        sub_vid = final_output.replace(".mp4", "_sub.mp4")
        result = run_cmd(
            f'ffmpeg -y -i "{final_output}" -vf "subtitles={srt_path}" '
            f'-c:v libx264 -preset fast -crf 23 -c:a copy '
            f'"{sub_vid}" 2>/dev/null',
            timeout=600
        )
        if os.path.exists(sub_vid) and os.path.getsize(sub_vid) > 100000:
            os.rename(sub_vid, final_output)
            log("字幕烧录完成")
        else:
            log("字幕烧录失败（可能缺少 libass），保留无字幕版本")

    # 4. 输出信息
    if os.path.exists(final_output):
        size_mb = os.path.getsize(final_output) / 1e6
        log(f"✅ 最终输出: {final_output} ({size_mb:.1f}MB)")
    else:
        log("❌ 最终输出不存在")

    return final_output


if __name__ == "__main__":
    dirs = get_dirs(EPISODE_NUM)
    storyboard = load_json(f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_storyboard.json")
    script_path = f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_script.json"
    script_data = load_json(script_path) if os.path.exists(script_path) else None
    main(storyboard, script_data)
