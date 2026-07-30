#!/usr/bin/env python
"""GPT-SoVITS 推理 CLI。

最简用法（参考音频的转写文本从 .list 自动查找）:
    uv run python infer.py -t "要合成的文本" -o out.wav

指定模型:
    uv run python infer.py \\
        --gpt GPT_weights_v2Pro/girlish_voice-e20.ckpt \\
        --sovits SoVITS_weights_v2Pro/girlish_voice_e20_s120.pth \\
        --ref data/voices/girlish_voice_sliced/xxx.wav \\
        --ref-text "参考音频里说的话" \\
        -t "要合成的文本" -o out.wav

从文件读取待合成文本:
    uv run python infer.py -t @script.txt -o out.wav
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_GPT = "GPT_weights_v2Pro/girlish-e30.ckpt"
DEFAULT_SOVITS = "SoVITS_weights_v2Pro/girlish_e15_s330.pth"
DEFAULT_REF = "data/voices/girlish_sliced/ref-vocal.wav_0004307200_0004589760.wav"
DEFAULT_LIST = "data/voices/girlish.list"

LANG_CHOICES = ["中文", "英文", "日文", "粤语", "韩文", "中英混合", "多语种混合"]

# inference_webui.py:872-881 用这些中文字面量做分支判断（infer.py 强制 language=zh_CN）
CUT_CHOICES = [
    "不切",
    "凑四句一切",
    "凑50字一切",
    "按中文句号。切",
    "按英文句号.切",
    "按标点符号切",
]


def lookup_ref_text(ref_wav: str, list_file: str) -> str | None:
    """从 .list 标注文件里按文件名反查参考音频的转写文本。"""
    if not os.path.exists(list_file):
        return None
    target = os.path.basename(ref_wav)
    with open(list_file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) == 4 and os.path.basename(parts[0]) == target:
                return parts[3].strip()
    return None


def read_text_arg(value: str) -> str:
    """支持 `@path` 形式从文件读取文本。"""
    if value.startswith("@"):
        path = value[1:]
        if not os.path.exists(path):
            sys.exit(f"error: text file not found: {path}")
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return value


def main() -> None:
    p = argparse.ArgumentParser(
        description="GPT-SoVITS TTS inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-t", "--text", required=True, help="要合成的文本，或 @文件路径")
    p.add_argument("-o", "--output", required=True, help="输出 wav 路径")
    p.add_argument("--gpt", default=DEFAULT_GPT, help=f"GPT 权重 (默认 {DEFAULT_GPT})")
    p.add_argument(
        "--sovits", default=DEFAULT_SOVITS, help=f"SoVITS 权重 (默认 {DEFAULT_SOVITS})"
    )
    p.add_argument("--ref", default=DEFAULT_REF, help="参考音频 wav")
    p.add_argument(
        "--ref-text", default=None, help="参考音频的转写文本 (默认从 --list 反查)"
    )
    p.add_argument(
        "--list",
        default=DEFAULT_LIST,
        help=f"用于反查 ref-text 的标注文件 (默认 {DEFAULT_LIST})",
    )
    p.add_argument("--lang", default="中文", choices=LANG_CHOICES, help="目标文本语言")
    p.add_argument(
        "--ref-lang", default="中文", choices=LANG_CHOICES, help="参考音频语言"
    )
    p.add_argument(
        "--cut",
        default="按中文句号。切",
        choices=CUT_CHOICES,
        help="长文本切分方式。AR 生成上限 1500 步（t2s_model.py:533），"
        "整段长文本不切会被静默截断（默认 按中文句号。切）",
    )
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--speed", type=float, default=1.0, help="语速倍率")
    p.add_argument("-q", "--quiet", action="store_true", help="抑制上游库的冗余输出")
    args = p.parse_args()

    target_text = read_text_arg(args.text)
    if not target_text:
        sys.exit("error: target text is empty")

    for label, path in [("gpt", args.gpt), ("sovits", args.sovits), ("ref", args.ref)]:
        if not os.path.exists(path):
            sys.exit(f"error: {label} not found: {path}")

    ref_text = args.ref_text or lookup_ref_text(args.ref, args.list)
    if not ref_text:
        sys.exit(
            f"error: 无法确定参考音频的转写文本。\n"
            f"  请用 --ref-text 显式指定，或确认 {args.list} 中包含 {os.path.basename(args.ref)}"
        )

    # 必须在导入 inference_webui 之前设置：version 决定模型分支，
    # language=zh_CN 决定 dict_language 的键是中文字面量而非 i18n 翻译结果
    os.environ.setdefault("version", "v2Pro")
    os.environ["language"] = "zh_CN"
    now_dir = os.getcwd()
    sys.path.insert(0, now_dir)
    sys.path.insert(0, os.path.join(now_dir, "GPT_SoVITS"))

    if args.quiet:
        import logging

        logging.disable(logging.WARNING)

    import soundfile as sf

    from GPT_SoVITS.inference_webui import (
        change_gpt_weights,
        change_sovits_weights,
        get_tts_wav,
    )

    print(f"GPT    : {args.gpt}")
    print(f"SoVITS : {args.sovits}")
    print(f"Ref    : {os.path.basename(args.ref)}")
    print(f"RefText: {ref_text[:50]}{'...' if len(ref_text) > 50 else ''}")
    print(f"Text   : {len(target_text)} 字")
    print(f"Cut    : {args.cut}")

    change_gpt_weights(gpt_path=args.gpt)
    change_sovits_weights(sovits_path=args.sovits)

    result = list(
        get_tts_wav(
            ref_wav_path=args.ref,
            prompt_text=ref_text,
            prompt_language=args.ref_lang,
            text=target_text,
            text_language=args.lang,
            how_to_cut=args.cut,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            speed=args.speed,
        )
    )
    if not result:
        sys.exit("error: 合成失败，未产生音频")

    sr, audio = result[-1]
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    sf.write(args.output, audio, sr)
    print(f"\n-> {args.output}  ({len(audio) / sr:.1f}s @ {sr}Hz)")


if __name__ == "__main__":
    main()
