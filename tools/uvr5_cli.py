#!/usr/bin/env python
"""UVR5 人声分离 / 降噪的无头封装（上游只提供 Gradio WebUI）。

单文件输入 -> 单文件输出（只要人声轨）。逻辑对齐 tools/uvr5/webui.py:45-125，
去掉 Gradio 和批处理，改成可脚本调用。

用法:
    python tools/uvr5_cli.py -i in.wav -o vocal.wav --model HP5_only_main_vocal
    python tools/uvr5_cli.py -i in.wav -o vocal.wav --model VR-DeEchoDeReverb
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

UVR5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uvr5")
WEIGHT_ROOT = os.path.join(UVR5_DIR, "uvr5_weights")

# vr.py / bsroformer.py 内部用 `from lib.lib_v5 import ...` 这类平铺 import，
# 必须把 tools/uvr5 放进 sys.path 才能导入
sys.path.insert(0, UVR5_DIR)


def to_44k_stereo(src: str, dst: str) -> None:
    """UVR5 的 4band_v2 参数假定 44.1kHz 立体声输入（webui.py:87-99 同样先转码）。"""
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            src,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "2",
            "-ar",
            "44100",
            dst,
        ],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="UVR5 人声分离（无头）")
    ap.add_argument("-i", "--input", required=True, help="输入音频")
    ap.add_argument("-o", "--output", required=True, help="输出人声 wav")
    ap.add_argument(
        "--model",
        default="HP5_only_main_vocal",
        help="uvr5_weights 下的模型名（不含后缀）",
    )
    ap.add_argument("--agg", type=int, default=10, help="人声提取激进程度 0-20")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--half", default="True", choices=["True", "False"])
    args = ap.parse_args()

    is_half = args.half == "True"
    name = args.model

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，回退 CPU", file=sys.stderr)
        args.device = "cpu"
        is_half = False

    if "roformer" in name.lower():
        from bsroformer import Roformer_Loader

        pre_fun = Roformer_Loader(
            model_path=os.path.join(WEIGHT_ROOT, name + ".ckpt"),
            config_path=os.path.join(WEIGHT_ROOT, name + ".yaml"),
            device=args.device,
            is_half=is_half,
        )
    elif name == "onnx_dereverb_By_FoxJoy":
        from mdxnet import MDXNetDereverb

        pre_fun = MDXNetDereverb(15)
    else:
        from vr import AudioPre, AudioPreDeEcho

        func = AudioPre if "DeEcho" not in name else AudioPreDeEcho
        ckpt = os.path.join(WEIGHT_ROOT, name + ".pth")
        if not os.path.exists(ckpt):
            sys.exit(f"模型不存在: {ckpt}\n  可用: {sorted(os.listdir(WEIGHT_ROOT))}")
        pre_fun = func(
            agg=args.agg, model_path=ckpt, device=args.device, is_half=is_half
        )

    tmpdir = tempfile.mkdtemp(prefix="uvr5_")
    try:
        src = os.path.join(tmpdir, "in.wav")
        to_44k_stereo(args.input, src)

        vocal_dir = os.path.join(tmpdir, "vocal")
        ins_dir = os.path.join(tmpdir, "ins")
        # is_hp3=True 时上游会交换人声/伴奏的输出命名
        pre_fun._path_audio_(src, ins_dir, vocal_dir, "wav", "HP3" in name)

        produced = [f for f in os.listdir(vocal_dir) if f.endswith(".wav")]
        if not produced:
            sys.exit(f"未产生人声输出于 {vocal_dir}")

        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        shutil.move(os.path.join(vocal_dir, produced[0]), args.output)
        print(f"-> {args.output}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        try:
            if name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except AttributeError as e:  # 释放显存的 best-effort 清理
            print(f"清理模型引用时跳过: {e}", file=sys.stderr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
