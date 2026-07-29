#!/usr/bin/env python
"""GPT-SoVITS 微调全流程（切片 → ASR → 特征提取 → S1/S2 训练）。

最简用法:
    uv run python train.py --ref data/voices/ref.wav --name myvoice

只跑部分阶段:
    uv run python train.py --ref ... --name myvoice --stages slice,asr
    uv run python train.py --ref ... --name myvoice --stages s2

带增广（把片段压进 bucket0 并做变速，绕开 min_num=100 复制机制）:
    uv run python train.py --ref ... --name myvoice --augment

日志同时写到 logs/<name>/train.log。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 常量

ALL_STAGES = ["slice", "asr", "text", "ssl", "sv", "semantic", "s1", "s2"]

PRETRAIN = {
    "v2Pro": {
        "s2G": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
        "s2D": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth",
        "s2_config": "GPT_SoVITS/configs/s2v2Pro.json",
    },
    "v2ProPlus": {
        "s2G": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
        "s2D": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth",
        "s2_config": "GPT_SoVITS/configs/s2v2ProPlus.json",
    },
}
S1_CKPT = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
S1_CONFIG = "GPT_SoVITS/configs/s1longer-v2.yaml"
BERT_DIR = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
CNHUBERT_DIR = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
SV_CKPT = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
G2PW_DIR = "GPT_SoVITS/text/G2PWModel"

# DistributedBucketSampler 的分桶边界（frames），hop_length=640 @ 32kHz → 1 frame = 20ms
BUCKET_BOUNDS = [
    32,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
    1900,
]
HOP, SR = 640, 32000
MIN_NUM = 100  # data_utils.py:55，低于此值样本会被静默复制

log = logging.getLogger("train")


# ---------------------------------------------------------------- 日志


def setup_logging(logfile: str, verbose: bool) -> None:
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    )
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(sh)


def banner(text: str) -> None:
    log.info("")
    log.info("=" * 70)
    log.info(f"  {text}")
    log.info("=" * 70)


class Fail(RuntimeError):
    """阶段失败，带可读的排查提示。"""


# ---------------------------------------------------------------- 子进程


def run(cmd: list[str], env: dict[str, str] | None = None, stage: str = "") -> None:
    """跑子进程，逐行转发输出到日志（DEBUG 级，-v 时可见）。失败抛 Fail。"""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
        for k, v in sorted(env.items()):
            log.debug(f"    env {k}={v}")
    log.debug(f"    cmd {' '.join(cmd)}")

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        log.debug(f"    | {line}")
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
    proc.wait()
    dt = time.time() - t0

    if proc.returncode != 0:
        log.error(
            f"  [{stage}] 子进程退出码 {proc.returncode}，最后 {len(tail)} 行输出："
        )
        for line in tail:
            log.error(f"    | {line}")
        raise Fail(f"{stage} 失败（exit {proc.returncode}）")
    log.debug(f"    done in {dt:.1f}s")


# ---------------------------------------------------------------- 辅助


def count_lines(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def dir_count(path: str, suffix: str = "") -> int:
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path) if f.endswith(suffix))


def wav_durations(d: str) -> list[float]:
    """按文件大小估算 16bit 单声道 wav 时长，避免逐个解码。"""
    out = []
    for f in os.listdir(d):
        if f.endswith(".wav"):
            out.append(os.path.getsize(os.path.join(d, f)) / (SR * 2))
    return sorted(out)


def report_buckets(wav_dir: str, batch_size: int) -> None:
    """打印分桶分布，提示稀疏桶和 batch_size 是否匹配。"""
    if not os.path.isdir(wav_dir):
        return
    frames = [
        os.path.getsize(os.path.join(wav_dir, f)) // (2 * HOP)
        for f in os.listdir(wav_dir)
        if f.endswith(".wav")
    ]
    if not frames:
        return
    buckets: dict[int, int] = {}
    out_of_range = 0
    for fr in frames:
        for i in range(len(BUCKET_BOUNDS) - 1):
            if BUCKET_BOUNDS[i] < fr <= BUCKET_BOUNDS[i + 1]:
                buckets[i] = buckets.get(i, 0) + 1
                break
        else:
            out_of_range += 1

    log.info(f"  分桶分布（batch_size={batch_size}）:")
    for i in sorted(buckets):
        lo = BUCKET_BOUNDS[i] * HOP / SR
        hi = BUCKET_BOUNDS[i + 1] * HOP / SR
        n = buckets[i]
        pad = (batch_size - n % batch_size) % batch_size
        flag = ""
        if n < batch_size:
            flag = f"  <- 不足一个 batch，将补齐 {pad} 条重复样本"
        elif pad:
            flag = f"  (补齐 {pad} 条)"
        log.info(f"    bucket{i:<2} {lo:5.2f}-{hi:5.2f}s : {n:4d} 条{flag}")
    if out_of_range:
        log.warning(
            f"    {out_of_range} 条落在边界外（<0.64s 或 >38s），会被采样器丢弃"
        )


# ---------------------------------------------------------------- 配置


@dataclass
class Cfg:
    ref: str
    name: str
    version: str = "v2Pro"
    lang: str = "zh"
    gpu: str = "0"
    augment: bool = False

    # 切片
    threshold: int = -30
    min_length: int = 4000
    min_interval: int = 600
    hop_size: int = 10
    max_sil_kept: int = 800

    # 训练
    s1_batch: int = 6
    s1_epochs: int = 12
    s2_batch: int = 8
    s2_epochs: int = 10
    save_every: int = 2

    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))

    @property
    def slice_dir(self) -> str:
        return f"data/voices/{self.name}_sliced"

    @property
    def asr_dir(self) -> str:
        return f"data/voices/{self.name}_asr"

    @property
    def list_file(self) -> str:
        return f"data/voices/{self.name}.list"

    @property
    def opt_dir(self) -> str:
        return f"logs/{self.name}"

    @property
    def s2_log_dir(self) -> str:
        return f"{self.opt_dir}/logs_s2_{self.version}"

    @property
    def s1_log_dir(self) -> str:
        return f"{self.opt_dir}/logs_s1_{self.version}"

    @property
    def sovits_out(self) -> str:
        return f"SoVITS_weights_{self.version}"

    @property
    def gpt_out(self) -> str:
        return f"GPT_weights_{self.version}"


# ---------------------------------------------------------------- 预检


def preflight(cfg: Cfg) -> None:
    banner("预检")

    if not os.path.exists(cfg.ref):
        raise Fail(f"参考音频不存在: {cfg.ref}")
    size_mb = os.path.getsize(cfg.ref) / 1024 / 1024
    log.info(f"  参考音频 : {cfg.ref} ({size_mb:.1f} MB)")

    p = PRETRAIN[cfg.version]
    required = [
        ("S1 底模", S1_CKPT),
        ("S1 配置", S1_CONFIG),
        ("S2 生成器", p["s2G"]),
        ("S2 判别器", p["s2D"]),
        ("S2 配置", p["s2_config"]),
        ("BERT", BERT_DIR),
        ("CNHuBERT", CNHUBERT_DIR),
        ("SV 模型", SV_CKPT),
    ]
    missing = [(label, path) for label, path in required if not os.path.exists(path)]
    for label, path in required:
        mark = "OK " if os.path.exists(path) else "缺失"
        log.info(f"  [{mark}] {label:<10} {path}")
    if missing:
        raise Fail(
            "缺少预训练模型:\n"
            + "\n".join(f"    - {label}: {path}" for label, path in missing)
            + "\n  从 https://huggingface.co/lj1995/GPT-SoVITS 下载，参见 install.sh"
        )

    if cfg.lang == "zh" and not os.path.exists(G2PW_DIR):
        raise Fail(
            f"中文 TTS 需要 G2PW 模型: {G2PW_DIR}\n  下载 G2PWModel.zip 解压到 GPT_SoVITS/text/"
        )

    if not shutil.which("ffmpeg"):
        raise Fail(
            "未找到 ffmpeg，训练数据加载依赖它（tools/my_utils.load_audio 调用 ffmpeg CLI）"
        )

    try:
        import torch

        if torch.cuda.is_available():
            log.info(
                f"  GPU      : {torch.cuda.get_device_name(0)} "
                f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB)"
            )
        else:
            log.warning("  GPU      : 不可用，将用 CPU（极慢）")
    except ImportError:
        raise Fail(
            "torch 未安装，先执行 uv pip install torch --index-url ..."
        ) from None

    # 输出目录必须预建：logs_s* 缺失会让首次 save_checkpoint 崩溃，
    # *_weights_* 缺失会让 process_ckpt.savee() 静默失败
    for d in [
        cfg.opt_dir,
        cfg.s1_log_dir,
        cfg.s2_log_dir,
        cfg.sovits_out,
        cfg.gpt_out,
        "TEMP",
    ]:
        os.makedirs(d, exist_ok=True)
    log.debug(
        f"  已确保输出目录存在: {cfg.sovits_out}, {cfg.gpt_out}, {cfg.s1_log_dir}, {cfg.s2_log_dir}"
    )


# ---------------------------------------------------------------- 阶段：切片


def stage_slice(cfg: Cfg) -> None:
    banner("Stage 1/8: 音频切片")
    os.makedirs(cfg.slice_dir, exist_ok=True)
    for f in os.listdir(cfg.slice_dir):
        os.remove(os.path.join(cfg.slice_dir, f))

    log.info(
        f"  参数: threshold={cfg.threshold}dB min_length={cfg.min_length}ms "
        f"min_interval={cfg.min_interval}ms max_sil_kept={cfg.max_sil_kept}ms"
    )
    log.info("  提示: min_interval 越大越只切长停顿（句间），越小越容易切断句子中间")

    run(
        [
            sys.executable,
            "-s",
            "tools/slice_audio.py",
            cfg.ref,
            cfg.slice_dir,
            str(cfg.threshold),
            str(cfg.min_length),
            str(cfg.min_interval),
            str(cfg.hop_size),
            str(cfg.max_sil_kept),
            "0.9",
            "0.25",
            "0",
            "1",
        ],
        stage="slice",
    )

    n = dir_count(cfg.slice_dir, ".wav")
    if n == 0:
        raise Fail(
            "切片输出为空。可能 threshold 过低导致整段被判为静音，尝试调高（如 -25）"
        )
    ds = wav_durations(cfg.slice_dir)
    log.info(
        f"  切出 {n} 条，总时长 {sum(ds):.1f}s，"
        f"均值 {sum(ds) / len(ds):.1f}s，范围 {ds[0]:.1f}-{ds[-1]:.1f}s"
    )

    if cfg.augment:
        stage_augment(cfg)
        n = dir_count(cfg.slice_dir, ".wav")

    report_buckets(cfg.slice_dir, cfg.s2_batch)
    if n < MIN_NUM:
        log.warning(
            f"  样本数 {n} < {MIN_NUM}：data_utils.py 会把列表重复 "
            f"{max(2, int(MIN_NUM / n))} 次（wav_data_len 会变成 ~{n * max(2, int(MIN_NUM / n))}）"
        )
        log.warning("  配置的 epoch 数因此不是真实 epoch 数。可加 --augment 绕开")


# ---------------------------------------------------------------- 阶段：增广


def stage_augment(cfg: Cfg) -> None:
    """错位子窗口 + 变速，把片段压进 bucket0 (0.64-6.00s) 并扩充样本数。

    只做 time_stretch（不改音高，音色不变），不做 pitch shift / 加噪 —— 那些会破坏音色目标。
    """
    import librosa
    import numpy as np
    import soundfile as sf

    log.info("  增广: 错位子窗口 + 变速（目标 bucket0 = 0.64-6.00s）")
    win_min, win_max, overlap = 3.0, 5.5, 0.5
    speeds = [1.0, 0.95, 1.05]

    bases = sorted(
        os.path.join(cfg.slice_dir, f)
        for f in os.listdir(cfg.slice_dir)
        if f.endswith(".wav")
    )
    staging = "/tmp/_gsv_aug"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)

    n = 0
    for bi, path in enumerate(bases):
        y, sr = librosa.load(path, sr=SR, mono=True)
        dur = len(y) / sr
        if dur <= win_max:
            windows = [y] if dur >= 1.0 else []
        else:
            windows = []
            step = int(win_max * sr * (1 - overlap))
            pos = 0
            while pos + int(win_min * sr) <= len(y):
                seg = y[pos : pos + int(win_max * sr)]
                if len(seg) / sr >= win_min:
                    windows.append(seg)
                pos += step

        for wi, seg in enumerate(windows):
            for sp in speeds:
                s = seg if sp == 1.0 else librosa.effects.time_stretch(y=seg, rate=sp)
                d = len(s) / sr
                if not (0.7 <= d <= 5.9):  # 严格锁在 bucket0 内，留安全边界
                    continue
                peak = float(np.abs(s).max())
                if peak < 1e-4:
                    continue
                sf.write(
                    f"{staging}/aug_{bi:03d}_{wi:02d}_sp{int(sp * 100)}.wav",
                    s / peak * 0.9,
                    sr,
                )
                n += 1

    for f in os.listdir(cfg.slice_dir):
        os.remove(os.path.join(cfg.slice_dir, f))
    for f in os.listdir(staging):
        shutil.move(os.path.join(staging, f), os.path.join(cfg.slice_dir, f))
    shutil.rmtree(staging, ignore_errors=True)

    log.info(f"  {len(bases)} 条基础片段 -> {n} 条增广片段")
    if n == 0:
        raise Fail("增广输出为空，检查基础片段时长是否都 < 1s")


# ---------------------------------------------------------------- 阶段：ASR


def stage_asr(cfg: Cfg) -> None:
    banner("Stage 2/8: ASR 转写")
    if dir_count(cfg.slice_dir, ".wav") == 0:
        raise Fail(f"{cfg.slice_dir} 无切片，先跑 slice 阶段")

    os.makedirs(cfg.asr_dir, exist_ok=True)
    log.info(f"  输入 {dir_count(cfg.slice_dir, '.wav')} 条，语言 {cfg.lang}")
    log.info("  首次运行会下载 ASR 模型（Fun-ASR-Nano，需 transformers>=4.57）")

    try:
        run(
            [
                sys.executable,
                "-s",
                "tools/asr/funasr_asr.py",
                "-i",
                cfg.slice_dir,
                "-o",
                cfg.asr_dir,
                "-l",
                cfg.lang,
            ],
            stage="asr",
        )
    except Fail as e:
        raise Fail(
            f"{e}\n  常见原因: KeyError 'qwen3' -> transformers 版本过低，"
            f"执行 uv pip install 'transformers>=4.57,<5.0'"
        ) from None

    produced = [f for f in os.listdir(cfg.asr_dir) if f.endswith(".list")]
    if not produced:
        raise Fail(f"ASR 未产生 .list 文件于 {cfg.asr_dir}")
    raw = os.path.join(cfg.asr_dir, produced[0])

    # ASR 用输入目录名当 speaker，改成 cfg.name；同时丢掉空/过短转写
    kept, dropped = [], 0
    with open(raw, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) != 4:
                dropped += 1
                continue
            text = parts[3].strip()
            if len(text) < 4:
                dropped += 1
                log.debug(f"    丢弃过短转写: {os.path.basename(parts[0])} -> {text!r}")
                continue
            kept.append(f"{parts[0]}|{cfg.name}|{parts[2]}|{text}")

    if not kept:
        raise Fail("ASR 全部转写为空，检查音频是否有人声")
    with open(cfg.list_file, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")

    log.info(f"  保留 {len(kept)} 条，丢弃 {dropped} 条 -> {cfg.list_file}")
    for line in kept[:3]:
        p = line.split("|")
        log.info(f"    {os.path.basename(p[0])}  {p[3][:40]}")
    log.info("  建议人工校对转写文本，错误标注会直接影响音色学习")


# ---------------------------------------------------------------- 阶段：特征提取


def prep_env(cfg: Cfg) -> dict[str, str]:
    """prepare_datasets/ 下的脚本全部通过环境变量传参，没有 CLI。"""
    return {
        "inp_text": cfg.list_file,
        "inp_wav_dir": cfg.slice_dir,
        "exp_name": cfg.name,
        "opt_dir": cfg.opt_dir,
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": cfg.gpu,
        "is_half": "True",
    }


def stage_text(cfg: Cfg) -> None:
    banner("Stage 3/8: 文本 -> 音素 + BERT 特征")
    if not os.path.exists(cfg.list_file):
        raise Fail(f"标注文件不存在: {cfg.list_file}，先跑 asr 阶段")

    env = prep_env(cfg) | {"bert_pretrained_dir": BERT_DIR}
    run(
        [sys.executable, "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"], env, "text"
    )

    shard = f"{cfg.opt_dir}/2-name2text-0.txt"
    merged = f"{cfg.opt_dir}/2-name2text.txt"
    if not os.path.exists(shard):
        raise Fail(f"未生成 {shard}")
    shutil.move(shard, merged)

    n = count_lines(merged)
    total = count_lines(cfg.list_file)
    log.info(f"  音素条目 {n} / 标注 {total}")
    log.info(f"  BERT 特征 {dir_count(cfg.opt_dir + '/3-bert', '.pt')} 个")
    if n == 0:
        raise Fail("音素提取全部失败，检查语言代码与 G2PW 模型")
    if n < total:
        log.warning(
            f"  有 {total - n} 条未能提取音素（通常是不支持的语言代码或异常字符）"
        )


def stage_ssl(cfg: Cfg) -> None:
    banner("Stage 4/8: 音频 -> SSL 特征 + 32kHz 重采样")
    env = prep_env(cfg) | {"cnhubert_base_dir": CNHUBERT_DIR}
    run(
        [sys.executable, "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"],
        env,
        "ssl",
    )

    n4 = dir_count(f"{cfg.opt_dir}/4-cnhubert", ".pt")
    n5 = dir_count(f"{cfg.opt_dir}/5-wav32k", ".wav")
    log.info(f"  SSL 特征 {n4} 个，wav32k {n5} 个")
    if n4 == 0:
        raise Fail("SSL 提取失败，检查 CNHuBERT 模型与 ffmpeg")
    if n4 != n5:
        log.warning(
            f"  SSL({n4}) 与 wav32k({n5}) 数量不一致，可能有音频被峰值过滤或 NaN 过滤"
        )


def stage_sv(cfg: Cfg) -> None:
    banner("Stage 5/8: 音频 -> 说话人向量（v2Pro 系列必需）")
    if cfg.version not in ("v2Pro", "v2ProPlus"):
        log.info("  非 v2Pro 系列，跳过")
        return
    if dir_count(f"{cfg.opt_dir}/5-wav32k", ".wav") == 0:
        raise Fail("5-wav32k 为空，先跑 ssl 阶段（本阶段读取其输出）")

    env = prep_env(cfg) | {"sv_path": SV_CKPT}
    try:
        run(
            [sys.executable, "-s", "GPT_SoVITS/prepare_datasets/2-get-sv.py"], env, "sv"
        )
    except Fail as e:
        raise Fail(
            f"{e}\n  若报 libnppicc.so.12 缺失: torchaudio.load 走 torchcodec 需要 CUDA-12 NPP 库。"
            f"\n  本仓库已将该脚本改用 soundfile.read()，确认改动未被覆盖。"
        ) from None

    n7 = dir_count(f"{cfg.opt_dir}/7-sv_cn", ".pt")
    log.info(f"  说话人向量 {n7} 个")
    if n7 == 0:
        raise Fail("SV 提取全部失败")


def stage_semantic(cfg: Cfg) -> None:
    banner("Stage 6/8: SSL -> 语义 token")
    p = PRETRAIN[cfg.version]
    env = prep_env(cfg) | {"pretrained_s2G": p["s2G"], "s2config_path": p["s2_config"]}
    run(
        [sys.executable, "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"],
        env,
        "semantic",
    )

    shard = f"{cfg.opt_dir}/6-name2semantic-0.tsv"
    merged = f"{cfg.opt_dir}/6-name2semantic.tsv"
    if not os.path.exists(shard):
        raise Fail(f"未生成 {shard}")
    with open(shard, encoding="utf-8") as fi, open(merged, "w", encoding="utf-8") as fo:
        fo.write("item_name\tsemantic_audio\n")
        fo.write(fi.read())
    os.remove(shard)

    n = count_lines(merged) - 1
    log.info(f"  语义 token {n} 条")
    if n == 0:
        raise Fail("语义 token 提取全部失败")
    verify_alignment(cfg)


def verify_alignment(cfg: Cfg) -> None:
    """交叉校验各阶段产物计数。3-get-semantic.py 会静默丢条，必须显式检查。"""
    counts = {
        "2-name2text": count_lines(f"{cfg.opt_dir}/2-name2text.txt"),
        "4-cnhubert": dir_count(f"{cfg.opt_dir}/4-cnhubert", ".pt"),
        "5-wav32k": dir_count(f"{cfg.opt_dir}/5-wav32k", ".wav"),
        "6-name2semantic": count_lines(f"{cfg.opt_dir}/6-name2semantic.tsv") - 1,
    }
    if cfg.version in ("v2Pro", "v2ProPlus"):
        counts["7-sv_cn"] = dir_count(f"{cfg.opt_dir}/7-sv_cn", ".pt")

    log.info("  各阶段产物计数:")
    for k, v in counts.items():
        log.info(f"    {k:<18} {v}")

    if len(set(counts.values())) == 1:
        log.info("  全部对齐")
        return

    lo = min(counts.values())
    log.warning(f"  计数不一致，实际可用样本数取交集（约 {lo} 条）")

    # 找出 semantic 缺失的具体条目
    sem_path = f"{cfg.opt_dir}/6-name2semantic.tsv"
    txt_path = f"{cfg.opt_dir}/2-name2text.txt"
    if os.path.exists(sem_path) and os.path.exists(txt_path):
        with open(sem_path, encoding="utf-8") as f:
            sem = {
                line.split("\t")[0] for line in f.read().strip().split("\n")[1:] if line
            }
        with open(txt_path, encoding="utf-8") as f:
            txt = {line.split("\t")[0] for line in f.read().strip().split("\n") if line}
        for name in sorted(txt - sem):
            wav = f"{cfg.opt_dir}/5-wav32k/{name}"
            dur = os.path.getsize(wav) / (SR * 2) if os.path.exists(wav) else 0
            log.warning(f"    缺少 semantic: {name} ({dur:.2f}s)")


# ---------------------------------------------------------------- 阶段：训练


def stage_s1(cfg: Cfg) -> None:
    banner("Stage 7/8: GPT (S1) 训练")
    import yaml

    sem = f"{cfg.opt_dir}/6-name2semantic.tsv"
    pho = f"{cfg.opt_dir}/2-name2text.txt"
    for path in (sem, pho):
        if not os.path.exists(path):
            raise Fail(f"缺少训练数据 {path}，先跑特征提取阶段")

    with open(S1_CONFIG, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["train"].update(
        {
            "batch_size": cfg.s1_batch,
            "epochs": cfg.s1_epochs,
            "save_every_n_epoch": cfg.save_every,
            "if_save_every_weights": True,
            "if_save_latest": True,
            "if_dpo": False,
            "half_weights_save_dir": cfg.gpt_out,
            "exp_name": cfg.name,
        }
    )
    data["pretrained_s1"] = S1_CKPT
    data["train_semantic_path"] = sem
    data["train_phoneme_path"] = pho
    data["output_dir"] = cfg.s1_log_dir

    tmp = "TEMP/tmp_s1.yaml"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    log.info(
        f"  batch_size={cfg.s1_batch} epochs={cfg.s1_epochs} save_every={cfg.save_every}"
    )
    log.info(f"  底模 {S1_CKPT}")
    log.info(
        "  注意: lr_schedulers.py 把学习率硬锁为 0.002，yaml 里的 lr/warmup_steps/decay_steps 全部无效"
    )
    log.info(f"  输出 -> {cfg.gpt_out}/  (小权重), {cfg.s1_log_dir}/  (完整 ckpt)")

    before = set(os.listdir(cfg.gpt_out))
    run(
        [sys.executable, "-s", "GPT_SoVITS/s1_train.py", "--config_file", tmp],
        {"_CUDA_VISIBLE_DEVICES": cfg.gpu, "hz": "25hz"},
        "s1",
    )

    new = sorted(set(os.listdir(cfg.gpt_out)) - before)
    if not new:
        raise Fail(
            f"S1 训练未产生新权重于 {cfg.gpt_out}/（目录不存在时 savee() 会静默失败）"
        )
    log.info(
        f"  新增 {len(new)} 个权重: {', '.join(new[:5])}{' ...' if len(new) > 5 else ''}"
    )


def stage_s2(cfg: Cfg) -> None:
    banner("Stage 8/8: SoVITS (S2) 训练")
    p = PRETRAIN[cfg.version]
    with open(p["s2_config"], encoding="utf-8") as f:
        data = json.load(f)

    data["train"].update(
        {
            "batch_size": cfg.s2_batch,
            "epochs": cfg.s2_epochs,
            "save_every_epoch": cfg.save_every,
            "if_save_latest": False,
            "if_save_every_weights": True,
            "gpu_numbers": cfg.gpu,
            "grad_ckpt": False,
            "lora_rank": 0,
            "text_low_lr_rate": 0.4,
            "pretrained_s2G": p["s2G"],
            "pretrained_s2D": p["s2D"],
        }
    )
    data["model"]["version"] = cfg.version
    data["data"]["exp_dir"] = cfg.opt_dir
    data["s2_ckpt_dir"] = cfg.opt_dir
    data["save_weight_dir"] = cfg.sovits_out
    data["name"] = cfg.name
    data["version"] = cfg.version

    tmp = "TEMP/tmp_s2.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log.info(
        f"  batch_size={cfg.s2_batch} epochs={cfg.s2_epochs} save_every={cfg.save_every}"
    )
    log.info(f"  底模 G={p['s2G']}")
    log.info(f"       D={p['s2D']}")
    log.info(
        f"  注意: segment_size=20480 -> 每步只监督 {20480 / SR:.2f}s 波形，长音频不增加波形监督"
    )
    log.info(f"  输出 -> {cfg.sovits_out}/  (小权重), {cfg.s2_log_dir}/  (G/D ckpt)")

    before = set(os.listdir(cfg.sovits_out))
    run([sys.executable, "-s", "GPT_SoVITS/s2_train.py", "--config", tmp], None, "s2")

    new = sorted(set(os.listdir(cfg.sovits_out)) - before)
    if not new:
        raise Fail(
            f"S2 训练未产生新权重于 {cfg.sovits_out}/\n"
            f"  若 {cfg.s2_log_dir}/ 里有 G_*.pth 但此处为空，说明 savee() 静默失败了（目录缺失）"
        )
    log.info(f"  新增 {len(new)} 个权重: {', '.join(new)}")


# ---------------------------------------------------------------- 入口

STAGE_FN = {
    "slice": stage_slice,
    "asr": stage_asr,
    "text": stage_text,
    "ssl": stage_ssl,
    "sv": stage_sv,
    "semantic": stage_semantic,
    "s1": stage_s1,
    "s2": stage_s2,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GPT-SoVITS 微调全流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--ref", required=True, help="参考音频（长 wav，会被切片）")
    ap.add_argument("--name", required=True, help="实验名，决定所有输出路径")
    ap.add_argument(
        "--version", default="v2Pro", choices=list(PRETRAIN), help="模型版本"
    )
    ap.add_argument(
        "--lang", default="zh", choices=["zh", "en", "ja", "ko", "yue"], help="语言"
    )
    ap.add_argument("--gpu", default="0", help="GPU 编号")
    ap.add_argument(
        "--augment",
        action="store_true",
        help="启用增广（压进 bucket0 + 变速），绕开 min_num=100 复制机制",
    )
    ap.add_argument(
        "--stages",
        default="all",
        help=f"要执行的阶段，逗号分隔。可选: {','.join(ALL_STAGES)}",
    )

    g = ap.add_argument_group("切片参数")
    g.add_argument("--threshold", type=int, default=-30, help="静音判定阈值 dB")
    g.add_argument("--min-length", type=int, default=4000, help="片段最短 ms")
    g.add_argument(
        "--min-interval",
        type=int,
        default=600,
        help="最短切割间隔 ms（越大越只切句间长停顿）",
    )
    g.add_argument("--max-sil-kept", type=int, default=800, help="片段边缘保留静音 ms")

    g = ap.add_argument_group("训练参数")
    g.add_argument("--s1-batch", type=int, default=6)
    g.add_argument("--s1-epochs", type=int, default=12)
    g.add_argument("--s2-batch", type=int, default=8)
    g.add_argument("--s2-epochs", type=int, default=10)
    g.add_argument(
        "--save-every", type=int, default=2, help="每几轮保存（小数据集建议 2）"
    )

    ap.add_argument("-v", "--verbose", action="store_true", help="打印子进程完整输出")
    args = ap.parse_args()

    stages = (
        ALL_STAGES
        if args.stages == "all"
        else [s.strip() for s in args.stages.split(",")]
    )
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        sys.exit(f"error: 未知阶段 {bad}，可选: {','.join(ALL_STAGES)}")

    cfg = Cfg(
        ref=args.ref,
        name=args.name,
        version=args.version,
        lang=args.lang,
        gpu=args.gpu,
        augment=args.augment,
        threshold=args.threshold,
        min_length=args.min_length,
        min_interval=args.min_interval,
        max_sil_kept=args.max_sil_kept,
        s1_batch=args.s1_batch,
        s1_epochs=args.s1_epochs,
        s2_batch=args.s2_batch,
        s2_epochs=args.s2_epochs,
        save_every=args.save_every,
        stages=stages,
    )

    os.makedirs(cfg.opt_dir, exist_ok=True)
    setup_logging(f"{cfg.opt_dir}/train.log", args.verbose)

    log.info(f"实验    : {cfg.name}")
    log.info(f"版本    : {cfg.version}")
    log.info(f"阶段    : {' -> '.join(stages)}")
    log.info(f"增广    : {'开' if cfg.augment else '关'}")
    log.info(f"日志    : {cfg.opt_dir}/train.log")

    t0 = time.time()
    try:
        preflight(cfg)
        for s in stages:
            ts = time.time()
            STAGE_FN[s](cfg)
            log.info(f"  [{s}] 耗时 {time.time() - ts:.1f}s")
    except Fail as e:
        log.error("")
        log.error(f"失败: {e}")
        log.error(f"完整日志: {cfg.opt_dir}/train.log（用 -v 可看子进程全部输出）")
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning(
            "\n已中断。重新运行会从最近的 checkpoint 继续（S1/S2 均支持断点续训）"
        )
        sys.exit(130)

    banner(f"完成，总耗时 {time.time() - t0:.0f}s")
    log.info(f"  GPT 权重    : {cfg.gpt_out}/")
    log.info(f"  SoVITS 权重 : {cfg.sovits_out}/")
    log.info("")
    log.info("  试听（挑不同 epoch 对比，效果非单调，不一定是最后一轮最好）:")
    log.info('    uv run python infer.py -t "测试文本" -o out.wav \\')
    log.info(f"      --gpt {cfg.gpt_out}/<ckpt> --sovits {cfg.sovits_out}/<pth> \\")
    log.info(f"      --ref <{cfg.slice_dir}/某片段> --list {cfg.list_file}")


if __name__ == "__main__":
    main()
