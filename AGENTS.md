# AGENTS.md — GPT-SoVITS

## Architecture

GPT-SoVITS is a **research-grade TTS/VC project**, not a pip package. There is no `setup.py`, `pyproject.toml`, or packaging.

**Pipeline**: Text → GPT (AR text→semantic tokens, `s1`) → SoVITS (VITS/BigVGAN semantic→audio, `s2`)

**6 model versions**: v1, v2, v3, v4, v2Pro, v2ProPlus — each with separate pretrained weights, weight directories, and configs. `config.py` maps versions to paths.

### Key directories

| Path | Purpose |
|---|---|
| `GPT_SoVITS/AR/` | Autoregressive T2S GPT model (modules, models, data, processing) |
| `GPT_SoVITS/module/` | SoVITS/VITS core (synthesizer, discriminators, loss, mel) |
| `GPT_SoVITS/TTS_infer_pack/` | Config-driven v2 inference pipeline (used by `api_v2.py`) |
| `GPT_SoVITS/text/` | Multilingual text frontends (zh/en/ja/ko/yue) |
| `GPT_SoVITS/prepare_datasets/` | Data prep: text→phonemes, audio→SSL, SSL→semantic tokens |
| `GPT_SoVITS/pretrained_models/` | Pretrained model weights (downloaded separately, not in repo) |
| `tools/uvr5/` | Voice separation (webui, mdxnet, bsroformer) |
| `tools/asr/` | ASR tools (Fun-ASR-Nano, Faster-Whisper, FunASR) |
| `tools/AP_BWE_main/` | Audio super-resolution (24k→48k) |
| `tools/i18n/` | Internationalization engine |
| `SoVITS_weights*/` | Trained SoVITS checkpoints (gitignored) |
| `GPT_weights*/` | Trained GPT checkpoints (gitignored) |

### Entrypoints

- **`webui.py`** — Master Gradio WebUI orchestrator. Sets `version=v2Pro`, injects site-packages, spawns subprocesses via `Popen` for each pipeline tab (UVR5, dataset prep, S1/S2 training, TTS inference).
- **`api_v2.py`** — Modern FastAPI TTS API (config-driven via `TTS_infer_pack/TTS.py`). Preferred over `api.py`.
- **`api.py`** — Legacy monolithic FastAPI TTS API. Tightly coupled, manually wires models.
- **`GPT_SoVITS/inference_webui.py`** — Standalone Gradio TTS inference UI (spawned by `webui.py`).
- **`GPT_SoVITS/inference_cli.py`** — CLI batch TTS inference.
- **`GPT_SoVITS/s1_train.py`** — GPT (S1) training via PyTorch Lightning.
- **`GPT_SoVITS/s2_train.py`** — SoVITS (S2) training v1/v2/v2Pro/v2ProPlus.
- **`GPT_SoVITS/s2_train_v3_lora.py`** — SoVITS (S2) training v3/v4 with LoRA.

### Repo-local helper scripts (not upstream)

Two wrappers exist so you don't have to hand-assemble the env-var-only `prepare_datasets/` scripts or remember the i18n/version env vars that inference needs.

#### `train.py` — end-to-end finetune driver

Pipeline: `[vocal] → [denoise] → slice → asr → text → ssl → sv → semantic → s1 → s2`.
`vocal` (UVR5) and `denoise` are opt-in (`OPT_IN_STAGES`) — they need `--vocal` / `--denoise`, or naming them in `--stages`.

```bash
# Full run with defaults (v2Pro, Chinese, GPU 0)
uv run python train.py --ref data/voices/ref.wav --name myvoice

# Recommended for short (~2-3 min) source audio: augmentation packs clips into
# bucket0 and pushes sample count past min_num=100, so epochs become real epochs
uv run python train.py --ref data/voices/ref.wav --name myvoice --augment

# Run a subset of stages (re-run after fixing ASR transcripts, retrain S2 only, ...)
uv run python train.py --ref data/voices/ref.wav --name myvoice --stages slice,asr
uv run python train.py --ref data/voices/ref.wav --name myvoice --stages text,ssl,sv,semantic
uv run python train.py --ref data/voices/ref.wav --name myvoice --stages s2

# Forward all subprocess output (essential when a stage fails)
uv run python train.py --ref data/voices/ref.wav --name myvoice -v

# v2ProPlus, non-Chinese, second GPU
uv run python train.py --ref data/voices/ref.wav --name myvoice \
    --version v2ProPlus --lang ja --gpu 1

# Tune slicing: larger --min-interval only cuts at long (sentence-level) pauses
uv run python train.py --ref data/voices/ref.wav --name myvoice \
    --threshold -30 --min-length 4000 --min-interval 600 --max-sil-kept 800

# Tune training. Defaults: s1 batch 6 / 2 epochs / save every 1,
# s2 batch 4 / 10 epochs / save every 5. On 6GB VRAM keep both batches <= 4.
uv run python train.py --ref data/voices/ref.wav --name myvoice \
    --s1-batch 4 --s1-epochs 30 --s1-save-every 5 \
    --s2-batch 4 --s2-epochs 20 --s2-save-every 5
```

Paths are all derived from `--name`: clips in `data/voices/<name>_sliced/`, annotations at `data/voices/<name>.list`, features under `logs/<name>/`, weights in `GPT_weights_<version>/` and `SoVITS_weights_<version>/`. Logs go to both stdout and `logs/<name>/train.log`.

What it checks for you: all 8 pretrained models + G2PW + ffmpeg + GPU before starting; pre-creates the weight dirs that `savee()` silently needs; prints the bucket histogram with per-bucket padding warnings; flags the `min_num=100` duplication; cross-checks counts across all five stage outputs and **names the specific dropped item**; verifies new weights actually landed after each training stage. Interrupting is safe — both S1 and S2 resume from their latest checkpoint.

#### `infer.py` — TTS CLI

```bash
# Minimal: model/ref default to the verified e20 pair, ref transcript is
# auto-looked-up from the .list file
uv run python infer.py -t "今天天气真好呀，我们一起出去玩吧。" -o out.wav

# Read target text from a file (for long-form synthesis)
uv run python infer.py -t @script.txt -o out.wav

# Pick specific checkpoints — do this to A/B different epochs
uv run python infer.py \
    --gpt GPT_weights_v2Pro/girlish_voice-e20.ckpt \
    --sovits SoVITS_weights_v2Pro/girlish_voice_e20_s120.pth \
    -t "测试文本" -o out.wav

# New voice: point at its own clip + .list
uv run python infer.py \
    --gpt GPT_weights_v2Pro/myvoice-e12.ckpt \
    --sovits SoVITS_weights_v2Pro/myvoice_e10_s180.pth \
    --ref data/voices/myvoice_sliced/aug_000_00_sp100.wav \
    --list data/voices/myvoice.list \
    -t "测试文本" -o out.wav

# Explicit ref transcript (skips .list lookup)
uv run python infer.py --ref path/to/ref.wav --ref-text "参考音频里说的话" \
    -t "测试文本" -o out.wav

# Sampling / speed / language; -q silences upstream library chatter
uv run python infer.py -t "测试文本" -o out.wav \
    --lang 中英混合 --top-k 15 --top-p 1.0 --temperature 1.0 --speed 1.0 -q
```

`--lang` / `--ref-lang` take the Chinese literals (`中文`, `英文`, `日文`, `粤语`, `韩文`, `中英混合`, `多语种混合`) because the script forces `language=zh_CN` — see the `dict_language` gotcha below.

### Packaging a trained voice for reuse

Ship a self-contained bundle so a finished voice can be re-used without re-running the pipeline. Layout that `infer.py` consumes directly:

```
girlish_v2Pro/
├── README.md
├── gpt/      girlish-e15.ckpt  girlish-e20.ckpt  girlish-e30.ckpt   (149M each)
├── sovits/   girlish_e15_s330.pth  girlish_e20_s440.pth             (129M each)
└── ref/      6 clips spanning different deliveries + ref.list
```

Only ship the checkpoints worth A/B-ing (skip under-trained early epochs); each GPT is 149M and each SoVITS 129M, so pruning matters. GPT and SoVITS are independent — any pair works.

**Ship several reference clips, not one.** The reference sets the *emotion and delivery*, not just the timbre, so a single narration-style clip locks every output into that register. Pick clips covering the registers you'll actually need (narration / gentle / laughing / questioning / …), name them with their duration, and keep every one inside the 3–10s limit.

`ref/ref.list` carries the transcripts in standard 4-field format (`path|speaker|LANG|text`) with **paths relative to the bundle root**, so `--list` auto-looks-up the reference text after extraction:

```bash
uv run python infer.py \
  --gpt    girlish_v2Pro/gpt/girlish-e30.ckpt \
  --sovits girlish_v2Pro/sovits/girlish_e15_s330.pth \
  --ref    girlish_v2Pro/ref/01_解说_8.8s.wav \
  --list   girlish_v2Pro/ref/ref.list \
  -t "要合成的文本" -o out.wav
```

Build the zip with Python — **`zip`/`unzip` are not installed** on this host. Store the weights uncompressed; they're already-compressed binaries, so deflate burns minutes for ~1%:

```bash
cd /tmp/opencode/pkg && python3 - <<'EOF'
import zipfile, os
out = "/home/rookie/voice/girlish_v2Pro.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for root, _, files in os.walk("girlish_v2Pro"):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            z.write(p, p, compress_type=zipfile.ZIP_STORED if fn.endswith((".ckpt", ".pth")) else zipfile.ZIP_DEFLATED)
print(f"{os.path.getsize(out) / 1024 / 1024:.0f} MB")
EOF
```

Before zipping, copy the staged dir somewhere else and run one synthesis through the in-bundle relative paths — that is what catches a broken `ref.list`. Verify afterwards with `zipfile.ZipFile(...).testzip()`.

The bundle deliberately **excludes pretrained models**. The target host still needs `GPT_SoVITS/pretrained_models/` (s1v3, chinese-hubert-base, chinese-roberta-wwm-ext-large, `sv/`, `v2Pro/`) plus `GPT_SoVITS/text/G2PWModel` for Chinese. Say so in the bundle README.

`infer.py`'s `DEFAULT_GPT`/`DEFAULT_SOVITS`/`DEFAULT_REF`/`DEFAULT_LIST` are hardcoded to one specific voice and **go stale whenever old weights are deleted**. Re-point them after training a new voice, or the no-argument invocation breaks.

### Ports

| Port | Service |
|---|---|
| 9874 | Main WebUI |
| 9873 | UVR5 WebUI |
| 9872 | TTS Inference WebUI |
| 9871 | SubFix WebUI |
| 9880 | API |

## Dev commands

### Using uv (recommended)

This project uses **uv** to manage the Python virtual environment. No conda needed.

```bash
# 1. Create venv with Python 3.10
uv venv --python 3.10

# 2. Install PyTorch first (match your CUDA version; see install.sh for other options)
uv pip install torch torchcodec --index-url https://download.pytorch.org/whl/cu128

# 3. Install dependencies in the REQUIRED order (extra-req first, no-deps)
uv pip install -r extra-req.txt --no-deps
uv pip install -r requirements.txt

# 4. Run any entrypoint
uv run python webui.py
uv run python api_v2.py
```

**All commands** (lint, run, python) must be prefixed with `uv run` to use the project venv:

```
uv run pre-commit run --all-files   # lint + format all
uv run ruff check .                 # lint only
uv run ruff format .                # format only (line-length 120, py311)
```

### Pretrained models not managed by uv

Pretrained weights, G2PWModel, UVR5 weights, and other assets must be downloaded separately. See `install.sh` for the full list of URLs. Model weights go in `GPT_SoVITS/pretrained_models/`, are never committed, and are gitignored.

## Important conventions & gotchas

- **No `requirements.txt` alone.** Install order: `pip install -r extra-req.txt --no-deps` then `pip install -r requirements.txt`. The `requirements.txt` line `--no-binary=opencc` breaks pip if run first.
- **`webui.py` is the source of truth for startup.** It patches `sys.path` by writing to `site-packages/users.pth`, sets `version=v2Pro` env, creates `TEMP/`, and initializes `TORCH_DISTRIBUTED_DEBUG=INFO`. Other entrypoints may not work without similar setup.
- **Config lives in `config.py`** (module-level globals), not an env file. GPU detection, default model paths, ports, and weight root dirs are all here.
- **Model weights are downloaded separately**, never committed. Pretrained baselines go in `GPT_SoVITS/pretrained_models/`, trained checkpoints in `SoVITS_weights*/` / `GPT_weights*/` (both gitignored).
- **Gradio < 5** (pinned: `gradio<5`). Do not bump past 4.x.
- **Transformers pinned**: `>=4.43,<=4.50`. PEFT `<0.18.0`. `numpy<2.0`.
- **Chinese TTS requires G2PW** (`G2PWModel` in `GPT_SoVITS/text/`), downloaded and unpacked separately.
- **Version-dependent code paths** are common. Check `version` env var and `config.pretrained_sovits_name`/`pretrained_gpt_name` dicts to understand which branch is hit.
- **`dict_language` keys are i18n-translated.** `inference_webui.py:172-193` builds them via `i18n("中文")`, so in an English locale the key becomes `"Chinese"` and passing `"中文"` raises `KeyError`. Set `os.environ["language"] = "zh_CN"` **before** importing `inference_webui` to keep the Chinese literals (this is what `infer.py` does).
- **`fast_langdetect` needs its cache dir to exist.** Inference fails with `FileNotFoundError: Cache directory not found: .../pretrained_models/fast_langdetect`. Create the dir and the model downloads itself.
- **IPC via subprocess kill signals**: `webui.py` launches `inference_webui.py` and `inference_webui_fast.py` as child processes, managing them with `Popen.pid` + signal handlers. Use `psutil` to find/kill children.
- **No type checking configured.** There is no `mypy` config.
- **`transformers>=4.57` needed for Fun-ASR-Nano**: The default ASR backend uses Fun-ASR-Nano which internally needs Qwen3 support (added in transformers 4.57). Upgrade if ASR fails with `KeyError: 'qwen3'`.
- **`torchaudio.load()` may fail via torchcodec.** torchaudio 2.11 routes `load()` through torchcodec, which needs CUDA-12 NPP libs (`libnppicc.so.12`). If the host only has CUDA 13, this raises `OSError`. Symlinking `.so.13` as `.so.12` does **not** work (symbol version mismatch). Workaround used here: replace `torchaudio.load()` with `soundfile.read()` (patched in `prepare_datasets/2-get-sv.py` and `inference_webui.py:get_spepc`). Note `sf.read` returns `[samples, ch]` — transpose to `[ch, samples]`. Training data loading is unaffected (`tools/my_utils.load_audio` shells out to ffmpeg).
- **Create weight output dirs before training.** `logs_s2_*`/`logs_s1_*` missing → first `save_checkpoint` crashes. `SoVITS_weights_*`/`GPT_weights_*` missing → `process_ckpt.savee()` fails *silently* (G/D checkpoints land fine, but the small inference weights never appear).
- **Reference audio must be 3–10s or inference dies.** `inference_webui.py:857` raises `OSError: 参考音频在3~10秒范围外，请更换！`. Slice filenames encode sample offsets at the *source* rate, so the implied duration does not match the resampled file — always confirm with `ffprobe -show_entries format=duration` instead of doing arithmetic on the filename.
- **Hand-edited ASR transcripts must land in `data/voices/<name>.list`.** `stage_asr` writes its raw output to `data/voices/<name>_asr/<name>_sliced.list` and then copies a filtered version to `data/voices/<name>.list`; the `text` stage reads **only** the latter. Editing the `_asr/` copy has no effect and silently trains on uncorrected text.
- **The ONNX `libcublasLt.so.12` error during inference is harmless.** G2PW's CUDA provider fails to load and falls back to CPU; audio is still produced. Do not chase it.

## Small-dataset training gotchas (source-verified)

Non-obvious behaviors that materially change how training should be configured:

- **Samples are silently duplicated below 100.** `module/data_utils.py:54-59`: if the dataset has `< min_num = 100` usable items, the whole list is repeated `max(2, int(100/len))` times. 21 items → `wav_data_len: 84`. **Configured epochs are then not real epochs** (30 configured ≈ 120+ passes over the same audio). This is a bucketing safeguard, *not* a hint that 100+ samples are required — 1 minute of audio genuinely works for v2Pro.
- **`DistributedBucketSampler` buckets by audio length, and each bucket fills batches independently.** Boundaries `[32,300,400,...,1900]` frames at `hop_length=640`/32kHz mean `bucket0` spans **0.64–6.00s** while every later bucket covers only 2s. Sparse buckets get padded with repeats (`data_utils.py:1015`). Keeping clips inside `bucket0` and using a modest `batch_size` matters far more than raw dataset duration.
- **S2 only supervises 0.64s per step.** `segment_size=20480` @32kHz; `s2_train.py:405,417` random-slices that window for mel/discriminator loss. Audio beyond ~6s adds SSL/spec context but **no extra waveform supervision** — long clips are not inherently better.
- **S1's LR schedule is hard-coded off.** `AR/modules/lr_schedulers.py:58` (`self.lr = lr = self.end_lr = 0.002  ###锁定用线性###不听话，直接锁定！`) overrides everything. `optimizer.lr`, `lr_init`, `lr_end`, `warmup_steps`, `decay_steps` in `s1longer-v2.yaml` are **all inert**. Ignore any external guide that tells you to tune them.
- **`3-get-semantic.py` can silently drop items.** Observed: 21 entries in `2-name2text.txt` but only 20 rows in `6-name2semantic.tsv` (a ~2s clip vanished, no error). Cross-check counts across `2-name2text.txt` / `4-cnhubert` / `5-wav32k` / `7-sv_cn` / `6-name2semantic.tsv` before training.
- **webui's own defaults are the sane baseline**, not the values in the config templates. `webui.py:124-127` for non-v3/v4: `default_sovits_epoch = 8`, `save_every = 4`, `max_sovits_epoch = 25`. The `epochs: 100` sitting in `s2v2Pro.json` is not a recommendation.
- **v2Pro vs v2ProPlus differ only in vocoder width.** Configs differ by two lines (`upsample_initial_channel` 512→768, first kernel 16→20); all 19M extra params are in the decoder (14.9M→33.8M), and both share the same GPT base (`s1v3.ckpt`). They take identical code paths (`models.py:623` `v2pro_set`). Expect no gain on timbre or prosody — only potentially on waveform quality, and only if the source material actually has high-frequency content to reconstruct.
