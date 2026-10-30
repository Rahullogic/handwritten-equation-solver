"""Central configuration: paths, model config, training hyper-parameters, device/AMP helpers.

Paths are taken from environment variables so the code runs on any machine:
    EQ_DATA_DIR    folder holding `mathwriting_human/` (and optionally `mathwriting_synthetic/`)
    EQ_BACKUP_DIR  optional second folder that receives a copy of every new best.pt
"""
import os
import random

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SEED = 42

# ---- paths -----------------------------------------------------------------
DATA_DIR = os.environ.get("EQ_DATA_DIR", "data")
HF_PATH = os.path.join(DATA_DIR, "mathwriting_human")
SYNTH_PATH = os.path.join(DATA_DIR, "mathwriting_synthetic")
USE_SYNTH = True
RUN_DIR = os.path.join(DATA_DIR, "v3")                 # last.pt, best.pt, vocab.json, history.json
BACKUP_DIR = os.environ.get("EQ_BACKUP_DIR")           # optional

# ---- model (saved inside every checkpoint; resume requires it to match exactly) --
CFG = dict(img_h=128, img_w=512, max_tokens=150,
           d_model=384, nhead=8, n_enc_layers=3, n_dec_layers=6, dim_ff=1536,
           dropout=0.1, pretrained=True)

# ---- training --------------------------------------------------------------
BATCH = 32
EPOCHS = 30
PEAK_LR = 5e-4
WARMUP_STEPS = 1500
MIN_LR_RATIO = 0.02
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
EMA_DECAY = 0.9995
EVAL_N = 1500          # random val samples scored (greedy) after every epoch
PATIENCE = 6           # early stopping on val TER
NUM_WORKERS = 0        # 0 is safest on Windows

MIN_FREQ = 5           # tokens rarer than this become <unk>


def setup_runtime(seed=SEED):
    """Seed RNGs, pick the device, set backend flags. Returns (device, amp_kwargs)."""
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    return device, amp_kwargs(device)


def amp_kwargs(device):
    """kwargs for torch.autocast: bf16 on CUDA, disabled elsewhere."""
    return dict(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda"))


def ensure_dirs():
    os.makedirs(RUN_DIR, exist_ok=True)
    if BACKUP_DIR:
        os.makedirs(BACKUP_DIR, exist_ok=True)
