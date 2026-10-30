"""Datasets, image preprocessing/augmentation, collation and loaders."""
import os
import random
from collections import Counter

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from . import config
from .tokenizer import EOS_ID, PAD_ID, SOS_ID, Vocab, tokenize_latex


class ResizeWithPad:
    """Keep aspect ratio, centre on a white canvas of (h, w)."""

    def __init__(self, h=128, w=512, fill=255):
        self.h, self.w, self.fill = h, w, fill

    def __call__(self, img):
        w, h = img.size
        s = min(self.w / w, self.h / h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        img = img.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("L", (self.w, self.h), self.fill)
        canvas.paste(img, ((self.w - nw) // 2, (self.h - nh) // 2))
        return canvas


def preprocess_pil(img, h=128, w=512):
    """PIL image -> normalised (1, h, w) tensor. Same path used for val/test/inference."""
    img = ResizeWithPad(h, w)(img.convert("L"))
    return (TF.to_tensor(img) - 0.5) / 0.5


class EquationDataset(Dataset):
    def __init__(self, hf_ds, vocab: Vocab, max_tokens=150, train=False, h=128, w=512):
        self.ds, self.vocab = hf_ds, vocab
        self.max_tokens, self.train, self.h, self.w = max_tokens, train, h, w
        self.resize = ResizeWithPad(h, w)
        self.affine = T.RandomAffine(degrees=3, translate=(0.02, 0.05), scale=(0.9, 1.1),
                                     shear=(-5, 5, -3, 3), fill=255)

    def __len__(self):
        return len(self.ds)

    @staticmethod
    def _stroke_jitter(img):
        r = random.random()
        if r < 0.15:
            img = img.filter(ImageFilter.MinFilter(3))      # thicker strokes
        elif r < 0.30:
            img = img.filter(ImageFilter.MaxFilter(3))      # thinner strokes
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
        return img

    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex["image"].convert("L")
        if self.train:
            img = self.affine(img)
        img = self.resize(img)
        if self.train:
            img = self._stroke_jitter(img)
        x = (TF.to_tensor(img) - 0.5) / 0.5

        toks = tokenize_latex(ex["latex"])[: self.max_tokens]
        ids = [SOS_ID] + self.vocab.encode(toks) + [EOS_ID]
        return x, torch.tensor(ids, dtype=torch.long)


class Collator:
    def __init__(self, pad_id=PAD_ID):
        self.pad_id = pad_id

    def __call__(self, batch):
        images, seqs = zip(*batch)
        return torch.stack(images), pad_sequence(seqs, batch_first=True, padding_value=self.pad_id)


def make_loader(dset, batch, device, shuffle=False, drop_last=False):
    kw = dict(batch_size=batch, shuffle=shuffle, drop_last=drop_last, collate_fn=Collator(PAD_ID),
              num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"))
    if config.NUM_WORKERS > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(dset, **kw)


def _drop_long(hf, max_tokens, name):
    lat = list(hf.select_columns(["latex"])["latex"])
    keep = [i for i, s in enumerate(lat) if len(tokenize_latex(s)) <= max_tokens]
    print(f"  {name}: kept {len(keep)}/{len(lat)} (dropped {len(lat) - len(keep)} longer than {max_tokens} tokens)")
    return hf.select(keep)


def load_hf_splits(max_tokens=None):
    """Load the HF datasets from disk. Returns (train_hf, val_hf, test_hf, vocab).

    The vocab is built from the *unfiltered* training latex (as in the original notebook)
    and saved to RUN_DIR/vocab.json.
    """
    from datasets import DatasetDict, concatenate_datasets, load_from_disk

    max_tokens = max_tokens or config.CFG["max_tokens"]
    ds = load_from_disk(config.HF_PATH)
    print(ds)

    cols = ["image", "latex"]
    train_hf = ds["train"].select_columns(cols)
    if config.USE_SYNTH and os.path.exists(config.SYNTH_PATH):
        synth = load_from_disk(config.SYNTH_PATH)
        synth = synth["train"] if isinstance(synth, DatasetDict) else synth
        train_hf = concatenate_datasets([train_hf, synth.select_columns(cols)])
        print("added synthetic data ->", len(train_hf), "training samples")

    counter, lengths = Counter(), []
    for s in train_hf.select_columns(["latex"])["latex"]:
        t = tokenize_latex(s)
        counter.update(t)
        lengths.append(len(t))
    lengths = np.array(lengths)
    print("token length  p50/p95/p99/max:", *(int(np.percentile(lengths, q)) for q in (50, 95, 99)), lengths.max())

    train_hf = _drop_long(train_hf, max_tokens, "train")
    val_hf = _drop_long(ds["val"].select_columns(cols), max_tokens, "val")
    test_hf = _drop_long(ds["test"].select_columns(cols), max_tokens, "test")

    vocab = Vocab.from_counter(counter, config.MIN_FREQ)
    config.ensure_dirs()
    vocab.save(os.path.join(config.RUN_DIR, "vocab.json"))
    print("vocab size:", len(vocab))
    return train_hf, val_hf, test_hf, vocab


def build_datasets(train_hf, val_hf, test_hf, vocab, cfg=None):
    """Wrap the HF splits into EquationDatasets (+ a fixed random val subset for per-epoch eval)."""
    cfg = cfg or config.CFG
    h, w, mt = cfg["img_h"], cfg["img_w"], cfg["max_tokens"]
    train_ds = EquationDataset(train_hf, vocab, mt, train=True, h=h, w=w)
    val_ds = EquationDataset(val_hf, vocab, mt, train=False, h=h, w=w)
    test_ds = EquationDataset(test_hf, vocab, mt, train=False, h=h, w=w)
    eval_idx = random.Random(config.SEED).sample(range(len(val_ds)), min(config.EVAL_N, len(val_ds)))
    return train_ds, val_ds, test_ds, Subset(val_ds, eval_idx)
