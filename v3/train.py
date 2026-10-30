"""Training loop (resumable). Run:  python -m v3.train

Re-running continues from RUN_DIR/last.pt. Validation metrics are computed from the EMA weights.
"""
import json
import math
import os
import shutil
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from . import config
from .data import build_datasets, load_hf_splits, make_loader
from .metrics import evaluate
from .model import EMA, build_model


def make_optimizer(model):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": config.WEIGHT_DECAY},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=config.PEAK_LR, betas=(0.9, 0.98))


def make_scheduler(opt, total_steps):
    """Linear warmup then cosine decay to MIN_LR_RATIO * peak."""
    def lr_lambda(step):
        if step < config.WARMUP_STEPS:
            return (step + 1) / config.WARMUP_STEPS
        prog = (step - config.WARMUP_STEPS) / max(1, total_steps - config.WARMUP_STEPS)
        return config.MIN_LR_RATIO + (1 - config.MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train(train_loader, eval_loader, vocab, device, amp):
    cfg = config.CFG
    V = len(vocab)
    model = build_model(cfg, V).to(device)
    ema = EMA(model, config.EMA_DECAY)
    opt = make_optimizer(model)
    sched = make_scheduler(opt, config.EPOCHS * len(train_loader))
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=config.LABEL_SMOOTHING)

    start_epoch, best_ter, bad, history = 0, float("inf"), 0, []
    last_path = os.path.join(config.RUN_DIR, "last.pt")
    best_path = os.path.join(config.RUN_DIR, "best.pt")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck["cfg"] == cfg:
            model.load_state_dict(ck["model"])
            ema.shadow.load_state_dict(ck["ema"])
            ema.n = ck["ema_n"]
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_epoch, best_ter, bad, history = ck["epoch"], ck["best_ter"], ck["bad"], ck["history"]
            print(f"resumed from epoch {start_epoch} (best val TER so far {best_ter:.2f}%)")
        else:
            print("CFG changed since last.pt was written -> starting fresh (delete last.pt to silence this)")

    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        t0 = time.time()
        run_loss, n_ok = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{config.EPOCHS}", leave=False)
        for images, seqs in pbar:
            images, seqs = images.to(device, non_blocking=True), seqs.to(device, non_blocking=True)
            with torch.autocast(**amp):
                logits = model(images, seqs[:, :-1])
            loss = criterion(logits.float().reshape(-1, V), seqs[:, 1:].reshape(-1))
            if not torch.isfinite(loss):
                print("non-finite loss, skipping batch")
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            opt.step()
            sched.step()
            ema.update(model)
            run_loss += loss.item()
            n_ok += 1
            if n_ok % 50 == 0:
                pbar.set_postfix(loss=f"{run_loss / n_ok:.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")

        m, _ = evaluate(ema.shadow, eval_loader, vocab, device, amp)
        m.update(epoch=epoch + 1, train_loss=run_loss / max(1, n_ok), minutes=(time.time() - t0) / 60)
        history.append(m)
        left = (config.EPOCHS - epoch - 1) * m["minutes"]
        print(f"epoch {epoch + 1:2d}/{config.EPOCHS} | train {m['train_loss']:.3f} | val_loss {m['val_loss']:.3f} | "
              f"exact {m['exact']:5.1f}% | <=1 err {m['exact_le1']:5.1f}% | TER {m['ter']:5.2f}% | "
              f"{m['minutes']:.1f} min | ETA {left / 60:.1f} h")

        if m["ter"] < best_ter:
            best_ter, bad = m["ter"], 0
            torch.save({"model": ema.shadow.state_dict(), "cfg": cfg, "vocab": vocab.tokens,
                        "epoch": epoch + 1, "metrics": m}, best_path)
            if config.BACKUP_DIR:
                try:
                    shutil.copy2(best_path, os.path.join(config.BACKUP_DIR, "best.pt"))
                except Exception as e:
                    print("backup copy failed:", e)
            print("   -> new best (saved best.pt)")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(), "ema_n": ema.n,
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "cfg": cfg,
                    "epoch": epoch + 1, "best_ter": best_ter, "bad": bad, "history": history},
                   last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)
        with open(os.path.join(config.RUN_DIR, "history.json"), "w") as f:
            json.dump(history, f)
        if bad >= config.PATIENCE:
            print(f"early stop: no val-TER improvement for {config.PATIENCE} epochs")
            break
    return ema.shadow


def main():
    device, amp = config.setup_runtime()
    config.ensure_dirs()
    print("torch", torch.__version__, "| device:", device)
    train_hf, val_hf, test_hf, vocab = load_hf_splits()
    train_ds, _, _, eval_ds = build_datasets(train_hf, val_hf, test_hf, vocab)
    train_loader = make_loader(train_ds, config.BATCH, device, shuffle=True, drop_last=True)
    eval_loader = make_loader(eval_ds, 64, device)
    print("train batches/epoch:", len(train_loader))
    train(train_loader, eval_loader, vocab, device, amp)


if __name__ == "__main__":
    main()
