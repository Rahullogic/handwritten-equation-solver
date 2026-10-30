"""Overfit 64 samples (~1-2 min). Exact match should approach 100% - if not, do NOT start the long run.

Run:  python -m v3.sanity_check
"""
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from . import config
from .data import Collator, EquationDataset, load_hf_splits
from .metrics import evaluate
from .model import build_model
from .tokenizer import PAD_ID

OVERFIT_STEPS = 300


def main():
    device, amp = config.setup_runtime()
    cfg = config.CFG
    train_hf, _, _, vocab = load_hf_splits()
    V = len(vocab)
    small = Subset(EquationDataset(train_hf, vocab, cfg["max_tokens"], train=False,
                                   h=cfg["img_h"], w=cfg["img_w"]), list(range(64)))
    loader = DataLoader(small, batch_size=16, shuffle=True, collate_fn=Collator(PAD_ID))

    m = build_model(cfg, V).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=0.0)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    step, t0 = 0, time.time()
    while step < OVERFIT_STEPS:
        for images, seqs in loader:
            m.train()
            images, seqs = images.to(device), seqs.to(device)
            with torch.autocast(**amp):
                logits = m(images, seqs[:, :-1])
            loss = crit(logits.float().reshape(-1, V), seqs[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)")
            if step >= OVERFIT_STEPS:
                break

    res, _ = evaluate(m, DataLoader(small, batch_size=32, collate_fn=Collator(PAD_ID)), vocab, device, amp)
    print({k: round(v, 2) for k, v in res.items()})


if __name__ == "__main__":
    main()
