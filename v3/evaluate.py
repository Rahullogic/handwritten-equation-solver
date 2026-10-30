"""Final evaluation of best.pt on the test set.  Run:  python -m v3.evaluate [--beam-n 300]"""
import argparse
import os
import time

import torch

from . import config
from .data import build_datasets, load_hf_splits, make_loader
from .metrics import error_report, evaluate, evaluate_beam
from .model import build_model
from .tokenizer import Vocab


def load_best(device, path=None):
    """Load best.pt -> (model in eval mode, Vocab, checkpoint dict)."""
    path = path or os.path.join(config.RUN_DIR, "best.pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    vocab = Vocab(ck["vocab"])
    model = build_model(ck["cfg"], len(vocab), pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, vocab, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam-n", type=int, default=300, help="test samples for beam search (0 = skip)")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    device, amp = config.setup_runtime()
    model, vocab, ck = load_best(device, args.checkpoint)
    print("best epoch:", ck["epoch"], {k: round(v, 2) for k, v in ck["metrics"].items() if isinstance(v, float)})

    train_hf, val_hf, test_hf, _ = load_hf_splits()
    _, _, test_ds, _ = build_datasets(train_hf, val_hf, test_hf, vocab, ck["cfg"])

    t0 = time.time()
    test_m, records = evaluate(model, make_loader(test_ds, 64, device), vocab, device, amp)
    print(f"TEST greedy  ({test_m['n']} samples, {time.time() - t0:.0f}s):", {k: round(v, 2) for k, v in test_m.items()})
    if args.beam_n:
        beam_m, _ = evaluate_beam(model, test_ds, vocab, amp, n=args.beam_n, beam=args.beam)
        print(f"TEST beam={args.beam}  ({args.beam_n} random samples):", {k: round(v, 2) for k, v in beam_m.items()})
    error_report(records)


if __name__ == "__main__":
    main()
