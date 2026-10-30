"""Evaluation: exact match, <=1-error match, token error rate, and error analysis."""
import random
from collections import Counter, defaultdict

import editdistance
import torch
import torch.nn.functional as F

from .decoding import beam_search, greedy_batch
from .tokenizer import PAD_ID, detokenize


def summarize(records):
    """records: list of (pred_tokens, target_tokens)."""
    n = len(records)
    exact = le1 = err = tot = 0
    for p, t in records:
        d = editdistance.eval(p, t)
        exact += d == 0
        le1 += d <= 1
        err += d
        tot += max(1, len(t))
    return dict(exact=100 * exact / n, exact_le1=100 * le1 / n, ter=100 * err / tot, n=n)


@torch.no_grad()
def evaluate(model, loader, vocab, device, amp):
    """Teacher-forced loss + greedy-decoded metrics over a loader. Returns (metrics, records)."""
    model.eval()
    records, loss_sum, n_tok = [], 0.0, 0
    V = len(vocab)
    for images, seqs in loader:
        images, seqs = images.to(device, non_blocking=True), seqs.to(device, non_blocking=True)
        with torch.autocast(**amp):
            logits = model(images, seqs[:, :-1])
        tgt = seqs[:, 1:]
        loss_sum += F.cross_entropy(logits.float().reshape(-1, V), tgt.reshape(-1),
                                    ignore_index=PAD_ID, reduction="sum").item()
        n_tok += (tgt != PAD_ID).sum().item()
        pred = greedy_batch(model, images, amp).tolist()
        records += [(vocab.ids_to_tokens(p), vocab.ids_to_tokens(t)) for p, t in zip(pred, seqs.tolist())]
    m = summarize(records)
    m["val_loss"] = loss_sum / max(1, n_tok)
    return m, records


def evaluate_beam(model, dset, vocab, amp, n=300, beam=5, seed=0):
    idx = random.Random(seed).sample(range(len(dset)), min(n, len(dset)))
    records = []
    for i in idx:
        x, t = dset[i]
        records.append((vocab.ids_to_tokens(beam_search(model, x, amp, beam=beam)),
                        vocab.ids_to_tokens(t.tolist())))
    return summarize(records), records


def error_report(records, bucket=10, top=15):
    b = defaultdict(list)
    for p, t in records:
        b[len(t) // bucket * bucket].append((p, t))
    print("target length | n | exact% | TER%")
    for k in sorted(b):
        s = summarize(b[k])
        print(f"  {k:3d}-{k + bucket - 1:<3d}      {s['n']:4d}   {s['exact']:5.1f}   {s['ter']:5.1f}")
    unbalanced = sum(p.count("{") != p.count("}") for p, _ in records)
    print(f"\npredictions with unbalanced braces: {unbalanced}/{len(records)}")
    miss = Counter()
    for p, t in records:
        if p != t:
            miss.update(Counter(t) - Counter(p))
    print("most-missed tokens:", miss.most_common(top))
    print("\nexamples of errors:")
    for p, t in [r for r in records if r[0] != r[1]][:8]:
        print("  pred:  ", detokenize(p))
        print("  target:", detokenize(t))
        print()
