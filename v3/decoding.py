"""Greedy (batched) and beam-search decoding. Both return raw token ids; map them with Vocab."""
import torch

from .tokenizer import EOS_ID, PAD_ID, SOS_ID


def _device(model):
    return next(model.parameters()).device


@torch.no_grad()
def greedy_batch(model, images, amp, max_len=151):
    model.eval()
    with torch.autocast(**amp):
        memory = model.encode(images)
    B = images.size(0)
    ys = torch.full((B, 1), SOS_ID, dtype=torch.long, device=images.device)
    done = torch.zeros(B, dtype=torch.bool, device=images.device)
    for _ in range(max_len):
        with torch.autocast(**amp):
            logits = model.decode(ys, memory)[:, -1]
        nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        done |= nxt == EOS_ID
        if done.all():
            break
    return ys[:, 1:]


@torch.no_grad()
def beam_search(model, image, amp, beam=5, max_len=151, alpha=0.7):
    """Beam search for ONE image (C,H,W). Length-normalised score = logp / len**alpha. Returns token ids."""
    model.eval()
    device = _device(model)
    with torch.autocast(**amp):
        memory = model.encode(image.unsqueeze(0).to(device))
    memory = memory.expand(beam, -1, -1).contiguous()
    V = model.out.out_features
    seqs = torch.full((beam, 1), SOS_ID, dtype=torch.long, device=device)
    scores = torch.zeros(beam, device=device)
    scores[1:] = float("-inf")
    finished = torch.zeros(beam, dtype=torch.bool, device=device)
    for _ in range(max_len):
        with torch.autocast(**amp):
            logits = model.decode(seqs, memory)[:, -1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        logp[finished] = float("-inf")
        logp[finished, PAD_ID] = 0.0
        scores, idx = (scores.unsqueeze(1) + logp).view(-1).topk(beam)
        bi, ti = idx // V, idx % V
        seqs = torch.cat([seqs[bi], ti.unsqueeze(1)], dim=1)
        finished = finished[bi] | (ti == EOS_ID)
        if finished.all():
            break
    lengths = ((seqs != PAD_ID) & (seqs != SOS_ID)).sum(1).float().clamp(min=1)
    best = (scores / lengths.pow(alpha)).argmax()
    return seqs[best].tolist()
