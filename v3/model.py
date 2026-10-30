"""ResNet-34 + Transformer encoder, Transformer decoder, and EMA."""
import copy
import math

import torch
import torch.nn as nn
import torchvision as tv

from .tokenizer import PAD_ID


def sinusoid_1d(n, d, device=None):
    pos = torch.arange(n, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class PositionalEncoding2D(nn.Module):
    """Parameter-free 2D sinusoidal encoding: first half of channels = row, second half = column."""

    def __init__(self, d_model):
        super().__init__()
        assert d_model % 4 == 0
        self.d = d_model

    def forward(self, H, W, device):
        d = self.d // 2
        pe_h = sinusoid_1d(H, d, device)[:, None, :].expand(H, W, d)
        pe_w = sinusoid_1d(W, d, device)[None, :, :].expand(H, W, d)
        return torch.cat([pe_h, pe_w], dim=-1)


class ImageEncoder(nn.Module):
    """ResNet-34 up to layer3 (stride 16 vertically, 8 horizontally) -> 8x64 = 512 tokens for a 128x512 input."""

    def __init__(self, d_model, nhead, n_layers, dim_ff, dropout, pretrained):
        super().__init__()
        r = tv.models.resnet34(weights=tv.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        conv1.weight.data.copy_(r.conv1.weight.data.sum(1, keepdim=True))        # RGB -> grayscale filters
        r.layer3[0].conv1.stride = (2, 1)                                         # keep horizontal resolution
        r.layer3[0].downsample[0].stride = (2, 1)
        self.stem = nn.Sequential(conv1, r.bn1, r.relu, r.maxpool)
        self.body = nn.Sequential(r.layer1, r.layer2, r.layer3)
        self.proj = nn.Conv2d(256, d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.pe2d = PositionalEncoding2D(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model),
                                                 enable_nested_tensor=False)

    def forward(self, x):
        f = self.proj(self.body(self.stem(x)))                     # (B, D, H', W')
        B, D, Hh, Ww = f.shape
        f = self.norm(f.permute(0, 2, 3, 1))                       # (B, H', W', D)
        f = f + self.pe2d(Hh, Ww, f.device).to(f.dtype)
        return self.transformer(f.reshape(B, Hh * Ww, D))          # (B, S, D)


class EquationModel(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=8, n_enc_layers=3, n_dec_layers=6, dim_ff=1536,
                 dropout=0.1, pretrained=True, max_len=256, **_):
        super().__init__()
        self.d_model = d_model
        self.encoder = ImageEncoder(d_model, nhead, n_enc_layers, dim_ff, dropout, pretrained)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        nn.init.normal_(self.embed.weight, 0.0, d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[PAD_ID].zero_()
        self.register_buffer("pos", sinusoid_1d(max_len, d_model), persistent=False)
        self.emb_drop = nn.Dropout(dropout)
        dl = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, activation="gelu",
                                        batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dl, n_dec_layers, norm=nn.LayerNorm(d_model))
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.embed.weight                        # tied input/output embeddings

    def encode(self, images):
        return self.encoder(images)

    def decode(self, tgt, memory, tgt_pad_mask=None):
        T = tgt.size(1)
        x = self.emb_drop(self.embed(tgt) * math.sqrt(self.d_model) + self.pos[:T])
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt.device), 1)
        return self.out(self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=tgt_pad_mask))

    def forward(self, images, tgt):
        return self.decode(tgt, self.encode(images), tgt == PAD_ID)


def build_model(cfg, vocab_size, pretrained=None):
    p = cfg["pretrained"] if pretrained is None else pretrained
    kw = {k: v for k, v in cfg.items() if k != "pretrained"}
    return EquationModel(vocab_size, pretrained=p, **kw)


class EMA:
    """Exponential moving average of the weights (decay warms up from ~0.1 to `decay`)."""

    def __init__(self, model, decay):
        self.decay, self.n = decay, 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)
