"""Plot helpers: training-sample preview and validation predictions.

Run:  python -m v3.visualize samples
      python -m v3.visualize predictions
      python -m v3.visualize image path/to/equation.png [--save out.png]
"""
import argparse
import random

import matplotlib.pyplot as plt
from PIL import Image

from . import config
from .data import build_datasets, load_hf_splits, preprocess_pil
from .evaluate import load_best
from .predict import predict_pil
from .tokenizer import detokenize, tokenize_latex


def show_train_samples(train_ds, vocab, n=6):
    fig, axes = plt.subplots(3, 2, figsize=(12, 6))
    for ax, i in zip(axes.ravel(), random.sample(range(len(train_ds)), n)):
        x, ids = train_ds[i]
        ax.imshow(x[0], cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
        ax.set_title(detokenize(vocab.ids_to_tokens(ids.tolist())), fontsize=9)
    plt.tight_layout()
    plt.show()


def show_predictions(model, vocab, val_hf, amp, n=8):
    fig, axes = plt.subplots(4, 2, figsize=(12, 8))
    for ax, i in zip(axes.ravel(), random.sample(range(len(val_hf)), n)):
        ex = val_hf[i]
        pred = predict_pil(model, vocab, ex["image"], amp)
        tgt = detokenize(tokenize_latex(ex["latex"]))
        ok = pred == tgt
        ax.imshow(ex["image"].convert("L"), cmap="gray")
        ax.axis("off")
        ax.set_title(("OK " if ok else "X ") + pred + ("" if ok else f"\n(target: {tgt})"),
                     fontsize=8, color="green" if ok else "red")
    plt.tight_layout()
    plt.show()


def show_image(model, vocab, path, amp, beam=5, save=None):
    """Show one image, what the model sees after preprocessing, the predicted LaTeX and the solved result."""
    from .solve import solve_latex

    cfg = config.CFG
    img = Image.open(path)
    latex = predict_pil(model, vocab, img, amp, beam=beam)
    res = solve_latex(latex)

    seen = preprocess_pil(img, cfg["img_h"], cfg["img_w"])[0]          # what the network actually receives
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    axes[0].imshow(img.convert("L"), cmap="gray")
    axes[0].set_title("input image", fontsize=10)
    axes[1].imshow(seen, cmap="gray", vmin=-1, vmax=1)
    axes[1].set_title(f"model input ({cfg['img_h']}x{cfg['img_w']}, padded)", fontsize=9)
    for ax in axes[:2]:
        ax.axis("off")

    axes[2].axis("off")
    try:                                                                # render the LaTeX if matplotlib can
        axes[2].text(0.5, 0.72, f"${latex}$", ha="center", va="center", fontsize=18)
        fig.canvas.draw()
    except Exception:
        axes[2].texts[-1].remove()
        axes[2].text(0.5, 0.72, latex, ha="center", va="center", fontsize=12, family="monospace")
    axes[2].text(0.5, 0.2, f"{res['kind']}: {res['detail']}", ha="center", va="center", fontsize=12,
                 color="green" if res["kind"] != "error" else "red")
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        print("saved", save)
    plt.show()
    return latex, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["samples", "predictions", "image"])
    ap.add_argument("path", nargs="?", help="image file (for 'image')")
    ap.add_argument("--save", default=None, help="also save the figure to this file (for 'image')")
    ap.add_argument("--beam", type=int, default=5)
    args = ap.parse_args()
    device, amp = config.setup_runtime()
    if args.what == "image":
        if not args.path:
            ap.error("usage: python -m v3.visualize image path/to/equation.png")
        model, vocab, _ = load_best(device)
        show_image(model, vocab, args.path, amp, beam=args.beam, save=args.save)
        return
    train_hf, val_hf, test_hf, vocab = load_hf_splits()
    if args.what == "samples":
        train_ds, *_ = build_datasets(train_hf, val_hf, test_hf, vocab)
        show_train_samples(train_ds, vocab)
    else:
        model, vocab, _ = load_best(device)
        show_predictions(model, vocab, val_hf, amp)


if __name__ == "__main__":
    main()
