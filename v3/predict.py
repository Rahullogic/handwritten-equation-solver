"""Inference on any image.  Run:  python -m v3.predict path/to/equation.png [--beam 5] [--solve]

Tip: crop tightly around the equation; dark ink on a white background works best.
"""
import argparse

from PIL import Image

from . import config
from .data import preprocess_pil
from .decoding import beam_search
from .evaluate import load_best
from .tokenizer import detokenize


def predict_pil(model, vocab, img, amp, beam=5):
    cfg = config.CFG
    x = preprocess_pil(img, cfg["img_h"], cfg["img_w"])
    return detokenize(vocab.ids_to_tokens(beam_search(model, x, amp, beam=beam)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--solve", action="store_true", help="also solve/evaluate the predicted LaTeX")
    args = ap.parse_args()
    device, amp = config.setup_runtime()
    model, vocab, _ = load_best(device, args.checkpoint)
    latex = predict_pil(model, vocab, Image.open(args.image), amp, beam=args.beam)
    print("LaTeX:", latex)
    if args.solve:
        from .solve import solve_latex
        r = solve_latex(latex)
        print(f"Result ({r['kind']}):", r["detail"])


if __name__ == "__main__":
    main()
