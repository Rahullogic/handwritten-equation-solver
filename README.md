# Handwritten Equation Solver

Convert an image of a handwritten math equation into its LaTeX form and solve it.

This is an end-to-end deep learning project built in PyTorch, going from raw handwritten symbol images all the way to a trained system that understands math. It's being built in phases — this README covers Phase 1: Symbol Classifier.

Project Goal

Take a photo/scan of a handwritten equation (e.g. 2 + 3 = ?) and:

Detect and recognize each individual symbol in the image
Assemble the symbols into a correct LaTeX / math expression
(Eventually) solve the equation and return the answer.



Phase 1: Symbol Classifier ✅

Before the model can read a whole equation, it first needs to recognize individual handwritten math symbols (digits, +, -, x, y, (, ), and so on). Phase 1 builds and trains that symbol classifier.

Dataset
Handwritten Math Symbols (Kaggle)
82 symbol classes after extracting the full dataset (digits, operators, letters, etc.)
Model

A convolutional neural network (CNN) built from scratch in PyTorch:

*Conv2D → ReLU → MaxPool blocks (with BatchNorm)
*Trained on grayscale 28x28 images
*Input: single symbol image → Output: predicted class (82-way classification)
Training
*Optimizer: Adam (lr=0.01, weight_decay=1e-4)
*Loss: CrossEntropyLoss
*85/15 train/validation split
*Trained for 25 epochs on the full 82-class dataset
*Result: ~85% validation accuracy


How to run this phase:

1.Open the notebook in Google Colab
2.Mount Google Drive (for dataset storage) and set your Kaggle API credentials
3.Run the cells top to bottom:
*Downloads and extracts the dataset
*Builds the ImageFolder dataset and dataloaders
*Defines the CNN (MyNN)
*Trains the model and saves weights to Google Drive
*Trained weights are saved as symbol_classifier_82class.pth

---

## Phase 2: Whole-equation recognition (image → LaTeX → answer)

Phase 1 recognised one symbol at a time. Phase 2 reads a whole equation image directly and writes out the LaTeX, then solves it. I built it twice: **v2** (first attempt) and **v3** (rebuild). Both are in this repo.

**Data:** [MathWriting](https://github.com/google-research/google-research/tree/master/mathwriting) human-written equations (229,864 train / 15,674 val / 7,644 test), tokenised into 224 LaTeX tokens.

### v2 – first attempt (`legacy_v2/`)
A small 4-layer CNN reads the image and a Transformer decoder writes the LaTeX token by token.

**What went wrong:** the CNN had no idea *where* in the image each feature came from, so the decoder struggled with layout (fractions, superscripts, long lines). I tried to fix this by adding 2D positional encoding to the already-trained model. That changed the encoder's features and the model collapsed: it output `}}}}` for everything, loss jumped from 0.22 to 2.7 and validation accuracy dropped to 0%. You can't bolt position information onto a trained model. It has to be there from the start, so I rebuilt it.

### v3 – rebuild (`v3/`)
Everything retrained from scratch, with these changes:

| Change | Why it helps |
|---|---|
| Pretrained **ResNet-34** encoder instead of a 4-layer CNN | Starts with real visual features (edges, strokes, shapes) instead of learning them from zero |
| **2D positional encoding from the start** | The model knows where each feature sits on the page, which is what the v2 fix was missing |
| Keeps horizontal resolution (layer 3 stride 2×1) | Image becomes 8×64 = 512 tokens, so thin strokes and small symbols aren't blurred away |
| 3 Transformer layers on the image features | Distant parts of the equation (like both sides of a fraction) can look at each other |
| Pre-LayerNorm + GELU, tied embeddings | More stable training, fewer parameters |
| AdamW, warmup + cosine LR, label smoothing, gradient clipping | Smoother training, less overconfidence |
| **EMA weights**, bf16, early stopping, resumable checkpoints | Better final model, faster training, and I can stop and continue |
| Augmentation (small rotations/shifts, stroke thickness, blur) and aspect-ratio-preserving resize | Handles different handwriting styles without squashing equations |

### Results (v3, best epoch 15)

| Split | Decoding | Exact match | ≤1 token wrong | Token error rate |
|---|---|---|---|---|
| Validation | greedy | 76.3% | 87.3% | 3.67% |
| Test (7,644) | greedy | 63.6% | 78.6% | 5.48% |
| Test (300 random) | beam search (5) | 60.3% | 74.0% | 6.59% |

Short equations are much easier than long ones: about 72% exact for equations up to 9 tokens, but around 32% for 60+ tokens. Beam search didn't beat greedy here, but it was only tested on 300 samples.

### Solving
`v3/solve.py` takes the predicted LaTeX and solves it with SymPy: arithmetic, equations (`x^2-5x+6=0` → x = 2, 3), systems, true/false checks, derivatives and indefinite integrals. A single misread symbol can change the answer, so the app shows the LaTeX next to the result.

### Run it
```bash
pip install -r requirements.txt
set EQ_DATA_DIR=path\to\data              # folder containing mathwriting_human/

python -m v3.train                        # train (resumable)
python -m v3.evaluate --beam-n 300        # test-set results
python -m v3.predict equation.png --solve # image -> LaTeX -> answer
python -m v3.visualize image equation.png # show the image, prediction and result
```
Run commands from the repo root. See `v3/README.md` for the file layout.
