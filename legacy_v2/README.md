# v2 baseline (previous work)

`train.ipynb` — 4-conv CNN encoder + post-LN Transformer decoder, trained from scratch.

Note: the final evaluation cells in this notebook show a collapsed output (`}}}}`) after
2D positional encodings were added to the already-trained model. That failure motivated the
from-scratch rebuild in `../v3` (ResNet-34 encoder, 2D PE from the start).
