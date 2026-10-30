"""LaTeX tokenisation and the token <-> id vocabulary."""
import json
import re
from collections import Counter

PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

_TOKEN_RE = re.compile(r"\\[a-zA-Z]+|[^\s]")


def tokenize_latex(s):
    return _TOKEN_RE.findall(s)


def detokenize(tokens):
    """Join tokens back into valid LaTeX (a space is needed between \\cmd and a following letter)."""
    out = []
    for i, t in enumerate(tokens):
        out.append(t)
        if (len(t) > 1 and t[0] == "\\" and t[1:].isalpha()
                and i + 1 < len(tokens) and tokens[i + 1][:1].isalpha()):
            out.append(" ")
    return "".join(out)


class Vocab:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.token2id = {t: i for i, t in enumerate(self.tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        assert [self.token2id[t] for t in SPECIAL_TOKENS] == [PAD_ID, SOS_ID, EOS_ID, UNK_ID]

    def __len__(self):
        return len(self.tokens)

    @classmethod
    def from_counter(cls, counter: Counter, min_freq: int):
        return cls(SPECIAL_TOKENS + sorted(t for t, c in counter.items() if c >= min_freq))

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.tokens, f, ensure_ascii=False)

    def encode(self, tokens):
        return [self.token2id.get(t, UNK_ID) for t in tokens]

    def ids_to_tokens(self, ids):
        """Stop at <eos>/<pad>, skip <sos>."""
        out = []
        for i in ids:
            if i in (EOS_ID, PAD_ID):
                break
            if i == SOS_ID:
                continue
            out.append(self.id2token[i])
        return out
