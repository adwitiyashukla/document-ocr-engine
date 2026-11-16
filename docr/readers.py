import math
import unicodedata

import numpy as np
import torch
from torch import nn

from docr.ctc import beam, greedy
from docr.imaging import resize

CHARS = "".join(chr(c) for c in range(32, 127))


class Charset:
    def __init__(self, chars=CHARS, offset=1):
        self.chars = chars
        self.offset = offset
        self.index = {c: i + offset for i, c in enumerate(chars)}

    def __len__(self):
        return len(self.chars) + self.offset

    def normalize(self, text):
        out = []
        for ch in text:
            if ch in self.index:
                out.append(ch)
                continue
            folded = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii")
            out.extend(c for c in folded if c in self.index)
        return "".join(out)

    def encode(self, text):
        return [self.index[c] for c in self.normalize(text)]

    def decode(self, ids):
        return "".join(self.chars[i - self.offset] for i in ids if self.offset <= i < len(self))


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CRNN(nn.Module):
    def __init__(self, classes, channels=(32, 64, 128, 128, 192, 192, 256), hidden=192):
        super().__init__()
        c = channels
        self.features = nn.Sequential(
            block(1, c[0]), nn.MaxPool2d(2, 2),
            block(c[0], c[1]), nn.MaxPool2d(2, 2),
            block(c[1], c[2]), block(c[2], c[3]), nn.MaxPool2d((2, 1), (2, 1)),
            block(c[3], c[4]), block(c[4], c[5]), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(c[5], c[6], (2, 1), bias=False), nn.BatchNorm2d(c[6]), nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(c[6], hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.head = nn.Linear(2 * hidden, classes)

    def forward(self, x):
        f = self.features(x).squeeze(2).transpose(1, 2)
        f, _ = self.rnn(f)
        return self.head(f)


def build(options):
    return CRNN(len(Charset()), tuple(options["channels"]), options["hidden"])


def prepare_crop(crop, height=32, max_width=512):
    crop = np.asarray(crop, np.float32)
    h, w = crop.shape[:2]
    width = int(min(max(round(w * height / max(h, 1)), 16), max_width))
    width = int(math.ceil(width / 4) * 4)
    return resize(crop, (height, width))


def to_tensor(crops, pad_value=None):
    width = max(c.shape[1] for c in crops)
    batch = np.zeros((len(crops), 1, crops[0].shape[0], width), np.float32)
    for i, c in enumerate(crops):
        fill = float(np.median(c[:, -2:])) if pad_value is None else pad_value
        batch[i, 0, :, :] = fill
        batch[i, 0, :, :c.shape[1]] = c
    return torch.from_numpy((batch - 127.5) / 127.5)


class Reader:
    def __init__(self, model, beam_width=0, max_width=512):
        self.model = model.eval()
        self.charset = Charset()
        self.beam_width = beam_width
        self.max_width = max_width

    @torch.no_grad()
    def __call__(self, crops, batch_size=64):
        if not len(crops):
            return [], []
        texts, scores = [None] * len(crops), [None] * len(crops)
        prepared = [prepare_crop(c, max_width=self.max_width) for c in crops]
        order = np.argsort([p.shape[1] for p in prepared])
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            logp = self.model(to_tensor([prepared[i] for i in idx])).log_softmax(-1)
            if self.beam_width > 1:
                decoded = [beam(lp, self.beam_width) for lp in logp]
                seqs = [d[0] for d in decoded]
                conf = [math.exp(d[1] / max(len(d[0]), 1)) for d in decoded]
            else:
                seqs = greedy(logp)
                conf = logp.max(-1).values.mean(-1).exp().tolist()
            for j, i in enumerate(idx):
                texts[i] = self.charset.decode(seqs[j])
                scores[i] = float(conf[j])
        return texts, scores


def ctc_targets(charset, texts):
    encoded = [charset.encode(t) for t in texts]
    lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    flat = torch.zeros((len(encoded), max(int(lengths.max()), 1)), dtype=torch.long)
    for i, e in enumerate(encoded):
        if e:
            flat[i, :len(e)] = torch.tensor(e)
    return flat, lengths
