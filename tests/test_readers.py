import math

import numpy as np
import torch
import torch.nn.functional as F

from docr import ctc, readers


def test_ctc_loss_and_gradient_match_torch():
    torch.manual_seed(0)
    t_max, batch, classes = 20, 5, 9
    logits = torch.randn(t_max, batch, classes, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(1, classes, (batch, 6))
    targets[1, 2] = targets[1, 1]
    input_lengths = torch.tensor([20, 18, 15, 20, 9])
    target_lengths = torch.tensor([6, 4, 3, 0, 2])
    for reduction in ("mean", "sum", "none"):
        ours = ctc.ctc_loss(logits.log_softmax(-1), targets, input_lengths, target_lengths, reduction=reduction)
        theirs = F.ctc_loss(logits.log_softmax(-1), targets, input_lengths, target_lengths, reduction=reduction)
        assert torch.allclose(ours, theirs, atol=1e-8)
    grad_ours, = torch.autograd.grad(ctc.ctc_loss(logits.log_softmax(-1), targets, input_lengths, target_lengths), logits)
    grad_theirs, = torch.autograd.grad(F.ctc_loss(logits.log_softmax(-1), targets, input_lengths, target_lengths), logits)
    assert torch.allclose(grad_ours, grad_theirs, atol=1e-8)


def test_greedy_collapses_repeats_and_blanks():
    path = [0, 3, 3, 0, 3, 5, 5, 0]
    logp = torch.full((len(path), 1, 7), -10.0)
    for t, k in enumerate(path):
        logp[t, 0, k] = 0.0
    assert ctc.greedy(logp.transpose(0, 1)) == [[3, 3, 5]]


def test_beam_finds_sum_over_paths_winner():
    probs = torch.tensor([[0.4, 0.6, 0.0], [0.4, 0.6, 0.0]])
    seq, score = ctc.beam(probs.clamp_min(1e-12).log(), width=4)
    assert seq == [1]
    assert abs(math.exp(score) - (0.6 * 0.6 + 0.6 * 0.4 + 0.4 * 0.6)) < 1e-6
    greedy = ctc.greedy(probs.clamp_min(1e-12).log()[None])
    assert greedy == [[1]]


def test_charset_round_trip_and_folding():
    charset = readers.Charset()
    assert charset.decode(charset.encode("Rp 25.000 (CASH)")) == "Rp 25.000 (CASH)"
    assert charset.normalize("caf\u00e9 \u2013") == "cafe "


def test_crnn_shape():
    crnn = readers.CRNN(96)
    assert crnn(torch.zeros(2, 1, 32, 128)).shape == (2, 32, 96)


def test_reader_wrapper_runs():
    crops = [np.full((20, 60), 200, np.float32), np.full((30, 200), 90, np.float32)]
    texts, scores = readers.Reader(readers.CRNN(96))(crops)
    assert len(texts) == 2 and all(0 <= s <= 1 for s in scores)
