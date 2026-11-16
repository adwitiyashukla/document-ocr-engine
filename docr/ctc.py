import math

import torch


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean"):
    t_max, batch, _ = log_probs.shape
    s_max = int(target_lengths.max()) if len(target_lengths) else 0
    width = 2 * s_max + 1
    device = log_probs.device
    ext = torch.full((batch, width), blank, dtype=torch.long, device=device)
    if s_max:
        ext[:, 1::2] = targets[:, :s_max]
    skip = torch.zeros((batch, width), dtype=torch.bool, device=device)
    if width > 2:
        skip[:, 2:] = (ext[:, 2:] != blank) & (ext[:, 2:] != ext[:, :-2])
    lengths = 2 * target_lengths + 1
    positions = torch.arange(width, device=device)[None, :]
    valid = positions < lengths[:, None]
    neg = torch.tensor(-1e30, device=device, dtype=log_probs.dtype)
    emit = log_probs.gather(2, ext[None].expand(t_max, batch, width))
    alpha = torch.where(positions < 2, emit[0], neg)
    alpha = torch.where(valid, alpha, neg)
    finals = [None] * batch
    ends = (input_lengths - 1).tolist()
    for b in range(batch):
        if ends[b] == 0:
            finals[b] = alpha[b]
    for t in range(1, t_max):
        one = torch.cat([neg.expand(batch, 1), alpha[:, :-1]], dim=1)
        two = torch.cat([neg.expand(batch, 2), alpha[:, :-2]], dim=1) if width > 2 else neg.expand(batch, width)
        two = torch.where(skip, two, neg)
        stacked = torch.stack([alpha, one, two])
        alpha = torch.logsumexp(stacked, dim=0) + emit[t]
        alpha = torch.where(valid, alpha, neg)
        for b in range(batch):
            if ends[b] == t:
                finals[b] = alpha[b]
    losses = []
    for b in range(batch):
        last = int(lengths[b]) - 1
        pair = finals[b][max(last - 1, 0):last + 1]
        losses.append(-torch.logsumexp(pair, dim=0))
    loss = torch.stack(losses)
    if reduction == "mean":
        return (loss / target_lengths.clamp(min=1).to(loss.dtype)).mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def greedy(log_probs, blank=0):
    best = log_probs.argmax(-1).cpu().numpy()
    out = []
    for row in best:
        seq, prev = [], blank
        for k in row:
            if k != blank and k != prev:
                seq.append(int(k))
            prev = k
        out.append(seq)
    return out


def _logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam(log_probs, width=8, blank=0, prune=-12.0):
    t_max, classes = log_probs.shape
    lp = log_probs.cpu().double().numpy()
    beams = {(): (0.0, -math.inf)}
    for t in range(t_max):
        row = lp[t]
        candidates = [k for k in range(classes) if row[k] > prune]
        nxt = {}
        for prefix, (p_blank, p_char) in beams.items():
            total = _logadd(p_blank, p_char)
            for k in candidates:
                p = row[k]
                if k == blank:
                    b, c = nxt.get(prefix, (-math.inf, -math.inf))
                    nxt[prefix] = (_logadd(b, total + p), c)
                    continue
                extended = prefix + (k,)
                b, c = nxt.get(extended, (-math.inf, -math.inf))
                if prefix and prefix[-1] == k:
                    nxt[extended] = (b, _logadd(c, p_blank + p))
                    sb, sc = nxt.get(prefix, (-math.inf, -math.inf))
                    nxt[prefix] = (sb, _logadd(sc, p_char + p))
                else:
                    nxt[extended] = (b, _logadd(c, total + p))
        beams = dict(sorted(nxt.items(), key=lambda item: -_logadd(*item[1]))[:width])
    best = max(beams.items(), key=lambda item: _logadd(*item[1]))
    return list(best[0]), _logadd(*best[1])
