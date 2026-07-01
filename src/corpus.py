"""Corpus batch pass: parse + extract features for many charts (cached), segment
each, and re-aggregate every segment into ONE vector.

Segments — not overlapping windows — are the unit of analysis (overlapping windows
correlate and smear results). Each segment vector = mean of its windows' features
in RAW units (the tagger reads these raw — no corpus-wide standardization).

Note: the cache is one pickle per (chart, win, hop, FEATURE_VERSION). Delete
.cache to recompute after changing the feature extractor.
"""
from __future__ import annotations
import os, pickle, hashlib
import numpy as np

from .parser import read_bms
from .features import extract, FEATURE_VERSION
from .segment import segment, PEN_MULT

_CACHE = os.path.join(os.path.dirname(__file__), os.pardir, '.cache')


def _key(path, win, hop):
    h = hashlib.md5(f"{path}|{win}|{hop}|v{FEATURE_VERSION}".encode()).hexdigest()[:16]
    return os.path.join(_CACHE, h + '.pkl')


def chart_features(path, win=2.0, hop=0.5):
    """(meta, WindowFeatures) for one chart, cached."""
    os.makedirs(_CACHE, exist_ok=True)
    k = _key(path, win, hop)
    if os.path.exists(k):
        with open(k, 'rb') as f:
            return pickle.load(f)
    ch = read_bms(path)
    out = (ch.meta, extract(ch, win, hop))
    with open(k, 'wb') as f:
        pickle.dump(out, f)
    return out


def seg_to_time(wf, segs):
    """Contiguous (t0, t1) per (a, b) window-index segment. End = next segment's
    start window, NOT this segment's last window end (which overshoots by the
    window width and makes adjacent segments overlap)."""
    n = len(wf.t0)
    return [(float(wf.t0[a]), float(wf.t0[b]) if b < n else float(wf.t1[-1])) for a, b in segs]


def chart_segments(path, win=2.0, hop=0.5, pen_mult=PEN_MULT, min_win=2):
    """(meta, wf, segs, vecs) for one chart. segs = list of (a,b) window-index
    ranges; vecs = per-segment raw mean vectors. Shared by the corpus table and
    the overlay viz."""
    meta, wf = chart_features(path, win, hop)
    if wf.X.shape[0] < 5:
        return meta, wf, [], np.zeros((0, len(wf.names)))
    bk, _ = segment(wf, pen_mult=pen_mult)
    edges = [0] + list(bk) + [len(wf.X)]
    segs = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a >= min_win]
    vecs = np.array([wf.X[a:b].mean(0) for a, b in segs]) if segs else np.zeros((0, len(wf.names)))
    return meta, wf, segs, vecs
