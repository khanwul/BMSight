"""Per-chart segmentation: standardize window features → PELT changepoint
detection → segment boundaries.

Boundaries are placed on TEXTURE change — note density + repetition degree — not
on pattern shape: shape is the tagger's job, so segmenting on it too just chops a
steady passage every time the hand-shape drifts. Per-chart z-score is fine —
corpus-wide standardization is only needed for cross-chart clustering.

Note: PELT(l2) on standardized means. l2 = piecewise-constant mean shift =
"density/repetition level changed". Switch to the rbf kernel if non-mean structure
(e.g. variance-only) changes need catching.
"""
from __future__ import annotations
import os
import numpy as np
import ruptures as rpt

from .features import WindowFeatures

# Texture axes only: note density + repetition degree. Pattern-SHAPE ratios
# (jack/trill/stair/denim/col_entropy) are deliberately excluded — those are what
# the tagger classifies a segment AS, not what defines where a segment begins/ends.
SEG_FEATURES = [
    # density: notes/sec, burst rate, chord thickness
    'nps', 'peak_nps', 'mean_simul',
    # repetition, pattern-agnostic:
    #   j2_jaccard = lag-2 self-similarity (jack/trill high, flowing run low)
    #   ioi_cv = timing steadiness · snap_entropy = subdivision variety
    'j2_jaccard', 'ioi_cv', 'snap_entropy',
]
# Note: soflan/scratch/LN are dropped from boundary-driving on purpose — a tempo
# swing still shifts nps → boundary. Re-add eff_bpm if pure-soflan needs own segs.

# Segmentation granularity knob (global, set via the BMS_PEN_MULT env var): lower =
# finer/more segments, higher = coarser. Default 3.0 targets ~10-16s segments; 1.0
# over-segments into 1-2s fragments (the d·log(n) penalty is small at only d=6 features).
PEN_MULT = float(os.environ.get('BMS_PEN_MULT', '3.0'))

# Optional pattern-transition sub-splitting (BMS_SUB_PATTERN=1). OFF by default.
# When on, chart_segments adds boundaries where the SMOOTHED pattern shape
# (jack/trill/stair) changes, on TOP of the texture ones — finer, pattern-aware
# segments for downstream per-segment analysis. It trades boundary precision for
# recall (more, sometimes spurious, splits), so it stays opt-in, not the default.
# The additive layer is deliberate: feeding shape straight into the texture vector
# was tested and corrupts texture boundaries on density/tempo charts (shape is flat
# OR high-and-variable there), so shape only ADDS sub-splits, never moves texture ones.
SHAPE_FEATURES = ['jack_ratio', 'trill_ratio', 'stair_ratio']
SUB_PATTERN = os.environ.get('BMS_SUB_PATTERN', '') not in ('', '0', 'false', 'False')
SHAPE_PEN_MULT = float(os.environ.get('BMS_SHAPE_PEN', '3.0'))
SHAPE_SMOOTH = int(os.environ.get('BMS_SHAPE_SMOOTH', '5'))


def select(wf: WindowFeatures) -> np.ndarray:
    idx = [wf.names.index(n) for n in SEG_FEATURES]
    return wf.X[:, idx]


def standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (X - mu) / sd


def segment(wf: WindowFeatures, pen_mult: float = PEN_MULT, min_size: int = 2):
    """Return (boundary_window_indices, standardized_seg_matrix).
    Penalty = pen_mult · d · log(n) (BIC-like; auto-selects segment count)."""
    Xs = standardize(select(wf))
    n, d = Xs.shape
    if n < 2 * min_size + 1:
        return [], Xs
    pen = pen_mult * d * np.log(n)
    bkps = rpt.Pelt(model='l2', min_size=min_size).fit(Xs).predict(pen=pen)
    return bkps[:-1], Xs  # drop the trailing n


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Centered box low-pass over the window axis (edge-padded). k<=1 = identity."""
    if k <= 1:
        return x
    pad = k // 2
    return np.convolve(np.pad(x, pad, mode='edge'), np.ones(k) / k, mode='valid')


def shape_boundaries(wf: WindowFeatures, pen_mult: float = SHAPE_PEN_MULT,
                     k: int = SHAPE_SMOOTH, min_size: int = 2):
    """PELT changepoints on SMOOTHED pattern-shape features — the pattern TRANSITIONS.
    Low-pass first: raw shape ratios jitter window-to-window and would over-split.
    Kept separate from segment(): these only ADD sub-splits to the texture boundaries
    (see hier_boundaries), never replace them."""
    idx = [wf.names.index(n) for n in SHAPE_FEATURES]
    Xs = standardize(np.column_stack([_smooth(wf.X[:, i], k) for i in idx]))
    n, d = Xs.shape
    if n < 2 * min_size + 1:
        return []
    pen = pen_mult * d * np.log(n)
    return rpt.Pelt(model='l2', min_size=min_size).fit(Xs).predict(pen=pen)[:-1]


def hier_boundaries(wf: WindowFeatures, pen_mult: float = PEN_MULT,
                    shape_pen: float = SHAPE_PEN_MULT, k: int = SHAPE_SMOOTH, min_size: int = 2):
    """Texture boundaries + pattern-shape sub-boundaries (union). A shape breakpoint is
    added only if it doesn't crowd a texture one (>= min_size apart), so every texture
    boundary survives and finer splits appear only at genuine pattern transitions."""
    tb, _ = segment(wf, pen_mult=pen_mult, min_size=min_size)
    n = len(wf.X)
    kept = list(tb)
    for b in sorted(shape_boundaries(wf, shape_pen, k, min_size)):
        if min_size <= b <= n - min_size and all(abs(b - x) >= min_size for x in kept):
            kept.append(b)
    return sorted(set(kept))


if __name__ == '__main__':  # self-check: density splits, pattern-shape does not
    names = list(SEG_FEATURES) + ['jack_ratio', 'trill_ratio', 'stair_ratio']  # + shape feats
    n = 40

    def _wf(col, lo, hi):
        X = np.zeros((n, len(names)))
        X[: n // 2, names.index(col)] = lo
        X[n // 2 :, names.index(col)] = hi
        t = np.arange(n, dtype=float)
        return WindowFeatures(t, t + 2, t, t + 2, X, names)

    # pen_mult pinned, not the default: tests split logic, not seg-count tuning.
    bk_density, _ = segment(_wf('nps', 2.0, 20.0), pen_mult=1.0)
    assert any(abs(b - n // 2) <= 2 for b in bk_density), f"density step missed: {bk_density}"
    bk_shape, _ = segment(_wf('jack_ratio', 0.0, 1.0), pen_mult=1.0)
    assert not bk_shape, f"shape-only step must NOT split: {bk_shape}"

    # optional sub-pattern layer: a shape transition DOES add a boundary; texture ignores it
    assert shape_boundaries(_wf('jack_ratio', 0.0, 1.0), pen_mult=1.0), 'shape step must give a boundary'
    assert not shape_boundaries(_wf('nps', 2.0, 20.0), pen_mult=1.0), 'flat shape -> no shape boundary'
    hb = hier_boundaries(_wf('jack_ratio', 0.0, 1.0), pen_mult=1.0, shape_pen=1.0)
    assert any(abs(b - n // 2) <= 2 for b in hb), f'hier must add the shape boundary: {hb}'
    print('ok — density', bk_density, '| shape ignored by texture', bk_shape, '| hier adds shape', hb)
