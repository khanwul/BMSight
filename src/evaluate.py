"""Validation harness.

Ground truth = a few hand-labelled charts (eval ONLY, never fed to fitting).
Two scores:
  - boundary F-measure (MIREX-style, with a tolerance window): do predicted
    pattern boundaries land near the human ones?
  - per-tag frame P/R/F1 (multi-label): for each tag, over 0.1s frames, do the
    tagger's tag-sets match the human tag-sets? Micro-F1 + mean per-frame
    Jaccard summarise; the per-tag rows are the THR calibration instrument
    (low recall on a tag → its THR is too high, and vice versa).

Predictions come from `tagger.classify` on each segment's raw mean vector.
Boundaries are TAG-SET-change points (adjacent same-tag-set segments merged) —
the right comparison to human pattern boundaries, not the finer raw changepoints.
"""
from __future__ import annotations
import os
import numpy as np

from .segment import PEN_MULT

# fixed vocabulary so labels stay consistent. Consolidated:
# stair (incl. fast stairs), chord (incl. big chords; double-stairs label as chord+stair).
TAGS = ['rest', 'jack', 'stream', 'trill', 'chord', 'long', 'soflan',
        'stair', 'denim', 'scratch', 'mix']


def parse_labels(file):
    """Returns (chart_path, [(start_tok, end_tok, tag)]) with RAW measure tokens.
    A boundary token is `M` (measure M, downbeat) or `M:B` (measure M, beat B);
    both 0-indexed. Resolve to seconds with `resolve(..., chart)`."""
    path, segs = None, []
    for line in open(file, encoding='utf-8'):
        s = line.strip()
        if s.startswith('# chart:'):
            path = s.split('chart:', 1)[1].strip()
        if not s or s.startswith('#'):
            continue
        p = s.split()
        if len(p) >= 3:
            segs.append((p[0], p[1], p[2]))
    return path, segs


def _tok_beat(tok, mstart):
    """`M` or `M:B` (0-indexed) -> absolute beat."""
    if ':' in tok:
        m, b = tok.split(':', 1)
        m, b = int(float(m)), float(b)
    else:
        m, b = int(round(float(tok))), 0.0
    return mstart[min(m, len(mstart) - 1)] + b


def resolve(raw, chart):
    """[(start_tok, end_tok, tag)] -> [(t0, t1, tag)] in seconds."""
    ms = chart.measure_starts
    return [(chart.tempo.beat_to_time(_tok_beat(s, ms)),
             chart.tempo.beat_to_time(_tok_beat(e, ms)), tag) for s, e, tag in raw]


def merge_same(segs):
    """Merge adjacent segments that share a label."""
    out = []
    for t0, t1, lab in segs:
        if out and out[-1][2] == lab and abs(out[-1][1] - t0) < 1e-6:
            out[-1] = (out[-1][0], t1, lab)
        else:
            out.append((t0, t1, lab))
    return out


def boundary_f(gt_b, pred_b, tol=1.0):
    """Greedy one-to-one match of interior boundary times within tol seconds."""
    gt_b, pred_b = sorted(gt_b), sorted(pred_b)
    used = [False] * len(pred_b)
    hit = 0
    for g in gt_b:
        for j, p in enumerate(pred_b):
            if not used[j] and abs(p - g) <= tol:
                used[j] = True
                hit += 1
                break
    P = hit / len(pred_b) if pred_b else (1.0 if not gt_b else 0.0)
    R = hit / len(gt_b) if gt_b else (1.0 if not pred_b else 0.0)
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return F, P, R


def _frame_tags(segs, vocab, hop=0.1, n=None):
    """[n_frames, n_tags] bool coverage from (t0, t1, set-of-tags) segments."""
    end = max(t1 for _, t1, _ in segs)
    m = int(round(end / hop)) + 1
    idx = {t: i for i, t in enumerate(vocab)}
    M = np.zeros((m, len(vocab)), bool)
    for t0, t1, tags in segs:
        a, b = int(round(t0 / hop)), int(round(t1 / hop))
        for t in tags:
            if t in idx:
                M[a:b, idx[t]] = True
    return M if n is None else M[:n]


def evaluate(label_files, win=2.0, hop=0.5, pen_mult=PEN_MULT, tol=1.0, frame_hop=0.1):
    from .corpus import chart_segments, seg_to_time
    from .parser import read_bms
    from .tagger import classify, vec_to_dict
    per_chart = []
    pool_g, pool_p = [], []      # frame×tag bool matrices, pooled over charts
    for lf in label_files:
        path, raw = parse_labels(lf)
        if not raw or not path:
            continue
        gt = merge_same([(t0, t1, frozenset(tag.split('+')))
                         for t0, t1, tag in resolve(raw, read_bms(path))])
        _, wf, segs, vecs = chart_segments(path, win, hop, pen_mult)
        if not segs:
            continue
        pred = merge_same([(s0, s1, frozenset(classify(vec_to_dict(v, wf.names))))
                           for (s0, s1), v in zip(seg_to_time(wf, segs), vecs)])
        F, P, R = boundary_f([s[1] for s in gt[:-1]], [s[1] for s in pred[:-1]], tol)
        g = _frame_tags(gt, TAGS, frame_hop)
        p = _frame_tags(pred, TAGS, frame_hop)
        n = min(len(g), len(p))
        g, p = g[:n], p[:n]
        pool_g.append(g)
        pool_p.append(p)
        jac = _jaccard(g, p)
        per_chart.append((os.path.basename(lf), F, P, R, jac))

    print(f"{'chart':28s} {'bF':>5} {'bP':>5} {'bR':>5} {'Jac':>5}")
    for name, F, P, R, jac in per_chart:
        print(f"{name[:28]:28s} {F:5.2f} {P:5.2f} {R:5.2f} {jac:5.2f}")
    if per_chart:
        arr = np.array([[r[1], r[2], r[3], r[4]] for r in per_chart])
        print(f"{'MEAN':28s} {arr[:,0].mean():5.2f} {arr[:,1].mean():5.2f} "
              f"{arr[:,2].mean():5.2f} {arr[:,3].mean():5.2f}")
    if pool_g:
        G, Pr = np.vstack(pool_g), np.vstack(pool_p)
        _per_tag_table(G, Pr)
    return per_chart


def _jaccard(G, P):
    """Mean per-frame Jaccard of two [n,tags] bool matrices (empty∧empty = 1)."""
    inter = (G & P).sum(1)
    union = (G | P).sum(1)
    return float(np.where(union == 0, 1.0, inter / np.maximum(union, 1)).mean())


def _per_tag_table(G, P):
    """Print per-tag precision/recall/F1 over pooled frames + micro/macro.
    This is the THR calibration instrument: low R on a tag → lower its THR."""
    tp = (G & P).sum(0).astype(float)
    fp = (~G & P).sum(0).astype(float)
    fn = (G & ~P).sum(0).astype(float)
    sup = G.sum(0)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=tp + fn > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=prec + rec > 0)
    print(f"\n{'tag':12s} {'sup':>6} {'P':>5} {'R':>5} {'F1':>5}")
    macro = []
    for i, t in enumerate(TAGS):
        if sup[i] == 0 and (tp[i] + fp[i]) == 0:
            continue                     # tag never appears in GT or prediction
        print(f"{t:12s} {int(sup[i]):6d} {prec[i]:5.2f} {rec[i]:5.2f} {f1[i]:5.2f}")
        if sup[i] > 0:
            macro.append(f1[i])
    mtp, mfp, mfn = tp.sum(), fp.sum(), fn.sum()
    micro = 2 * mtp / (2 * mtp + mfp + mfn) if mtp else 0.0
    print(f"{'micro-F1':12s} {'':6} {'':5} {'':5} {micro:5.2f}   "
          f"macro-F1={np.mean(macro) if macro else 0.0:5.2f}  Jaccard={_jaccard(G, P):.2f}")


if __name__ == '__main__':
    import sys, glob
    _files = sys.argv[1:] or sorted(glob.glob('labels/*.txt'))
    evaluate([f for f in _files if os.path.getsize(f) > 0])
