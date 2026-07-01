"""Beat-window feature extraction for 7K+S BMS charts.

For each sliding beat-window we emit one interpretable feature vector (density /
uniformity / sequence / chord / BMS-specific groups). Wall-clock derived channels
(nps, eff_bpm) keep absolute speed so fast-stair vs stair stay separable.

Scratch columns (chart.scratch_cols — one for SP, two for DP, none for PMS) are
kept OUT of keyboard pattern features and handled on their own channel. LN bodies
don't inflate density: tap features use note heads, LN-jack uses LN-active overlap.

Note: window features are computed in plain Python per window (windows hold
~8-32 notes); fine for a cached research pipeline. Vectorize if a full-corpus
pass gets too slow.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from .parser import Chart

FEATURE_VERSION = 10  # dropped denim_ratio + scratch_key_sim (viz-debug only; unread by tagger/segmenter)
ROW_EPS = 0.005          # s; notes within this gap count as one simultaneous row
_SNAP_NS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 48)


@dataclass
class WindowFeatures:
    beat0: np.ndarray
    beat1: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    X: np.ndarray            # [n_windows, n_features]
    names: list[str]


# ---------- chart-level precompute ----------

def _build_rows(onsets):
    """onsets: sorted list of (time, beat, col). Group by ROW_EPS time gap.
    Returns parallel arrays: rtime, rbeat, rcols (tuple of cols), rsize."""
    rtime, rbeat, rcols = [], [], []
    cur_t = cur_b = None
    cur_cols: list[int] = []
    for t, b, c in onsets:
        if cur_t is None or t - cur_t > ROW_EPS:
            if cur_cols:
                rtime.append(cur_t); rbeat.append(cur_b); rcols.append(tuple(sorted(cur_cols)))
            cur_t, cur_b, cur_cols = t, b, [c]
        else:
            cur_cols.append(c)
    if cur_cols:
        rtime.append(cur_t); rbeat.append(cur_b); rcols.append(tuple(sorted(cur_cols)))
    return np.array(rtime), np.array(rbeat), rcols


def _snap_class(beat: float, tol: float = 0.012):
    r = beat - math.floor(beat)
    if r > 1 - tol:
        r = 0.0
    for n in _SNAP_NS:
        if abs(r * n - round(r * n)) <= tol * n:
            return n
    return None  # off-grid


def _snap_bin(n):
    if n is None: return 5      # off-grid
    if n <= 1: return 0         # 4th (on beat)
    if n == 2: return 1         # 8th
    if n in (3, 6): return 2    # triplet (12/24)
    if n == 4: return 3         # 16th
    return 4                    # 32nd+


def _entropy(counts) -> float:
    s = float(sum(counts))
    if s <= 0:
        return 0.0
    p = np.asarray(counts, float) / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# ---------- per-window features ----------

_NAMES = [
    # A density
    'nps', 'peak_nps', 'mean_simul',
    # C uniformity / snap
    'ioi_cv', 'snap_entropy',
    # D sequence
    'jack_ratio', 'trill_ratio', 'stair_ratio',
    # E2 trajectory / chord-shape (J2 lag-2 Jaccard, span overlap)
    'j2_jaccard', 'span_overlap',
    # F2 scratch
    'scratch_nps',
    # F3 LN
    'ln_coverage', 'ln_active_tap_ratio',
    # F4 soflan (BPM change)
    'eff_bpm', 'stop_time_frac', 'bpm_off_main',
]


def _trill_ratio(cseq) -> float:
    """Fraction of single-notes inside a SUSTAINED two-note alternation (A-B-A-B:
    a run of >=2 consecutive lag-2 matches == >=4 notes). A plain lag-2-match ratio
    over-fires — random streams hit ~1/keys by chance and zigzag stairs (1-2-3-2-1)
    produce ISOLATED matches (run==1) — so it's run-gated, like jack_run_max and the
    stair up>=3 run. run>=2 (not 3) because at the 2-beat window scale a trill is
    often only 4-5 single notes; isolated stair/stream matches still score 0."""
    n = len(cseq)
    if n < 3:
        return 0.0
    covered = run = 0
    for i in range(2, n + 1):                # i == n flushes the final run
        if i < n and cseq[i] == cseq[i - 2] and cseq[i] != cseq[i - 1]:
            run += 1
        else:
            if run >= 2:                     # run of r matches == r+2 notes alternating
                covered += run + 2           # runs are gap-separated, so ranges don't overlap
            run = 0
    return covered / n


def _window(rtime, rbeat, rcols, scr_t, scr_b, lns,
            kb_onsets, b0, b1, t0, t1, stop_events) -> list[float]:
    wall = max(t1 - t0, 1e-6)
    span_beats = b1 - b0
    f = {n: 0.0 for n in _NAMES}
    f['eff_bpm'] = span_beats * 60.0 / wall

    # rows in window
    lo = int(np.searchsorted(rbeat, b0, 'left'))
    hi = int(np.searchsorted(rbeat, b1, 'left'))
    rt = rtime[lo:hi]; rc = rcols[lo:hi]
    sizes = np.array([len(c) for c in rc]) if rc else np.array([])
    n_notes = int(sizes.sum())

    if n_notes:
        f['nps'] = n_notes / wall
        f['mean_simul'] = float(sizes.mean())

    # IOI on row times
    if len(rt) >= 2:
        ioi = np.diff(rt)
        ioi = ioi[ioi > 0]
        if len(ioi):
            f['peak_nps'] = 1.0 / float(ioi.min())
            m = float(ioi.mean())
            f['ioi_cv'] = float(ioi.std() / m) if m > 0 else 0.0

    # snap distribution
    if rc:
        all_beats = [b for c, b in zip(rc, rbeat[lo:hi]) for _ in c]
        bins = np.zeros(6)
        for b in all_beats:
            bins[_snap_bin(_snap_class(b))] += 1
        f['snap_entropy'] = _entropy(bins)

    # jacks: shared columns between adjacent rows
    if len(rc) >= 2:
        jacks = 0
        for a, b in zip(rc[:-1], rc[1:]):
            jacks += len(set(a) & set(b))
        f['jack_ratio'] = jacks / n_notes

    # single-note-row sequence: trill / stair / mean step
    singles = [(b, cols[0]) for b, cols in zip(rbeat[lo:hi], rc) if len(cols) == 1]
    if len(singles) >= 3:
        cseq = [c for _, c in singles]
        steps = np.diff(cseq)
        f['trill_ratio'] = _trill_ratio(cseq)
        # monotone runs (stairs)
        up = dn = 1
        stair_notes = 0
        for s in steps:
            if s > 0:
                up += 1; dn = 1
            elif s < 0:
                dn += 1; up = 1
            else:
                up = dn = 1
            if up >= 3 or dn >= 3:
                stair_notes += 1
        f['stair_ratio'] = stair_notes / len(cseq)

    # trajectory / chord-shape metrics on ALL rows (single+chord, cols 0-6):
    #   j2  = mean lag-2 Jaccard of key sets (high = 2-periodic = denim/trill)
    #   span_overlap = fraction of adjacent rows whose [min,max] spans intersect (denim vs split-trill)
    if len(rc) >= 3:
        sets = [set(c) for c in rc]
        j2 = [len(sets[t] & sets[t + 2]) / len(sets[t] | sets[t + 2]) for t in range(len(sets) - 2)]
        f['j2_jaccard'] = float(np.mean(j2)) if j2 else 0.0
        ov = [max(min(sets[t]), min(sets[t + 1])) <= min(max(sets[t]), max(sets[t + 1]))
              for t in range(len(sets) - 1)]
        f['span_overlap'] = float(np.mean(ov)) if ov else 0.0

    # scratch channel
    slo = int(np.searchsorted(scr_b, b0, 'left'))
    shi = int(np.searchsorted(scr_b, b1, 'left'))
    n_scr = shi - slo
    if n_scr:
        f['scratch_nps'] = n_scr / wall

    # LN / LN-jack
    win_lns = [(c, max(a, t0), min(z, t1)) for (c, a, z, _, _) in lns if a < t1 and z > t0]
    if win_lns:
        # coverage = union of active intervals / wall
        ivs = sorted((a, z) for _, a, z in win_lns)
        cov = 0.0; ce = ivs[0][0]
        for a, z in ivs:
            a = max(a, ce);
            if z > a:
                cov += z - a; ce = z
            else:
                ce = max(ce, z)
        f['ln_coverage'] = min(cov / wall, 1.0)
        # taps occurring while an LN in another column is active (LN-jack signal)
        if kb_onsets and n_notes:
            active_taps = 0
            for (ot, oc) in kb_onsets[int(np.searchsorted([k[0] for k in kb_onsets], t0)):]:
                if ot >= t1:
                    break
                if any(c != oc and a <= ot <= z for c, a, z in win_lns):
                    active_taps += 1
            f['ln_active_tap_ratio'] = active_taps / n_notes

    # STOP freeze fraction (soflan BPM coverage is derived in extract() from eff_bpm)
    win_stop = [s for b, s in stop_events if b0 <= b < b1]
    f['stop_time_frac'] = sum(win_stop) / wall

    return [f[n] for n in _NAMES]


def extract(chart: Chart, win_beats: float = 2.0, hop_beats: float = 0.5) -> WindowFeatures:
    notes = chart.notes
    scratch_cols = chart.scratch_cols
    kb_on = sorted((n.time, n.beat, n.column) for n in notes if n.column not in scratch_cols)
    rtime, rbeat, rcols = _build_rows(kb_on)
    scr = sorted((n.beat, n.time) for n in notes if n.column in scratch_cols)
    scr_b = np.array([b for b, _ in scr]); scr_t = np.array([t for _, t in scr])
    lns = [(n.column, n.time, n.end_time, n.beat, n.end_beat) for n in notes if n.is_ln]
    kb_onsets = [(t, c) for t, _, c in kb_on]  # for ln_active_tap, sorted by time

    tempo = chart.tempo
    b0s = np.arange(0.0, max(chart.total_beats - win_beats * 0.5, hop_beats), hop_beats)
    rows = []
    beat0, beat1, t0a, t1a = [], [], [], []
    for b0 in b0s:
        b1 = min(b0 + win_beats, chart.total_beats)
        if b1 - b0 < win_beats * 0.5:
            continue
        t0 = tempo.beat_to_time(b0); t1 = tempo.beat_to_time(b1)
        rows.append(_window(rtime, rbeat, rcols, scr_t, scr_b, lns, kb_onsets,
                            b0, b1, t0, t1, chart.stop_events))
        beat0.append(b0); beat1.append(b1); t0a.append(t0); t1a.append(t1)
    X = np.array(rows) if rows else np.zeros((0, len(_NAMES)))
    if len(X):  # bpm_off_main: per-window tempo-deviation flag; its segment-MEAN = soflan coverage
        i_eb, i_off = _NAMES.index('eff_bpm'), _NAMES.index('bpm_off_main')
        eb = X[:, i_eb]
        vals, cnts = np.unique(np.round(eb), return_counts=True)
        main = vals[cnts.argmax()]   # main bpm = dominant tempo (mode), robust to brief gimmick spikes
        if main > 0:                 # 0.25 = "off-tempo"; the coverage threshold lives in tagger THR
            X[:, i_off] = (np.abs(eb - main) / main >= 0.25).astype(float)
    return WindowFeatures(np.array(beat0), np.array(beat1), np.array(t0a), np.array(t1a),
                          X, list(_NAMES))


if __name__ == '__main__':  # self-check: trill = sustained alternation only
    assert _trill_ratio([0, 6, 0, 6, 0, 6, 0, 6]) == 1.0, 'pure A-B-A-B trill'
    assert _trill_ratio([0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1]) == 0.0, 'zigzag stair must NOT read as trill'
    assert _trill_ratio([0, 1, 2, 3, 4, 5, 6]) == 0.0, 'ascending stair'
    assert _trill_ratio([0, 6, 0, 1, 2, 3, 4, 5]) == 0.0, 'single isolated match (run 1) rejected'
    assert _trill_ratio([0, 6, 0, 6, 1, 2, 3, 4]) == 0.5, '4-note ABAB (run 2) counts -> 4/8'
    assert _trill_ratio([0, 6, 0, 6, 0, 6, 1, 2, 3]) == 6 / 9, '6-note trill then stops -> 6/9'
    print('ok — trill_ratio counts only sustained run>=2 alternation; stairs/streams reject')
