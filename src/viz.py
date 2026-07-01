"""Chart + segment visualization for the classifier.

plot_segmented(path)          -> web-bms-style chart render + segment/tag overlay  (--png)
plot_segments_timeline(path)  -> texture-curve segmentation diagnostic             (--timeline)

Driven by the CLI (`python -m src.tag --png / --timeline`); not run directly.
"""
from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .parser import read_bms, SCRATCH_COL


def _tagcol():
    """tag -> stable tab20 color, shared by every plot fn."""
    from .evaluate import TAGS
    cmap = plt.get_cmap('tab20')
    return {t: cmap(i % 20) for i, t in enumerate(TAGS)}


def _setup_cjk_font():
    """Make CJK chart titles (often Japanese) render — the default font draws them
    as boxes. Register the Noto CJK .ttc by file path; matplotlib resolves .ttc
    collections poorly by family name."""
    import os
    from matplotlib import font_manager, rcParams
    paths = ['/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
             '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
             '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc']
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            font_manager.fontManager.addfont(p)
            name = font_manager.FontProperties(fname=p).get_name()
            rcParams['font.family'] = name
            rcParams['axes.unicode_minus'] = False
            return name
        except Exception:
            pass
    for name in ('Noto Sans CJK KR', 'Noto Sans CJK JP', 'NanumGothic'):  # fallback
        try:
            font_manager.findfont(name, fallback_to_default=False)
            rcParams['font.family'] = name
            rcParams['axes.unicode_minus'] = False
            return name
        except Exception:
            continue
    return None


# ---- web-bms-viewer-style render (Snack-X/web-bms-viewer) + classifier segments ----
# That viewer's layout: 16 beats per column, notes bottom-to-top, lanes left-to-right
# = scratch + keys 1-7. We reproduce it and overlay the classifier's segment
# boundaries + tags — the visual companion to `python -m src.tag`.
_BEATS_PER_COL = 16.0
_LANE_W = 8.0            # scratch + 7 keys occupy x = 0..7 within a column
_COL_W = _LANE_W + 3.0   # + a gap before the next column


def _lane_x(col):
    """parser column (0..6 = key1..7, 7 = scratch) -> x within a column (scratch leftmost)."""
    return 0.0 if col == SCRATCH_COL else col + 1.0


def _key_color(col):
    if col == SCRATCH_COL:
        return 'tab:red'
    return 'tab:blue' if col % 2 else '0.2'     # alternating like BMS keys, both visible on white bg


def _col_spans(b0, b1):
    """Split absolute-beat range [b0, b1) into (column_index, y_lo, y_hi) pieces, one
    per 16-beat column it crosses. y = in-column beat (0..16), drawn bottom-to-top."""
    out = []
    c = int(b0 // _BEATS_PER_COL)
    while c * _BEATS_PER_COL < b1:
        lo, hi = max(b0, c * _BEATS_PER_COL), min(b1, (c + 1) * _BEATS_PER_COL)
        if hi > lo:
            out.append((c, lo - c * _BEATS_PER_COL, hi - c * _BEATS_PER_COL))
        c += 1
    return out


# self-check: a range crossing a column boundary splits into two pieces
assert _col_spans(14.0, 18.0) == [(0, 14.0, 16.0), (1, 0.0, 2.0)], _col_spans(14.0, 18.0)


def plot_segmented(path, out=None):
    """Render a chart web-bms-viewer style (vertical, multi-column, bottom-to-top)
    with the classifier's segment boundaries + tags overlaid. Writes a PNG."""
    import os
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    from .corpus import chart_segments
    from .tagger import classify, vec_to_dict
    _setup_cjk_font()

    meta, wf, segs, vecs = chart_segments(path)
    chart = read_bms(path)
    if not chart.notes:
        print('no notes — nothing to draw:', path); return
    if out is None:                                        # cwd, not next to the chart (data dir may be read-only)
        out = os.path.basename(os.path.splitext(path)[0]) + '_segments.png'

    nb = len(wf.beat0)
    seg_tags = []                                          # (b0, b1, tags) in absolute beats
    for (a, b), v in zip(segs, vecs):
        b0 = float(wf.beat0[a])
        b1 = float(wf.beat0[b]) if b < nb else float(wf.beat1[-1])
        seg_tags.append((b0, b1, classify(vec_to_dict(v, wf.names))))

    max_beat = max([(n.end_beat or n.beat) for n in chart.notes] + [chart.measure_starts[-1]])
    ncol = int(max_beat // _BEATS_PER_COL) + 1
    tagcol = _tagcol()
    fig, ax = plt.subplots(figsize=(max(7.0, ncol * 1.0), 9))

    # tag bands (behind everything), one rectangle per column-slice of each segment
    for b0, b1, tags in seg_tags:
        col = tagcol.get(tags[0], (.6, .6, .6, 1.0))
        for c, ylo, yhi in _col_spans(b0, b1):
            ax.add_patch(mpatches.Rectangle((c * _COL_W - 0.6, ylo), _LANE_W + 0.2, yhi - ylo,
                                            color=col, alpha=0.16, lw=0, zorder=0))

    # measure lines + numbers
    for m, mb in enumerate(chart.measure_starts):
        c = int(mb // _BEATS_PER_COL); y = mb - c * _BEATS_PER_COL
        ax.hlines(y, c * _COL_W - 0.6, c * _COL_W + _LANE_W - 0.4, color='0.85', lw=0.5, zorder=1)
        ax.text(c * _COL_W - 0.75, y, str(m), ha='right', va='bottom', fontsize=4, color='0.6')

    # notes: LN bodies (translucent) under taps (opaque), batched as two collections
    ln, ln_c, tap, tap_c = [], [], [], []
    for n in chart.notes:
        lx, colr = _lane_x(n.column), _key_color(n.column)
        if n.is_ln:
            for c, ylo, yhi in _col_spans(n.beat, n.end_beat):
                ln.append(mpatches.Rectangle((c * _COL_W + lx - 0.45, ylo), 0.9, yhi - ylo)); ln_c.append(colr)
        c = int(n.beat // _BEATS_PER_COL); y = n.beat - c * _BEATS_PER_COL
        # x is compressed ~5.66x vs y on screen, so a wide bar needs height « width: 0.9 x 0.08 ≈ 2:1 on screen
        tap.append(mpatches.Rectangle((c * _COL_W + lx - 0.45, y - 0.04), 0.9, 0.08)); tap_c.append(colr)
    ax.add_collection(PatchCollection(ln, facecolor=ln_c, alpha=0.45, edgecolor='0.5', linewidths=0.3, zorder=2))
    ax.add_collection(PatchCollection(tap, facecolor=tap_c, edgecolor='0.4', linewidths=0.3, zorder=3))

    # segment boundaries (thick lines) + tag labels
    for b0, _, tags in seg_tags:
        c = int(b0 // _BEATS_PER_COL); y = b0 - c * _BEATS_PER_COL
        ax.hlines(y, c * _COL_W - 0.8, c * _COL_W + _LANE_W - 0.2, color='0.05', lw=1.3, zorder=4)
        ax.text(c * _COL_W - 0.5, y + 0.15, '+'.join(tags), fontsize=5, fontweight='bold',
                color=tagcol.get(tags[0], '0.2'), va='bottom', zorder=5)

    used = list(dict.fromkeys(t for _, _, ts in seg_tags for t in ts))
    ax.legend(handles=[mpatches.Patch(color=tagcol.get(t, '0.6'), label=t) for t in used],
              loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=7, title='tags')
    ax.set_xlim(-1.2, (ncol - 1) * _COL_W + _LANE_W + 0.5)
    ax.set_ylim(-0.5, _BEATS_PER_COL + 0.5)
    ax.set_xticks([]); ax.set_yticks(range(0, 17, 4))
    ax.set_ylabel('beat within 16-beat column  (bottom → top)')
    ax.set_title(f"{meta.get('title', '')} — classifier segments ({len(seg_tags)} segs, {ncol} cols)")
    fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches='tight'); plt.close(fig)
    print('wrote', out)


def plot_segments_timeline(path, out=None):
    """Segmentation diagnostic (not the chart): the six texture axes the segmenter
    actually splits on, z-scored on a shared time axis, with segment boundaries
    (dashed) + tag bands. Read it as 'does each boundary sit on a real texture
    change?' — the direct way to judge whether the split is any good."""
    import os
    from .corpus import chart_segments, seg_to_time
    from .segment import SEG_FEATURES, select, standardize
    from .tagger import classify, vec_to_dict
    _setup_cjk_font()

    meta, wf, segs, vecs = chart_segments(path)
    if not segs:
        print('no segments to show:', path); return
    if out is None:
        out = os.path.basename(os.path.splitext(path)[0]) + '_timeline.png'

    tc = (wf.t0 + wf.t1) / 2
    Xs = standardize(select(wf))                       # [n_win, 6] z-scored, same matrix PELT sees
    times = seg_to_time(wf, segs)
    tags = [classify(vec_to_dict(v, wf.names)) for v in vecs]
    tagcol = _tagcol()

    w = min(max(float(wf.t1[-1]) * 0.18, 10.0), 200.0)
    fig, ax = plt.subplots(3, 1, figsize=(w, 6.5), sharex=True, height_ratios=[1, 3.2, 3.2])

    # top: tag band per segment
    for (t0, t1), tg in zip(times, tags):
        ax[0].axvspan(t0, t1, color=tagcol.get(tg[0], (.6, .6, .6, 1.0)), alpha=0.85)
        ax[0].text((t0 + t1) / 2, 0.5, '+'.join(tg), ha='center', va='center',
                   fontsize=7, rotation=90 if (t1 - t0) < 6 else 0)
    ax[0].set_yticks([]); ax[0].set_ylim(0, 1); ax[0].set_ylabel('tag', fontsize=8)

    # middle: the z-scored texture axes — eyeball boundary vs. curve break
    colors = []
    for i, name in enumerate(SEG_FEATURES):
        line, = ax[1].plot(tc, Xs[:, i], lw=1.0, label=name)
        colors.append(line.get_color())
    ax[1].legend(loc='upper right', fontsize=7, ncol=3)
    ax[1].set_ylabel('texture (z-score)'); ax[1].grid(alpha=0.2)

    # bottom: per-segment mean step — the constant value PELT collapsed each segment to
    for (a, b), (t0, t1) in zip(segs, times):
        m = Xs[a:b].mean(0)
        for i in range(len(SEG_FEATURES)):
            ax[2].hlines(m[i], t0, t1, color=colors[i], lw=2.0)
    ax[2].set_ylabel('segment mean (z-score)'); ax[2].set_xlabel('time (s)'); ax[2].grid(alpha=0.2)

    for t0, _ in times:                              # shared boundaries on all panels
        for a in ax:
            a.axvline(t0, color='0.25', lw=0.8, ls='--')
    for a in ax:
        a.axvline(times[-1][1], color='0.25', lw=0.8, ls='--')
    ax[2].set_xlim(0, float(wf.t1[-1]))
    ax[0].set_title(f"{meta.get('title', '')} — segmentation diagnostic ({len(segs)} segs)")
    fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches='tight'); plt.close(fig)
    print('wrote', out)
