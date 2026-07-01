"""Tag a BMS chart's pattern segments — the deployment entry point.

CLI:  python -m bmspc.tag <chart.bms> [more...] [--json] [--png] [--timeline]
API:  tag_chart(path) -> dict   (reused by CLI and any future service)

Keymode (7K/14K-DP/9K-PMS) is auto-detected by the parser and reported in the
result. NOTE: the tagger thresholds are calibrated on 7K only — DP/PMS run through
the same detectors best-effort (no labelled data to tune them yet).
"""
from __future__ import annotations
import sys, json, bisect

from .parser import read_text, parse_bms
from .corpus import chart_segments, seg_to_time
from .tagger import classify, vec_to_dict
from .segment import SEG_FEATURES


def _tok(beat, ms):
    """abs beat -> measure token `M` / `M:B` (0-indexed), matching the label format."""
    i = max(0, bisect.bisect_right(ms, beat + 1e-6) - 1)
    b = beat - ms[i]
    return str(i) if abs(b) < 1e-6 else f'{i}:{b:g}'


def tag_chart(path: str) -> dict:
    """Full pipeline for one chart -> structured result (API-ready). Empty/noteless
    charts return with no segments. Note: this parses once here for the keymode
    report and again inside chart_segments (which is feature-cached), so the
    re-parse only costs on a cold cache."""
    text = read_text(path)
    ch = parse_bms(text, is_pms=path.lower().endswith('.pms'))
    out = {'path': path, 'title': ch.meta.get('title', ''), 'keymode': ch.keymode,
           'duration': round(ch.duration, 1), 'segments': []}
    if not ch.notes:
        return out
    meta, wf, segs, vecs = chart_segments(path)
    ms, n = ch.measure_starts, len(wf.beat0)
    for (a, b), (t0, t1), v in zip(segs, seg_to_time(wf, segs), vecs):
        fd = vec_to_dict(v, wf.names)
        out['segments'].append({
            'm0': _tok(wf.beat0[a], ms),
            'm1': _tok(wf.beat0[b] if b < n else wf.beat1[-1], ms),
            't0': round(float(t0), 1), 't1': round(float(t1), 1),
            'tags': classify(fd),
            'texture': {k: round(fd[k], 3) for k in SEG_FEATURES},  # the axes that drove the split
        })
    return out


def _print_text(r):
    print(f"# {r.get('title', '')}  ({r['keymode']}, {r.get('duration', '?')}s)  {r['path']}")
    if not r['segments']:
        print("  no segments (empty or too short)")
        return
    for s in r['segments']:
        print(f"  {s['m0']:>7}-{s['m1']:<7} {s['t0']:6.1f}-{s['t1']:<6.1f}s  {'+'.join(s['tags'])}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    do_json, do_png, do_tl = '--json' in argv, '--png' in argv, '--timeline' in argv
    paths = [a for a in argv if not a.startswith('-')]
    if not paths:
        print('usage: python -m bmspc.tag <chart.bms> [more...] [--json] [--png] [--timeline]', file=sys.stderr)
        return 2
    if do_png or do_tl:                        # renders; matplotlib is heavy, import on demand
        from .viz import plot_segmented, plot_segments_timeline
        for p in paths:
            try:
                if do_png:
                    plot_segmented(p)          # web-bms-style chart + segment overlay
                if do_tl:
                    plot_segments_timeline(p)  # texture-curve diagnostic: are the splits any good?
            except Exception as e:
                print(f"error: {p}: {type(e).__name__}: {e}", file=sys.stderr)
    if do_json or not (do_png or do_tl):       # text/json output, unless it was a render-only run
        results = []
        for p in paths:
            try:
                results.append(tag_chart(p))
            except Exception as e:             # per-file: log + skip, never crash the batch
                print(f"error: {p}: {type(e).__name__}: {e}", file=sys.stderr)
        if do_json:
            print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))
        else:
            for r in results:
                _print_text(r)
    return 0


if __name__ == '__main__':
    sys.exit(main())
