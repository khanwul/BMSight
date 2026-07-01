"""Heuristic absolute-threshold pattern tagger.

Operates on a segment's RAW mean-feature vector and emits a SET of tags using
ABSOLUTE thresholds. Absolute (not corpus-relative) bars are needed because
relative z-scoring inflates rare-but-above-average features — e.g. a jumpstream
(j2≈0.3) drifts toward the denim region even though true denim needs j2≥0.55.

Design: true denim = 2-periodic (j2) ∧ hand-overlap (span) ∧ chords, NOT jack.
Single-note stair/trill via the ratio detectors. Scratch / soflan / long are
orthogonal channels, always checked.

Thresholds live in THR for tuning against the hand-labelled eval set.
"""
from __future__ import annotations

THR = {
    'rest_nps': 2.5,                         # below this = rest (too sparse)
    'scratch_nps': 3.5,                      # scratches/sec: fast continuous scratch, not slow backbeat (~2% high-BPM quarter-note FP accepted)
    'soflan_cov': 0.15, 'soflan_stop': 0.05,   # cov = fraction of segment windows ≥25% off the main BPM
    'long_cov': 0.45, 'long_tap': 0.30,      # raised from 0.30/0.15: cut long over-fire (P0.43->0.51) from tap-less sustained LNs + coarse segments spreading long onto non-long frames
    'jack': 0.25,                            # jack guard (keeps jacks out of denim)
    'chord_simul': 1.8,
    'denim_j2': 0.55, 'denim_span': 0.50,    # true denim
    'stair': 0.16, 'trill': 0.15,            # single-note ratio detectors; stair leans high (precision over recall) since no threshold cleanly separates stair from stream (THR-stuck: 0.16 is the marginal optimum, lower over-fires on streams' incidental runs). trill_ratio is run-gated (non-trill ~0); 0.15 maximises overall Jaccard on the eval set (catches diluted trills, rejects streams/stairs)
    'stream_nps': 6.0,                       # dense flow; 8.0 dropped genuine nps-6-8 streams to mix. Single-note-density gating (to keep chord-heavy mix out) was tested and scored WORSE — 'stream' covers chord/jump-streams too, which density gating wrongly drops
}

# most-specific first (orders the multi-label output / picks a primary tag)
_RANK = ['denim', 'stair', 'trill', 'jack', 'chord',
         'long', 'scratch', 'soflan', 'stream', 'rest', 'mix']


def classify(feats: dict) -> list:
    """feats: {feature_name: raw_value}. Returns an ordered tag list (specific-first)."""
    get = feats.get
    nps, simul = get('nps', 0.0), get('mean_simul', 0.0)
    tags: set[str] = set()
    # Orthogonal channels are density-independent, so they bypass the rest gate:
    # a slow soflan intro, a sustained LN, or scratch-only all have low keyboard
    # nps but aren't rest. (nps counts keyboard notes only — scratch excluded.)
    if get('scratch_nps', 0) >= THR['scratch_nps']:   # fast continuous scratch, not a slow backbeat
        tags.add('scratch')
    if get('bpm_off_main', 0) >= THR['soflan_cov'] or get('stop_time_frac', 0) >= THR['soflan_stop']:
        tags.add('soflan')
    if get('ln_coverage', 0) >= THR['long_cov'] or get('ln_active_tap_ratio', 0) >= THR['long_tap']:
        tags.add('long')
    # Too sparse to read keyboard structure: emit only the orthogonal tags, else rest.
    if nps < THR['rest_nps']:
        return sorted(tags, key=_RANK.index) or ['rest']
    is_jack = get('jack_ratio', 0) >= THR['jack']
    chordy = simul >= THR['chord_simul']
    periodic = get('j2_jaccard', 0) >= THR['denim_j2']

    if is_jack:
        tags.add('jack')
    # true denim: 2-periodic ∧ hand-overlap ∧ chords ∧ not jack
    if periodic and get('span_overlap', 0) >= THR['denim_span'] and chordy and not is_jack:
        tags.update(['denim', 'chord'])
    # single-note stair / trill via the ratio detectors
    if get('stair_ratio', 0) >= THR['stair']:
        tags.add('stair')
    if get('trill_ratio', 0) >= THR['trill'] and not is_jack:
        tags.add('trill')
    # chord oscillation that isn't denim (split-trill) reads as trill+chord
    if periodic and chordy and 'denim' not in tags and not is_jack:
        tags.update(['trill', 'chord'])
    # plain chords with no specific sequence structure
    if chordy and 'denim' not in tags:
        tags.add('chord')
    # stream: dense but no dominant keyboard structure. Pattern-exclusion stays:
    # making stream a co-occurring density marker (no exclusion) over-fires (P 0.73->0.53),
    # because GT does NOT consistently add +stream to dense jack/chord/etc. regions.
    if nps >= THR['stream_nps'] and not (tags & {'jack', 'stair', 'trill', 'denim'}):
        tags.add('stream')

    if not tags:
        tags.add('mix')
    return sorted(tags, key=_RANK.index)


def vec_to_dict(vec, names) -> dict:
    return {n: float(vec[i]) for i, n in enumerate(names)}


if __name__ == '__main__':  # self-check: soflan rule (bpm_off_main = seg-mean of per-window off-tempo flag = coverage)
    base = {'nps': 10.0}                                                 # clears the rest gate
    assert 'soflan' in classify({**base, 'bpm_off_main': 0.20}), 'cov 0.20 should fire'
    assert 'soflan' not in classify({**base, 'bpm_off_main': 0.10}), 'cov 0.10 (blip) must NOT fire'
    assert 'soflan' in classify({**base, 'stop_time_frac': 0.10}), 'STOP channel should fire'
    assert 'soflan' not in classify(base), 'constant BPM must be clean'
    # regression: sparse section with a tempo gimmick is soflan, NOT swallowed as rest
    assert classify({'nps': 1.0, 'bpm_off_main': 0.20}) == ['soflan'], 'sparse+soflan → soflan, not rest'
    assert classify({'nps': 1.0}) == ['rest'], 'sparse+nothing → rest'
    # scratch = fast continuous scratch (≥3.5/s): slow backbeat (low rate) must NOT fire
    assert 'scratch' in classify({**base, 'scratch_nps': 6.0}), 'scratch 6/s should fire'
    assert 'scratch' not in classify({**base, 'scratch_nps': 2.0}), 'scratch 2/s (backbeat) must NOT fire'
    # trill: run-gated trill_ratio (sustained alternation) fires even when diluted; ~0 (stream/stair) does not
    assert 'trill' in classify({**base, 'trill_ratio': 0.2}), 'diluted-but-real trill should fire'
    assert 'trill' not in classify({**base, 'trill_ratio': 0.05}), 'near-zero (stream/stair floor) must NOT fire'
    print('ok — soflan + scratch + trill fire, slow/blip/diluted rejected, orthogonal channels survive rest gate')
