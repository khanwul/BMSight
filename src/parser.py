"""BMS parser for 7K + scratch charts.

Output: a Chart with notes (absolute beat + wall-clock time + column, optional
LN end), a beat<->time tempo map, and raw BPM/STOP events for the soflan
(BPM-change) feature channel. Pure stdlib.

Handles: key channels 11-19 (+16 scratch), LN via #LNOBJ and via 5x channels,
measure-length scale (ch 02), BPM change (ch 03 integer-hex / ch 08 #BPMxx
table), STOP (ch 09). Excludes BGM (01), BGA, mines, invisible notes.

Note: 7K-SP only. 2P channels (21-29 / 5x-2P), pop'n (.pms), and the CN/HCN-vs-LN
distinction are dropped — add when 14K/DP support is needed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from bisect import bisect_right, bisect_left
import re

NUM_KEYS = 7
SCRATCH_COL = 7
NCOLS = 8  # 0..6 keys, 7 scratch

# 7K single-play layout -> column index.
_KEY_CH = {'11': 0, '12': 1, '13': 2, '14': 3, '15': 4, '18': 5, '19': 6, '16': 7}
_LN_CH = {'51': 0, '52': 1, '53': 2, '54': 3, '55': 4, '58': 5, '59': 6, '56': 7}

_CHAN_RE = re.compile(r'#(\d{3})([0-9A-Za-z]{2}):(.*)')
_HEAD_RE = re.compile(r'#(\w+)\s*(.*)')


@dataclass
class Note:
    beat: float
    time: float
    column: int                  # 0..6 keys, 7 scratch
    end_beat: float | None = None  # set if long note
    end_time: float | None = None

    @property
    def is_ln(self) -> bool:
        return self.end_beat is not None


@dataclass
class TempoMap:
    """Piecewise beat<->time. Tempo segments are stop-free; stops are added as
    a separate step function so a note exactly on a stop beat is unaffected
    (the stop only delays what comes after)."""
    seg_beat: list[float]
    seg_time: list[float]
    seg_bpm: list[float]
    stop_beat: list[float] = field(default_factory=list)
    stop_cum: list[float] = field(default_factory=lambda: [0.0])  # prefix sums

    def bpm_at(self, beat: float) -> float:
        i = max(bisect_right(self.seg_beat, beat) - 1, 0)
        return self.seg_bpm[i]

    def beat_to_time(self, beat: float) -> float:
        i = max(bisect_right(self.seg_beat, beat) - 1, 0)
        t = self.seg_time[i] + (beat - self.seg_beat[i]) * 60.0 / self.seg_bpm[i]
        j = bisect_left(self.stop_beat, beat)  # stops strictly before `beat`
        return t + self.stop_cum[j]


@dataclass
class Chart:
    notes: list[Note]
    tempo: TempoMap
    bpm_events: list[tuple[float, float]]   # (beat, bpm), raw, incl. initial @0
    stop_events: list[tuple[float, float]]  # (beat, freeze_seconds)
    total_beats: float
    measure_starts: list[float]             # absolute beat at each measure start
    meta: dict

    @property
    def duration(self) -> float:
        return self.tempo.beat_to_time(self.total_beats)


def _build_tempo(bpm_changes: list[tuple[float, float]], init_bpm: float) -> TempoMap:
    seg_beat, seg_time, seg_bpm = [0.0], [0.0], [init_bpm]
    cur_beat, cur_time, cur_bpm = 0.0, 0.0, init_bpm
    for b, bpm in bpm_changes:
        if b < cur_beat:
            continue
        if b == cur_beat:                 # re-change at same beat: last one wins
            cur_bpm = bpm
            seg_bpm[-1] = bpm
            continue
        cur_time += (b - cur_beat) * 60.0 / cur_bpm
        cur_beat, cur_bpm = b, bpm
        seg_beat.append(b); seg_time.append(cur_time); seg_bpm.append(bpm)
    return TempoMap(seg_beat, seg_time, seg_bpm)


def parse_bms(text: str) -> Chart:
    headers: dict[str, str] = {}
    bpm_table: dict[str, float] = {}    # '01' -> bpm
    stop_table: dict[str, float] = {}   # '01' -> units (1/192 whole note)
    lnobj: str | None = None
    chan: dict[tuple[int, str], list[str]] = {}
    measure_scale: dict[int, float] = {}
    max_measure = 0

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] != '#':
            continue
        chan_m = _CHAN_RE.match(line)
        if chan_m:
            meas = int(chan_m.group(1)); ch = chan_m.group(2).upper(); body = chan_m.group(3).strip()
            max_measure = max(max_measure, meas)
            if ch == '02':                      # measure-length scale (plain float)
                try:
                    measure_scale[meas] = float(body)
                except ValueError:
                    pass
            else:
                chan.setdefault((meas, ch), []).append(body)
            continue
        head_m = _HEAD_RE.match(line)
        if not head_m:
            continue
        key, val = head_m.group(1).upper(), head_m.group(2).strip()
        if key.startswith('BPM') and len(key) == 5:        # #BPMxx
            try: bpm_table[key[3:]] = float(val)
            except ValueError: pass
        elif key.startswith('STOP') and len(key) == 6:     # #STOPxx
            try: stop_table[key[4:]] = float(val)
            except ValueError: pass
        elif key == 'LNOBJ':
            lnobj = val.upper()
        else:
            headers[key] = val

    n_meas = max_measure + 1
    mbeats = [4.0 * measure_scale.get(i, 1.0) for i in range(n_meas)]
    mstart = [0.0] * (n_meas + 1)
    for i in range(n_meas):
        mstart[i + 1] = mstart[i] + mbeats[i]
    total_beats = mstart[n_meas]

    def objects(ch: str):
        """Yield (absolute_beat, pair) for every non-empty object in channel `ch`,
        across all measures. Overlays repeated lines for the same measure."""
        for meas in range(n_meas):
            lines = chan.get((meas, ch))
            if not lines:
                continue
            slot: dict[float, str] = {}
            for body in lines:
                n = len(body) // 2
                if n == 0:
                    continue
                for j in range(n):
                    pair = body[2 * j:2 * j + 2]
                    if pair != '00':
                        slot[j / n] = pair
            for frac in sorted(slot):
                yield mstart[meas] + frac * mbeats[meas], slot[frac]

    # --- timing track ---
    try:
        init_bpm = float(headers.get('BPM', '130'))
    except ValueError:
        init_bpm = 130.0
    bpm_changes: list[tuple[float, float]] = []   # raw
    stops_units: list[tuple[float, float]] = []
    for beat, pair in objects('03'):
        try: bpm_changes.append((beat, float(int(pair, 16))))
        except ValueError: pass
    for beat, pair in objects('08'):
        v = bpm_table.get(pair.upper())
        if v is not None: bpm_changes.append((beat, v))
    for beat, pair in objects('09'):
        v = stop_table.get(pair.upper())
        if v is not None: stops_units.append((beat, v))
    bpm_changes.sort(); stops_units.sort()

    init_c = init_bpm if 0 < init_bpm <= 1e6 else 250.0
    clean = []
    for b, v in bpm_changes:
        if v <= 0 or v > 1e6:
            v = 250.0          # clamp gimmick BPM for the time map; raw value kept in bpm_events
        clean.append((b, v))
    tempo = _build_tempo(clean, init_c)

    stop_events = [(b, units / 48.0 * 60.0 / tempo.bpm_at(b)) for b, units in stops_units]
    stop_events.sort()
    tempo.stop_beat = [b for b, _ in stop_events]
    cum, acc = [0.0], 0.0
    for _, s in stop_events:
        acc += s; cum.append(acc)
    tempo.stop_cum = cum

    # --- notes ---
    notes: list[Note] = []
    for ch, col in _KEY_CH.items():               # key channels (+ #LNOBJ tails)
        col_notes: list[Note] = []
        for beat, pair in sorted(objects(ch)):
            if lnobj and pair.upper() == lnobj:
                if col_notes and not col_notes[-1].is_ln:
                    col_notes[-1].end_beat = beat
                    col_notes[-1].end_time = tempo.beat_to_time(beat)
                continue
            col_notes.append(Note(beat, tempo.beat_to_time(beat), col))
        notes.extend(col_notes)
    for ch, col in _LN_CH.items():                # explicit 5x LN channels (head/tail pairs)
        events = sorted(objects(ch))
        i = 0
        while i < len(events):
            head_b = events[i][0]
            if i + 1 < len(events):
                tail_b = events[i + 1][0]
                notes.append(Note(head_b, tempo.beat_to_time(head_b), col, tail_b, tempo.beat_to_time(tail_b)))
                i += 2
            else:
                notes.append(Note(head_b, tempo.beat_to_time(head_b), col))
                i += 1

    notes.sort(key=lambda n: (n.beat, n.column))
    meta = {
        'title': headers.get('TITLE', ''), 'artist': headers.get('ARTIST', ''),
        'genre': headers.get('GENRE', ''), 'playlevel': headers.get('PLAYLEVEL', ''),
        'init_bpm': init_bpm,
    }
    return Chart(notes, tempo, [(0.0, init_bpm)] + bpm_changes, stop_events,
                 total_beats, mstart, meta)


def read_text(path: str) -> str:
    """Decode a BMS file (BMS are Shift-JIS as often as UTF-8)."""
    with open(path, 'rb') as f:
        data = f.read()
    for enc in ('utf-8', 'cp932'):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode('latin-1')


def read_bms(path: str) -> Chart:
    return parse_bms(read_text(path))
