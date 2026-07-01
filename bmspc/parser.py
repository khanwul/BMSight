"""BMS parser for 7K/14K-DP + scratch and 9K-PMS charts.

Output: a Chart with notes (absolute beat + wall-clock time + column, optional
LN end), a beat<->time tempo map, and raw BPM/STOP events for the soflan
(BPM-change) feature channel. Pure stdlib.

Handles: 1P key channels 11-19 (+16 scratch), 2P channels 21-29 (+26 scratch) for
DP, and the pop'n 9-button layout; LN via #LNOBJ and via 5x/6x channels;
measure-length scale (ch 02); BPM change (ch 03 integer-hex / ch 08 #BPMxx table);
STOP (ch 09). Excludes BGM (01), BGA, mines, invisible notes.

Keymode is auto-detected from the channels present (see _layout): 5K/10K ride the
7K/14K column templates (unused lanes stay empty). The CN/HCN-vs-LN distinction is
still dropped.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from bisect import bisect_right, bisect_left
import re, itertools

NUM_KEYS = 7      # 7K default; a chart's real value is chart.num_keys
SCRATCH_COL = 7   # 7K default; a chart's real set is chart.scratch_cols
NCOLS = 8

# Keymode -> (key channels in play order, scratch channels). Columns are keys
# first (0..K-1) then scratch. 5K/10K ride the 7K/14K templates (extra lanes stay
# empty). The LN channel for note channel 'NX' is '(N+4)X' (1P 5X, 2P 6X).
_LAYOUTS = {
    '7k':  (['11', '12', '13', '14', '15', '18', '19'], ['16']),
    '14k': (['11', '12', '13', '14', '15', '18', '19',
             '21', '22', '23', '24', '25', '28', '29'], ['16', '26']),
    '9k':  (['11', '12', '13', '14', '15', '22', '23', '24', '25'], []),   # pop'n, no scratch
}
_P2 = {f'{a}{b}' for a in '26' for b in '123456789'}   # 2P note (2X) + 2P LN (6X) channels
_SCR = {'16', '26'}


def _ln_ch(ch: str) -> str:
    return f'{int(ch[0]) + 4}{ch[1]}'   # note channel -> its LN channel (11->51, 21->61)


def _layout(used: set[str], is_pms: bool):
    """(keymode, note_ch->col, ln_ch->col, scratch_cols) from the channels present.
    pop'n (no scratch, no 2P key 21) is only distinguished from DP by the .pms hint
    or its channel signature — a real DP chart uses a scratch and key 21."""
    p2 = used & _P2
    if is_pms or (p2 and not (used & _SCR) and '21' not in used and '61' not in used):
        km = '9k'
    elif p2:
        km = '14k'
    else:
        km = '7k'
    keys, scr = _LAYOUTS[km]
    note_ch = {c: i for i, c in enumerate(keys)}
    note_ch.update({c: len(keys) + i for i, c in enumerate(scr)})
    ln_ch = {_ln_ch(c): col for c, col in note_ch.items()}
    return km, note_ch, ln_ch, frozenset(note_ch[c] for c in scr)

_CHAN_RE = re.compile(r'#(\d{3})([0-9A-Za-z]{2}):(.*)')
_HEAD_RE = re.compile(r'#(\w+)\s*(.*)')


@dataclass
class Note:
    beat: float
    time: float
    column: int                  # keys 0..num_keys-1, then scratch (see Chart.scratch_cols)
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
    keymode: str = '7k'                     # '7k' | '14k' (DP) | '9k' (PMS)
    num_keys: int = NUM_KEYS                # keyboard columns (scratch excluded)
    scratch_cols: frozenset = frozenset({SCRATCH_COL})

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


def parse_bms(text: str, is_pms: bool = False) -> Chart:
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
    mstart = list(itertools.accumulate(mbeats, initial=0.0))
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
    tempo.stop_cum = list(itertools.accumulate((s for _, s in stop_events), initial=0.0))

    # --- notes ---
    keymode, note_ch, ln_ch, scratch_cols = _layout({ch for _, ch in chan}, is_pms)
    notes: list[Note] = []
    for ch, col in note_ch.items():               # key + scratch channels (+ #LNOBJ tails)
        col_notes: list[Note] = []
        for beat, pair in sorted(objects(ch)):
            if lnobj and pair.upper() == lnobj:
                if col_notes and not col_notes[-1].is_ln:
                    col_notes[-1].end_beat = beat
                    col_notes[-1].end_time = tempo.beat_to_time(beat)
                continue
            col_notes.append(Note(beat, tempo.beat_to_time(beat), col))
        notes.extend(col_notes)
    for ch, col in ln_ch.items():                 # explicit 5x/6x LN channels (head/tail pairs)
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
                 total_beats, mstart, meta, keymode, len(note_ch) - len(scratch_cols), scratch_cols)


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
    return parse_bms(read_text(path), is_pms=path.lower().endswith('.pms'))
