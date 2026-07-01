"""Self-check for the BMS parser. Run: python -m src.test_parser"""
from .parser import parse_bms, SCRATCH_COL

# measure 0 (#000): 4 keys on col0 (ch11) at beats 0,1,2,3 -> @120bpm = 0,.5,1,1.5
# measure 1 (#001): an LN on col1 (ch12) head beat4 -> tail (ZZ) beat5
# measure 2 (#002): half length (2 beats); scratch (ch16) at start = beat8;
#                   BPM 120->240 (hex F0) at frac .5 = beat9
SYNTH = """
#PLAYER 1
#TITLE Synth
#BPM 120
#LNOBJ ZZ
#00011:01010101
#00112:01ZZ0000
#00202:0.5
#00216:01000000
#00203:00F0
"""


def test_basic():
    c = parse_bms(SYNTH)
    col0 = [n for n in c.notes if n.column == 0]
    assert len(col0) == 4, len(col0)
    assert [round(n.beat, 3) for n in col0] == [0, 1, 2, 3]
    assert [round(n.time, 3) for n in col0] == [0.0, 0.5, 1.0, 1.5], [n.time for n in col0]

    # LN on col1: head at beat4, closed by ZZ at beat5. ZZ is not its own note.
    col1 = [n for n in c.notes if n.column == 1]
    assert len(col1) == 1, len(col1)
    ln = col1[0]
    assert ln.is_ln and round(ln.beat, 3) == 4 and round(ln.end_beat, 3) == 5

    # measure 2 is half-length: starts at beat 8 (4+4), scratch sits there.
    sc = [n for n in c.notes if n.column == SCRATCH_COL]
    assert len(sc) == 1 and round(sc[0].beat, 3) == 8.0, sc[0].beat

    # BPM change 120->240 at beat 9 (measure2, frac .5 of a 2-beat measure).
    assert any(round(b, 2) == 9.0 and v == 240 for b, v in c.bpm_events), c.bpm_events
    # measure 2 spans beats 8..10 -> total 10 beats.
    assert round(c.total_beats, 3) == 10.0, c.total_beats


def test_stop_freezes_time():
    # one beat at 120bpm = 0.5s. STOP 48 units = 1 beat freeze = 0.5s.
    src = """#BPM 120
#STOP01 48
#00009:0001000000000000
#00011:0000000001000000"""
    c = parse_bms(src)
    # note at beat 2 (ch11, slot 4 of 8 in measure 0) sits after the stop at
    # beat 0.5; its wall time = 2*0.5 + 0.5(stop) = 1.5s.
    n = next(x for x in c.notes if x.column == 0)
    assert round(n.time, 3) == 1.5, n.time


if __name__ == '__main__':
    test_basic()
    test_stop_freezes_time()
    print('parser self-check OK')
