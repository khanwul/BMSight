"""self-check for parser keymode detection (python -m bmspc.test_tag)."""
from .parser import parse_bms


def _km(body: str, is_pms: bool = False) -> str:
    return parse_bms('#BPM 120\n' + body, is_pms=is_pms).keymode


if __name__ == '__main__':
    assert _km('#00118:0101\n') == '7k', '1P keys only → 7k'
    assert _km('#00111:0101\n#00112:0101\n') == '7k', '5K rides the 7k template'
    assert _km('#00111:0101\n#00121:0101\n#00126:0101\n') == '14k', '2P keys + scratch → 14k (DP)'
    assert _km('#00111:0101\n#00122:0101\n#00125:0101\n') == '9k', 'PMS signature (2P keys, no 21/scratch) → 9k'
    assert _km('#00111:0101\n', is_pms=True) == '9k', '.pms hint → 9k'
    # scratch/key columns land where the layout says
    dp = parse_bms('#BPM 120\n#00111:0101\n#00121:0101\n#00126:0101\n')
    assert dp.num_keys == 14 and dp.scratch_cols == frozenset({14, 15}), (dp.num_keys, dp.scratch_cols)
    assert parse_bms('#BPM 120\n#00111:0101\n', is_pms=True).scratch_cols == frozenset(), 'PMS has no scratch'
    print('ok — keymode detection: 7k / 14k(dp) / 9k(pms), scratch columns placed')
