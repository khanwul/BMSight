"""self-check for the keymode gate (python -m src.test_tag)."""
from .parser import parse_bms
from .tag import keymode


def _km(body: str) -> str:
    text = '#BPM 120\n' + body
    return keymode(text, parse_bms(text))


if __name__ == '__main__':
    assert _km('#00118:0101\n') == '7k', '7K (uses ch18/col5) → 7k'
    assert _km('#00111:0101\n#00121:0101\n') == 'dp/multi', '2P channel (21) → dp/multi'
    assert _km('#00111:0101\n#00112:0101\n') == 'other', '5K (only ch11-12) → other'
    assert _km('') == 'empty', 'no notes → empty'
    print('ok — keymode gate: 7k / dp / other / empty')
