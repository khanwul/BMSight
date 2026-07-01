#!/usr/bin/env python3
"""BMS 7K pattern classifier — run this.

  python main.py <chart.bms> [more...]     segments + tags  (text)
  python main.py <chart.bms> --json        JSON: boundaries + tags + split-driving texture
  python main.py <chart.bms> --png         chart render with segment/tag overlay (PNG)
  python main.py <chart.bms> --timeline    segmentation diagnostic: texture curves + boundaries (PNG)

Flags pass straight through to bmspc.tag.main.
"""
import os, sys

# re-exec under the project venv so `python main.py` works without activating it first
_venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv')
_venv = os.path.join(_venv_dir, 'bin', 'python')
if os.path.exists(_venv) and os.path.realpath(sys.prefix) != os.path.realpath(_venv_dir):
    os.execv(_venv, [_venv, *sys.argv])

from bmspc.tag import main

if __name__ == '__main__':
    sys.exit(main())
