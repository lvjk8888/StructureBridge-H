"""Explicit local input helpers for extracted evaluation functions."""
import gzip,json
from pathlib import Path

def need(ok, message):
    if not ok:
        raise ValueError(message)

def jlines(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
