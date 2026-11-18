"""Original validation helpers used by frozen selection."""
from __future__ import annotations
import math

class Stop(ValueError): pass


def need(ok,message):
    if not ok:raise Stop(message)


def finite(x):
    v=float(x);need(math.isfinite(v),'Non-finite score');return v

