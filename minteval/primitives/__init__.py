"""Primitive ("building block") registry.

Each primitive contributes code fragments to fixed hook points of the program
template in ``assembler.py`` plus a precise English ``spec_template`` used for
back-translation. Every parameter is a discrete grid so K_bits is exact.

Hook points (all fragments are written at 4-space indentation and re-indented):
  signal    sets ``sig`` in {-1, 0, 1}; may use/maintain state
  gate      runs when flat and a signal fires; may set ``ok = False``
  size      expression for the entry size magnitude (sizing family only)
  on_entry  runs on the signal bar right after the trade keys are written
  hold      runs every bar while a position is held (updates, may set ``target``)
  exit      runs after ``hold`` while holding; may set ``target = 0.0``
  orders    runs whenever a trade is live; may update ``stop`` / ``take``
  on_exit   runs once when a trade is found to be over (before cleanup)
"""
from __future__ import annotations

from dataclasses import dataclass, field

REGISTRY: dict[str, "Primitive"] = {}


@dataclass
class Primitive:
    id: str
    family: str                       # signal | filter | sizing | risk
    params: dict                      # name -> list of grid values
    spec_template: str
    hooks: dict                       # hook name -> code template (str.format with params)
    trade_keys: tuple = ()            # per-trade state keys to delete when the trade ends
    excludes: tuple = ()              # primitive ids that cannot be combined with this one
    requires_any: tuple = ()          # (only for conditional variants)
    derive: object = None             # optional fn(values) -> extra template fields

    def _vals(self, values, conv):
        out = {k: conv(v) for k, v in values.items()}
        if self.derive is not None:
            out.update(self.derive(values))
        return out

    def render(self, hook: str, values: dict) -> str:
        tpl = self.hooks.get(hook)
        if not tpl:
            return ""
        return tpl.format(**self._vals(values, fmt))

    def describe(self, values: dict) -> str:
        return self.spec_template.format(**self._vals(values, human))


def fmt(v):
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, float):
        r = repr(v)
        return r
    return str(v)


def human(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def primitive(**kw):
    def deco(fn):
        p = Primitive(**kw, hooks=fn())
        REGISTRY[p.id] = p
        return p
    return deco


def by_family(family: str) -> list[Primitive]:
    return [p for p in REGISTRY.values() if p.family == family]


from . import signal, sizing, risk, filters  # noqa: E402,F401  (registration side effects)
