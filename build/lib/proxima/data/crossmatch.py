"""Resolucion de identidades: alias + cross-match posicional (union-find determinista)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

from .schema import CelestialRecord

# Prioridad de fuente al elegir el registro primario de un grupo (menor = preferido).
SOURCE_PRIORITY: dict[str, int] = {
    "gaia_dr3": 0,
    "exoplanet_tap": 1,
    "simbad": 2,
    "local_csv": 3,
}

_PREFIXES = ("NAME ", "* ", "V* ", "** ")


def normalize_alias(alias: str) -> str:
    """Normaliza identificadores: mayusculas, sin prefijos SIMBAD, espacios colapsados."""
    a = str(alias).strip().upper()
    for p in _PREFIXES:
        if a.startswith(p):
            a = a[len(p):]
    a = re.sub(r"[\s_]+", " ", a).strip()
    return a


def angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Separacion angular (formula haversine), en arcsec."""
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    h = math.sin((d2 - d1) / 2.0) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2.0) ** 2
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(h)))) * 3600.0


def _source_rank(record: CelestialRecord) -> int:
    return SOURCE_PRIORITY.get(record.source_name.split(":")[0], 99)


@dataclass
class CrossMatchReport:
    groups: dict[str, list[str]] = field(default_factory=dict)  # canonical -> miembros "fuente/id"
    n_input: int = 0
    n_output: int = 0
    match_reasons: dict[str, list[str]] = field(default_factory=dict)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


class IdentityResolver:
    """Unifica registros de multiples fuentes.

    Dos registros se consideran el mismo objeto si:
      1. comparten un alias normalizado (incluido el canonical_id), o
      2. su separacion angular <= tolerancia y su distancia relativa <= tolerancia.
    """

    def __init__(self, angular_tolerance_arcsec: float = 5.0, distance_rel_tolerance: float = 0.15) -> None:
        self.angular_tolerance_arcsec = float(angular_tolerance_arcsec)
        self.distance_rel_tolerance = float(distance_rel_tolerance)

    @classmethod
    def from_config(cls, config) -> "IdentityResolver":
        return cls(config.angular_tolerance_arcsec, config.distance_rel_tolerance)

    def _positional_match(self, a: CelestialRecord, b: CelestialRecord) -> bool:
        if angular_separation_arcsec(a.ra_deg, a.dec_deg, b.ra_deg, b.dec_deg) > self.angular_tolerance_arcsec:
            return False
        rel = abs(a.distance_m - b.distance_m) / max(a.distance_m, b.distance_m)
        return rel <= self.distance_rel_tolerance

    def resolve(self, records: list[CelestialRecord]) -> tuple[list[CelestialRecord], CrossMatchReport]:
        # Orden determinista independiente del orden de entrada.
        recs = sorted(records, key=lambda r: (_source_rank(r), r.source_name, r.canonical_id))
        n = len(recs)
        uf = _UnionFind(n)
        reasons: dict[int, set[str]] = {}

        alias_owner: dict[str, int] = {}
        for i, r in enumerate(recs):
            for a in (r.canonical_id, *r.aliases):
                key = normalize_alias(a)
                if not key:
                    continue
                if key in alias_owner:
                    uf.union(i, alias_owner[key])
                    reasons.setdefault(i, set()).add(f"alias:{key}")
                else:
                    alias_owner[key] = i

        # Cross-match posicional: barrido ordenado por declinacion.
        tol_deg = self.angular_tolerance_arcsec / 3600.0
        by_dec = sorted(range(n), key=lambda i: recs[i].dec_deg)
        for idx, i in enumerate(by_dec):
            for j in by_dec[idx + 1:]:
                if recs[j].dec_deg - recs[i].dec_deg > tol_deg:
                    break
                if uf.find(i) != uf.find(j) and self._positional_match(recs[i], recs[j]):
                    uf.union(i, j)
                    reasons.setdefault(j, set()).add(f"position:{recs[i].canonical_id}")

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(uf.find(i), []).append(i)

        merged: list[CelestialRecord] = []
        report = CrossMatchReport(n_input=n)
        for members in groups.values():
            members.sort(key=lambda i: (_source_rank(recs[i]), -recs[i].confidence_score, recs[i].canonical_id))
            primary = recs[members[0]]
            aliases: list[str] = []
            for i in members:
                for a in (recs[i].canonical_id, *recs[i].aliases):
                    if a != primary.canonical_id and a not in aliases:
                        aliases.append(a)
            confidence = min(1.0, max(recs[i].confidence_score for i in members) + 0.02 * (len(members) - 1))
            merged.append(replace(primary, aliases=tuple(sorted(aliases)), confidence_score=confidence))
            report.groups[primary.canonical_id] = [f"{recs[i].source_name}/{recs[i].canonical_id}" for i in members]
            report.match_reasons[primary.canonical_id] = sorted(
                {reason for i in members for reason in reasons.get(i, set())}
            )

        merged.sort(key=lambda r: r.canonical_id)
        report.n_output = len(merged)
        return merged, report
