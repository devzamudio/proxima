"""Agregacion de reporte de condiciones de energia por punto, tramo y corrida."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnergyReport:
    """Reporte de condiciones de energia para un punto o tramo."""

    point_index: int
    wec_ok: bool
    nec_ok: bool
    dec_ok: bool
    sec_ok: bool
    violation_magnitudes: dict = field(default_factory=dict)
    margins: dict = field(default_factory=dict)
    hawking_ellis_type: str = "unknown"


@dataclass
class AggregateEnergyReport:
    """Reporte agregado sobre toda la corrida."""

    wec_ok_all: bool
    nec_ok_all: bool
    dec_ok_all: bool
    sec_ok_all: bool
    wec_violated_points: int
    nec_violated_points: int
    dec_violated_points: int
    sec_violated_points: int
    total_points: int
    max_violation_magnitudes: dict = field(default_factory=dict)
    point_reports: list = field(default_factory=list)
    backend: str = "own"


def aggregate_point_reports(reports: list, backend: str = "own") -> AggregateEnergyReport:
    """Combina reportes por punto en un reporte agregado."""
    if not reports:
        return AggregateEnergyReport(
            wec_ok_all=True, nec_ok_all=True, dec_ok_all=True, sec_ok_all=True,
            wec_violated_points=0, nec_violated_points=0,
            dec_violated_points=0, sec_violated_points=0,
            total_points=0, backend=backend,
        )

    wec_ok_all = all(r.wec_ok for r in reports)
    nec_ok_all = all(r.nec_ok for r in reports)
    dec_ok_all = all(r.dec_ok for r in reports)
    sec_ok_all = all(r.sec_ok for r in reports)

    wec_violated = sum(1 for r in reports if not r.wec_ok)
    nec_violated = sum(1 for r in reports if not r.nec_ok)
    dec_violated = sum(1 for r in reports if not r.dec_ok)
    sec_violated = sum(1 for r in reports if not r.sec_ok)

    # Maximos de magnitudes de violacion
    max_mag = {}
    for r in reports:
        for k, v in r.violation_magnitudes.items():
            if k not in max_mag or v > max_mag[k]:
                max_mag[k] = v

    return AggregateEnergyReport(
        wec_ok_all=wec_ok_all,
        nec_ok_all=nec_ok_all,
        dec_ok_all=dec_ok_all,
        sec_ok_all=sec_ok_all,
        wec_violated_points=wec_violated,
        nec_violated_points=nec_violated,
        dec_violated_points=dec_violated,
        sec_violated_points=sec_violated,
        total_points=len(reports),
        max_violation_magnitudes=max_mag,
        point_reports=reports,
        backend=backend,
    )


def aggregate_segment_reports(
    point_reports: list[EnergyReport], n_segments: int = 4
) -> list[EnergyReport]:
    """Agrega reportes puntuales en reportes por tramo de trayectoria."""
    if not point_reports:
        return []
    n_pts = len(point_reports)
    n_seg = min(max(n_segments, 1), n_pts)
    seg_reports: list[EnergyReport] = []
    for s in range(n_seg):
        start = (s * n_pts) // n_seg
        end = ((s + 1) * n_pts) // n_seg
        chunk = point_reports[start:max(end, start + 1)]
        agg = aggregate_point_reports(chunk)
        seg_reports.append(
            EnergyReport(
                point_index=s,
                wec_ok=agg.wec_ok_all,
                nec_ok=agg.nec_ok_all,
                dec_ok=agg.dec_ok_all,
                sec_ok=agg.sec_ok_all,
                violation_magnitudes=agg.max_violation_magnitudes,
            )
        )
    return seg_reports