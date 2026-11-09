"""Pruebas de la suite de validacion integral, casos canonicos y no-regresion."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from proxima.contracts.types import (
    Frame,
    RouteSegment,
    SkyCoordinates,
    WarpParameterVector,
)
from proxima.data.snapshot import SnapshotStore
from proxima.validation import IntegralValidator


def test_canonical_cases_execution():
    validator = IntegralValidator()

    # Caso canonico 0.5c
    params = WarpParameterVector(
        bubble_radius_m=100.0,
        shell_thickness_m=5.0,
        velocity_fraction_c=0.5,
        shift_amplitude=1.0,
        lapse_floor=0.01,
    )
    sky = SkyCoordinates(ra_deg=83.8, dec_deg=9.9, distance_pc=400.0, frame=Frame.ICRS)
    seg = RouteSegment(start_xyz_m=(0, 0, 0), end_xyz_m=(1e15, 0, 0), frame=Frame.LOCAL_CARTESIAN_3P1)

    res = validator.run_canonical_case("bm_0.5c", params, sky, seg)

    assert res.case_name == "bm_0.5c"
    assert res.status == "success"
    assert res.delta_t_s > 0.0
    assert res.delta_tau_s > 0.0
    assert res.total_fitness > 0.0
    assert res.risk_score >= 0.0


def test_regression_suite_without_and_with_baseline(tmp_path):
    validator = IntegralValidator()

    # Corrida sin linea base previa
    rep1 = validator.run_regression_suite()
    assert rep1.n_cases == 2
    assert rep1.all_passed is True

    # Guardar linea base
    baseline_path = tmp_path / "baseline_cases.json"
    baseline_dict = {
        c.case_name: {
            "delta_t_s": c.delta_t_s,
            "delta_tau_s": c.delta_tau_s,
            "total_fitness": c.total_fitness,
            "risk_score": c.risk_score,
        }
        for c in rep1.cases
    }
    baseline_path.write_text(json.dumps(baseline_dict, indent=2), encoding="utf-8")

    # Corrida de regresion contra la linea base guardada
    rep2 = validator.run_regression_suite(baseline_file=baseline_path, tol_rel=1e-5)
    assert rep2.all_passed is True
    for case_name, diffs in rep2.differences.items():
        for metric, diff in diffs.items():
            assert diff <= 1e-5, f"{case_name} {metric} difirio {diff}"


def test_snapshot_reproducibility_integral(tmp_path):
    from proxima.data.pipeline import local_catalog_jobs, AstronomicalDataPipeline
    from proxima.data.config import DataLayerConfig

    cfg = DataLayerConfig.load()
    store = SnapshotStore(tmp_path / "snaps")
    pipeline = AstronomicalDataPipeline(local_catalog_jobs(cfg), cfg, snapshot_store=store)

    run_res = pipeline.run(created_utc="2026-11-09T00:00:00Z")
    snapshot_id = run_res.manifest.snapshot_id

    validator = IntegralValidator()
    is_reproducible = validator.verify_snapshot_reproducibility(store, snapshot_id)
    assert is_reproducible is True
