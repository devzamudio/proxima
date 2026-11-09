"""Verificacion Fase 3 (4): reproducibilidad de snapshots y pipeline completo."""

from __future__ import annotations

import json

import pytest

from proxima.contracts.exceptions import DataProviderUnavailable, DataSchemaMismatch
from proxima.data.connectors import ConstantExtinctionProvider, ExoplanetArchiveConnector
from proxima.data.enrichment import ContextEnricher
from proxima.data.pipeline import AstronomicalDataPipeline, SourceJob, local_catalog_jobs
from proxima.data.snapshot import SnapshotStore

from .conftest import EXOPLANET_ROWS

FIXED_TS = "2026-11-09T00:00:00Z"
DEPS = {"python": "3.x"}


def _pipeline(cfg, store, extra_jobs=()):
    exo = ExoplanetArchiveConnector(cfg, backend=lambda q: EXOPLANET_ROWS)
    jobs = local_catalog_jobs(cfg) + [
        SourceJob("exoplanet_tap", lambda: exo.query(), exo.dataset_version),
        *extra_jobs,
    ]
    return AstronomicalDataPipeline(jobs, cfg, enricher=ContextEnricher(ConstantExtinctionProvider()),
                                    snapshot_store=store)


def test_snapshot_roundtrip_and_determinism(tmp_path, data_config):
    store = SnapshotStore(tmp_path / "snaps")
    r1 = _pipeline(data_config, store).run(created_utc=FIXED_TS)
    r2 = _pipeline(data_config, store).run(created_utc=FIXED_TS)
    assert r1.manifest.snapshot_id == r2.manifest.snapshot_id
    assert r1.manifest.records_checksum == r2.manifest.records_checksum
    assert store.list_snapshots() == [r1.manifest.snapshot_id]

    records, manifest = store.load(r1.manifest.snapshot_id)
    assert [r.to_dict() for r in records] == [r.to_dict() for r in r1.records]
    assert manifest.n_records == len(records) == 11
    assert set(manifest.sources) == {"exoplanet_tap", "local_csv:mwsc", "local_csv:sharpless", "local_csv:snrs"}
    assert all(r.ebv_mag > 0 and r.h_column_m2 > 0 for r in records)  # enriquecido


def test_snapshot_detects_record_tampering(tmp_path, data_config):
    store = SnapshotStore(tmp_path)
    res = _pipeline(data_config, store).run(created_utc=FIXED_TS)
    path = tmp_path / res.manifest.snapshot_id / "records.json"
    data = json.loads(path.read_text())
    data[0]["mass_kg"] *= 2
    path.write_text(json.dumps(data))
    with pytest.raises(DataSchemaMismatch):
        store.load(res.manifest.snapshot_id)


def test_snapshot_detects_manifest_tampering(tmp_path, data_config):
    store = SnapshotStore(tmp_path)
    res = _pipeline(data_config, store).run(created_utc=FIXED_TS)
    path = tmp_path / res.manifest.snapshot_id / "manifest.json"
    data = json.loads(path.read_text())
    data["pipeline_version"] = "9.9.9"
    path.write_text(json.dumps(data))
    with pytest.raises(DataSchemaMismatch):
        store.load(res.manifest.snapshot_id)


def test_snapshot_records_dependencies(tmp_path, data_config):
    store = SnapshotStore(tmp_path)
    m = store.save(_pipeline(data_config, None).run().records, pipeline_version="1.0.0")
    assert "python" in m.dependencies and "numpy" in m.dependencies


def test_optional_source_failure_is_traced(tmp_path, data_config):
    def broken():
        raise DataProviderUnavailable("offline")

    store = SnapshotStore(tmp_path)
    res = _pipeline(data_config, store, [SourceJob("gaia_dr3", broken, required=False)]).run(FIXED_TS)
    assert "gaia_dr3" in res.source_errors and res.records


def test_required_source_failure_propagates(tmp_path, data_config):
    def broken():
        raise DataProviderUnavailable("offline")

    with pytest.raises(DataProviderUnavailable):
        _pipeline(data_config, None, [SourceJob("gaia_dr3", broken, required=True)]).run()
