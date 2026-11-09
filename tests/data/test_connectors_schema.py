"""Verificacion Fase 3 (1): pruebas de esquema por conector."""

from __future__ import annotations

import math

import pytest

from proxima.contracts.exceptions import DataProviderUnavailable, DataSchemaMismatch, RateLimitExceeded
from proxima.data.connectors import (
    BAYESTAR19_TO_EBV,
    BayestarExtinctionProvider,
    ExoplanetArchiveConnector,
    GaiaDR3Connector,
    LocalCatalogConnector,
    MistStellarModelProvider,
    SimbadConnector,
    main_sequence_backend,
)
from proxima.data.normalizer import PARSEC_TO_M, SOLAR_LUMINOSITY_W, galactic_to_icrs
from proxima.data.schema import CelestialRecord, validate_record_schema

from .conftest import EXOPLANET_ROWS, GAIA_ROWS, CountingBackend, fake_simbad

NO_SLEEP = {"sleeper": lambda s: None}


def _assert_valid(records: list[CelestialRecord]) -> None:
    assert records
    for r in records:
        validate_record_schema(r)
        x, y, z = r.xyz_icrs_m
        assert math.isclose(math.sqrt(x * x + y * y + z * z), r.distance_m, rel_tol=1e-12)


def test_exoplanet_connector_schema_and_host_grouping(data_config):
    conn = ExoplanetArchiveConnector(data_config, backend=lambda adql: EXOPLANET_ROWS, **NO_SLEEP)
    recs = conn.query(max_distance_pc=10.0)
    _assert_valid(recs)
    by_id = {r.canonical_id: r for r in recs}
    assert set(by_id) == {"Proxima Cen", "eps Eri"}
    prox = by_id["Proxima Cen"]
    assert set(prox.aliases) == {"Proxima Cen b", "Proxima Cen d"}
    assert math.isclose(prox.distance_m, 1.30119 * PARSEC_TO_M)
    assert math.isclose(prox.luminosity_w, 10 ** -2.81 * SOLAR_LUMINOSITY_W)
    # eps Eri sin st_lum => luminosidad por defecto, confianza reducida
    assert by_id["eps Eri"].confidence_score < prox.confidence_score


def test_exoplanet_adql_is_batch_and_escaped():
    adql = ExoplanetArchiveConnector.build_adql(5.0, ["O'Brien", "b", "a"])
    assert "FROM pscomppars" in adql and "sy_dist <= 5" in adql
    assert "'O''Brien'" in adql and adql.index("'a'") < adql.index("'b'")


def test_exoplanet_uses_mist_when_luminosity_missing(data_config):
    mist = MistStellarModelProvider(data_config, backend=main_sequence_backend)
    conn = ExoplanetArchiveConnector(data_config, backend=lambda q: EXOPLANET_ROWS, stellar_model=mist)
    eps = {r.canonical_id: r for r in conn.query()}["eps Eri"]
    assert math.isclose(eps.luminosity_w, 0.82**3.5 * SOLAR_LUMINOSITY_W, rel_tol=1e-9)


def test_gaia_connector_schema_and_drops_bad_parallax(data_config):
    backend = CountingBackend(lambda adql, async_: GAIA_ROWS)
    conn = GaiaDR3Connector(data_config, backend=backend, **NO_SLEEP)
    recs = conn.query_cone(217.4, -62.7, 1.0, top=5000)
    _assert_valid(recs)
    assert len(recs) == 2  # parallax negativa descartada
    assert backend.args_log[0][1] is True  # carga grande -> job asincrono
    assert all(r.canonical_id.startswith("Gaia DR3 ") for r in recs)


def test_gaia_source_ids_are_chunked(data_config):
    from dataclasses import replace

    cfg = replace(data_config, max_batch_size=2)
    backend = CountingBackend(lambda adql, async_: [])
    GaiaDR3Connector(cfg, backend=backend).query_source_ids([5, 4, 3, 2, 1])
    assert backend.calls == 3


def test_simbad_batch_schema_and_aliases(data_config):
    conn = SimbadConnector(data_config, backend=fake_simbad)
    recs = conn.query_batch(["eps Eri", "Proxima Centauri"])
    _assert_valid(recs)
    prox = {r.canonical_id: r for r in recs}["NAME Proxima Centauri"]
    assert "GJ 551" in prox.aliases and "Proxima Centauri" in prox.aliases


def test_simbad_microquery_loop_guard(data_config):
    conn = SimbadConnector(data_config, backend=fake_simbad)
    names = ["Proxima Centauri", "eps Eri", "x", "y"]
    with pytest.raises(RateLimitExceeded):
        for n in names:
            conn.query_object(n)


def test_local_catalog_connectors_schema(data_config):
    for name, otype in [("sharpless", "HII_region"), ("mwsc", "open_cluster"), ("snrs", "supernova_remnant")]:
        conn = LocalCatalogConnector(name, data_config)
        recs = conn.load()
        _assert_valid(recs)
        assert len(recs) == 3
        assert all(r.object_type == otype for r in recs)
        assert conn.dataset_version.startswith(f"{name}-sha256:")


def test_local_catalog_missing_column_raises(tmp_path, data_config):
    bad = tmp_path / "mwsc.csv"
    bad.write_text("object_id,ra_deg,dec_deg\nX,1,2\n", encoding="utf-8")
    with pytest.raises(DataSchemaMismatch):
        LocalCatalogConnector("mwsc", data_config, path=bad).load()


def test_connector_retries_then_succeeds(data_config):
    backend = CountingBackend(lambda q: EXOPLANET_ROWS, fail_first=2)
    conn = ExoplanetArchiveConnector(data_config, backend=backend, **NO_SLEEP)
    assert conn.query()
    assert backend.calls == 3


def test_connector_unavailable_after_retries(data_config):
    backend = CountingBackend(lambda q: EXOPLANET_ROWS, fail_first=99)
    conn = ExoplanetArchiveConnector(data_config, backend=backend, **NO_SLEEP)
    with pytest.raises(DataProviderUnavailable):
        conn.query()
    assert backend.calls == data_config.max_retries


def test_connector_rejects_non_tabular_payload(data_config):
    conn = ExoplanetArchiveConnector(data_config, backend=lambda q: {"not": "a list"}, **NO_SLEEP)
    with pytest.raises(DataSchemaMismatch):
        conn.query()


def test_bayestar_conversion_batching_and_cache(data_config):
    backend = CountingBackend(lambda coords: [1.0 for _ in coords])
    prov = BayestarExtinctionProvider(data_config, backend=backend)
    pts = [(10.0, 20.0, 100 * PARSEC_TO_M), (11.0, 21.0, 200 * PARSEC_TO_M), (10.0, 20.0, 100 * PARSEC_TO_M)]
    ebv = prov.ebv_batch(pts)
    assert ebv == pytest.approx([BAYESTAR19_TO_EBV] * 3)
    assert backend.calls == 1 and len(backend.args_log[0][0]) == 2  # deduplicado + lote
    prov.ebv_batch(pts)
    assert backend.calls == 1  # todo desde cache


def test_galactic_center_maps_to_known_icrs():
    ra, dec, ft = galactic_to_icrs(0.0, 0.0)
    assert ra == pytest.approx(266.405, abs=1e-2) and dec == pytest.approx(-28.936, abs=1e-2)
    assert ft.marco_origen == "GALACTIC" and ft.marco_destino == "ICRS"
