"""Verificacion Fase 3 (2): integridad de cross-match."""

from __future__ import annotations

import random

from proxima.data.connectors import ExoplanetArchiveConnector, GaiaDR3Connector, SimbadConnector
from proxima.data.crossmatch import IdentityResolver, angular_separation_arcsec, normalize_alias
from proxima.data.normalizer import build_normalized_record

from .conftest import EXOPLANET_ROWS, GAIA_ROWS, fake_simbad


def _all_records(cfg):
    return (
        ExoplanetArchiveConnector(cfg, backend=lambda q: EXOPLANET_ROWS).query()
        + GaiaDR3Connector(cfg, backend=lambda q, a: GAIA_ROWS).query_cone(0, 0, 1)
        + SimbadConnector(cfg, backend=fake_simbad).query_batch(["Proxima Centauri", "eps Eri"])
    )


def _rec(cid, ra, dec, dist_pc, aliases=(), source="simbad"):
    return build_normalized_record(canonical_id=cid, source_name=source, source_version="t",
                                   object_type="star", ra_deg=ra, dec_deg=dec, distance_pc=dist_pc,
                                   aliases=aliases)


def test_normalize_alias():
    assert normalize_alias("NAME  Proxima_Cen") == normalize_alias("proxima cen")
    assert normalize_alias("* eps Eri") == "EPS ERI"


def test_angular_separation():
    assert angular_separation_arcsec(10, 0, 10, 1) == __import__("pytest").approx(3600.0, rel=1e-9)


def test_crossmatch_unifies_three_sources(data_config):
    records = _all_records(data_config)
    merged, report = IdentityResolver.from_config(data_config).resolve(records)
    assert report.n_input == len(records) == 6
    assert report.n_output == len(merged) == 2
    # Gaia tiene prioridad como registro primario
    assert all(r.source_name == "gaia_dr3" for r in merged)
    members = sorted(m for ms in report.groups.values() for m in ms)
    assert len(members) == len(set(members)) == len(records)  # particion exacta
    prox = next(r for r in merged if "Proxima Cen" in r.aliases)
    assert {"GJ 551", "Proxima Cen b", "NAME Proxima Centauri"} <= set(prox.aliases)


def test_crossmatch_is_order_independent(data_config):
    records = _all_records(data_config)
    resolver = IdentityResolver.from_config(data_config)
    ref, _ = resolver.resolve(records)
    for seed in range(5):
        shuffled = records[:]
        random.Random(seed).shuffle(shuffled)
        out, _ = resolver.resolve(shuffled)
        assert [r.to_dict() for r in out] == [r.to_dict() for r in ref]


def test_positional_match_respects_distance_tolerance():
    resolver = IdentityResolver(angular_tolerance_arcsec=5.0, distance_rel_tolerance=0.15)
    a = _rec("A", 100.0, 10.0, 100.0)
    b = _rec("B", 100.0 + 1.0 / 3600.0, 10.0, 105.0)   # 1" y 5 % -> mismo objeto
    c = _rec("C", 100.0, 10.0, 300.0)                   # misma linea de vision, otra distancia
    d = _rec("D", 100.0 + 30.0 / 3600.0, 10.0, 100.0)  # 30" -> distinto
    merged, report = resolver.resolve([a, b, c, d])
    assert len(merged) == 3
    assert sorted(len(v) for v in report.groups.values()) == [1, 1, 2]


def test_alias_transitivity():
    resolver = IdentityResolver()
    a = _rec("A", 10, 10, 10, aliases=("X1",))
    b = _rec("B", 50, 50, 50, aliases=("X1", "X2"))
    c = _rec("C", 90, -10, 70, aliases=("x2",))
    merged, _ = resolver.resolve([a, b, c])
    assert len(merged) == 1
