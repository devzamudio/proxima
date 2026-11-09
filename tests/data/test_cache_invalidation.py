"""Verificacion Fase 3 (3): invalidacion de cache y gobernanza de cuota."""

from __future__ import annotations

import json

import pytest

from proxima.contracts.exceptions import RateLimitExceeded
from proxima.data.cache import CacheKey, MultiLevelCache
from proxima.data.connectors import ExoplanetArchiveConnector, LocalCatalogConnector
from proxima.data.governance import RateLimiter, chunked_batches

from .conftest import EXOPLANET_ROWS, CountingBackend


def _key(**kw):
    base = dict(query_signature="q", dataset_version="v1", pipeline_version="1.0.0", unit_system="SI")
    base.update(kw)
    return CacheKey(**base)


def test_l1_l2_hits(disk_cache, tmp_path, clock):
    disk_cache.put(_key(), {"a": 1})
    assert disk_cache.get(_key()) == {"a": 1}
    assert disk_cache.stats["l1_hits"] == 1
    fresh = MultiLevelCache(disk_dir=tmp_path / "cache", clock=clock)  # L1 vacio
    assert fresh.get(_key()) == {"a": 1}
    assert fresh.stats["l2_hits"] == 1


def test_ttl_expiration(disk_cache, clock):
    disk_cache.put(_key(), [1], ttl_seconds=10.0)
    clock.advance(11.0)
    assert disk_cache.get(_key()) is None
    assert disk_cache.stats["invalidations"] >= 1


@pytest.mark.parametrize("field", ["dataset_version", "pipeline_version", "unit_system"])
def test_key_components_isolate_entries(disk_cache, field):
    disk_cache.put(_key(), [1])
    assert disk_cache.get(_key(**{field: "other"})) is None


def test_checksum_corruption_invalidates(tmp_path, clock):
    cache = MultiLevelCache(disk_dir=tmp_path / "c", clock=clock)
    digest = cache.put(_key(), {"value": 1})
    path = tmp_path / "c" / f"{digest}.json"
    entry = json.loads(path.read_text())
    entry["payload"]["value"] = 999  # manipulacion
    path.write_text(json.dumps(entry))
    fresh = MultiLevelCache(disk_dir=tmp_path / "c", clock=clock)
    assert fresh.get(_key()) is None
    assert not path.exists()


def test_invalidate_by_version(disk_cache):
    disk_cache.put(_key(query_signature="a", dataset_version="v1"), 1)
    disk_cache.put(_key(query_signature="b", dataset_version="v2"), 2)
    removed = disk_cache.invalidate_by_version("v2")
    assert removed >= 1
    assert disk_cache.get(_key(query_signature="b", dataset_version="v2")) == 2
    assert disk_cache.get(_key(query_signature="a", dataset_version="v1")) is None


def test_connector_hits_network_once(data_config, disk_cache):
    backend = CountingBackend(lambda q: EXOPLANET_ROWS)
    conn = ExoplanetArchiveConnector(data_config, cache=disk_cache, backend=backend)
    first = conn.query(10.0)
    second = conn.query(10.0)
    assert backend.calls == 1 and [r.to_dict() for r in first] == [r.to_dict() for r in second]


def test_dataset_version_bump_forces_refetch(data_config, disk_cache):
    backend = CountingBackend(lambda q: EXOPLANET_ROWS)
    ExoplanetArchiveConnector(data_config, cache=disk_cache, backend=backend, dataset_version="v1").query()
    ExoplanetArchiveConnector(data_config, cache=disk_cache, backend=backend, dataset_version="v2").query()
    assert backend.calls == 2


def test_local_csv_checksum_invalidation(tmp_path, data_config, disk_cache):
    src = data_config.catalogs_dir / "mwsc.csv"
    path = tmp_path / "mwsc.csv"
    path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    c1 = LocalCatalogConnector("mwsc", data_config, cache=disk_cache, path=path)
    assert len(c1.load()) == 3
    with path.open("a", encoding="utf-8") as fh:
        fh.write("MWSC_9999,Test,10.0,10.0,100.0,5.0,100.0,50.0,0.1,0.0\n")
    c2 = LocalCatalogConnector("mwsc", data_config, cache=disk_cache, path=path)
    assert c2.dataset_version != c1.dataset_version
    assert len(c2.load()) == 4


def test_rate_limiter_window(clock):
    rl = RateLimiter(max_calls=2, window_seconds=10.0, clock=clock)
    rl.acquire()
    rl.acquire()
    with pytest.raises(RateLimitExceeded):
        rl.acquire()
    clock.advance(10.1)
    rl.acquire()


def test_chunked_batches():
    assert chunked_batches(range(5), 2) == [[0, 1], [2, 3], [4]]
