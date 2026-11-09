"""Fixtures de Fase 3: backends falsos deterministas (sin red)."""

from __future__ import annotations

import pytest

from proxima.data.cache import MultiLevelCache
from proxima.data.config import DataLayerConfig


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class CountingBackend:
    """Envuelve una funcion y cuenta llamadas; puede fallar las primeras N veces."""

    def __init__(self, fn, fail_first: int = 0, exc: type[Exception] = ConnectionError) -> None:
        self.fn = fn
        self.calls = 0
        self.fail_first = fail_first
        self.exc = exc
        self.args_log: list = []

    def __call__(self, *args):
        self.calls += 1
        self.args_log.append(args)
        if self.calls <= self.fail_first:
            raise self.exc("falla transitoria simulada")
        return self.fn(*args)


EXOPLANET_ROWS = [
    {"pl_name": "Proxima Cen b", "hostname": "Proxima Cen", "ra": 217.3934657, "dec": -62.6761821,
     "sy_dist": 1.30119, "st_lum": -2.81, "st_mass": 0.12, "st_rad": 0.14, "st_met": 0.21, "st_age": 4.85},
    {"pl_name": "Proxima Cen d", "hostname": "Proxima Cen", "ra": 217.3934657, "dec": -62.6761821,
     "sy_dist": 1.30119, "st_lum": -2.81, "st_mass": 0.12, "st_rad": 0.14, "st_met": 0.21, "st_age": 4.85},
    {"pl_name": "eps Eri b", "hostname": "eps Eri", "ra": 53.2326873, "dec": -9.4582611,
     "sy_dist": 3.2026, "st_lum": None, "st_mass": 0.82, "st_rad": 0.76, "st_met": -0.08, "st_age": 0.8},
]

GAIA_ROWS = [
    {"source_id": 5853498713190525696, "ra": 217.39232147200883, "dec": -62.67607511676666,
     "parallax": 768.0665391873573, "parallax_over_error": 2400.0, "lum_flame": None,
     "mass_flame": None, "radius_flame": None, "mh_gspphot": None, "age_flame": None},
    {"source_id": 5164707970261630080, "ra": 53.228431633398, "dec": -9.458172174559,
     "parallax": 310.5772, "parallax_over_error": 2000.0, "lum_flame": 0.34,
     "mass_flame": 0.82, "radius_flame": 0.74, "mh_gspphot": -0.1, "age_flame": 1.0},
    {"source_id": 1, "ra": 10.0, "dec": 10.0, "parallax": -1.0, "parallax_over_error": 1.0,
     "lum_flame": None, "mass_flame": None, "radius_flame": None, "mh_gspphot": None, "age_flame": None},
]

SIMBAD_ROWS = {
    "Proxima Centauri": {"query_name": "Proxima Centauri", "main_id": "NAME Proxima Centauri",
                         "ra": 217.42894222, "dec": -62.67949458, "plx_value": 768.0665,
                         "ids": "Gaia DR3 5853498713190525696|GJ 551|NAME Proxima Cen|V* V645 Cen",
                         "otype": "Star"},
    "eps Eri": {"query_name": "eps Eri", "main_id": "* eps Eri", "ra": 53.23268708, "dec": -9.45825869,
                "plx_value": 310.5773, "ids": "Gaia DR3 5164707970261630080|HD 22049|GJ 144",
                "otype": "Star"},
}


def fake_simbad(names):
    return [SIMBAD_ROWS[n] for n in names if n in SIMBAD_ROWS]


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def data_config() -> DataLayerConfig:
    return DataLayerConfig.load()


@pytest.fixture
def disk_cache(tmp_path, clock) -> MultiLevelCache:
    return MultiLevelCache(disk_dir=tmp_path / "cache", memory_max_entries=64, clock=clock)
