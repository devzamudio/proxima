"""GAIA DR3 via astroquery.gaia (jobs asincronos para cargas grandes)."""

from __future__ import annotations

from typing import Any, Callable

from proxima.contracts.exceptions import DataProviderUnavailable

from ..governance import chunked_batches
from ..normalizer import build_normalized_record
from ..schema import CelestialRecord
from .base import SourceConnector, table_to_rows

_SELECT = (
    "SELECT g.source_id, g.ra, g.dec, g.parallax, g.parallax_over_error, "
    "ap.lum_flame, ap.mass_flame, ap.radius_flame, ap.mh_gspphot, ap.age_flame "
    "FROM gaiadr3.gaia_source AS g "
    "LEFT JOIN gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id"
)

ASYNC_ROW_THRESHOLD = 2000


def _astroquery_gaia_backend(adql: str, async_: bool) -> list[dict[str, Any]]:
    try:
        from astroquery.gaia import Gaia  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DataProviderUnavailable("astroquery no instalado: pip install astroquery") from exc
    job = Gaia.launch_job_async(adql) if async_ else Gaia.launch_job(adql)
    return table_to_rows(job.get_results())


class GaiaDR3Connector(SourceConnector):
    source_name = "gaia_dr3"
    required_columns = ("source_id", "ra", "dec", "parallax")

    def default_backend(self) -> Callable[[str, bool], list[dict[str, Any]]]:
        return _astroquery_gaia_backend

    @staticmethod
    def build_cone_adql(ra_deg: float, dec_deg: float, radius_deg: float,
                        min_parallax_mas: float = 0.0, top: int = 2000) -> str:
        return (
            f"SELECT TOP {int(top)} * FROM ({_SELECT}) AS t "
            f"WHERE 1 = CONTAINS(POINT('ICRS', t.ra, t.dec), "
            f"CIRCLE('ICRS', {ra_deg:.8f}, {dec_deg:.8f}, {radius_deg:.8f})) "
            f"AND t.parallax > {min_parallax_mas:.6g} ORDER BY t.source_id"
        )

    @staticmethod
    def build_ids_adql(source_ids: list[int]) -> str:
        ids = ",".join(str(int(s)) for s in sorted(set(source_ids)))
        return f"{_SELECT} WHERE g.source_id IN ({ids}) ORDER BY g.source_id"

    def query_cone(self, ra_deg: float, dec_deg: float, radius_deg: float,
                   min_parallax_mas: float = 0.0, top: int = 2000) -> list[CelestialRecord]:
        adql = self.build_cone_adql(ra_deg, dec_deg, radius_deg, min_parallax_mas, top)
        use_async = top >= ASYNC_ROW_THRESHOLD
        rows = self.cached_fetch(adql, lambda: self.backend(adql, use_async))
        return self.normalize(rows)

    def query_source_ids(self, source_ids: list[int]) -> list[CelestialRecord]:
        rows: list[dict] = []
        for chunk in chunked_batches(sorted(set(source_ids)), self.config.max_batch_size):
            adql = self.build_ids_adql(chunk)
            use_async = len(chunk) >= self.config.max_batch_size
            rows.extend(self.cached_fetch(adql, lambda q=adql, a=use_async: self.backend(q, a)))
        return self.normalize(rows)

    def row_to_record(self, row: dict[str, Any]) -> CelestialRecord | None:
        plx = row.get("parallax")
        if plx is None or float(plx) <= 0.0 or row.get("ra") is None or row.get("dec") is None:
            return None  # sin distancia fisica utilizable
        poe = row.get("parallax_over_error")
        confidence = 0.7 if poe is None else max(0.1, min(1.0, float(poe) / 10.0))
        if row.get("lum_flame") is None:
            confidence *= 0.8
        return build_normalized_record(
            canonical_id=f"Gaia DR3 {int(row['source_id'])}",
            source_name=self.source_name,
            source_version=self.dataset_version,
            object_type="star",
            ra_deg=float(row["ra"]),
            dec_deg=float(row["dec"]),
            parallax_mas=float(plx),
            luminosity_lsun=row.get("lum_flame"),
            mass_msun=row.get("mass_flame"),
            radius_rsun=row.get("radius_flame"),
            metallicity_feh=row.get("mh_gspphot"),
            age_gyr=row.get("age_flame"),
            confidence_score=confidence,
        )
