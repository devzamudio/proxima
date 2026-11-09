"""NASA Exoplanet Archive via TAP/ADQL (tabla pscomppars)."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Callable

from proxima.contracts.exceptions import DataProviderUnavailable, DataSchemaMismatch

from ..normalizer import build_normalized_record
from ..governance import chunked_batches
from ..schema import CelestialRecord
from .base import SourceConnector

TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

_COLUMNS = (
    "pl_name", "hostname", "ra", "dec", "sy_dist",
    "st_lum", "st_mass", "st_rad", "st_met", "st_age",
)


def _http_tap_backend(adql: str, timeout_s: float = 60.0) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"query": adql, "format": "json"})
    try:
        with urllib.request.urlopen(f"{TAP_SYNC_URL}?{query}", timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - red
        raise DataProviderUnavailable(f"NASA Exoplanet TAP no disponible: {exc}") from exc



class ExoplanetArchiveConnector(SourceConnector):
    """Estrellas anfitrionas de exoplanetas (un registro por estrella; planetas como alias)."""

    source_name = "exoplanet_tap"
    required_columns = ("hostname", "ra", "dec", "sy_dist")

    def __init__(self, *args: Any, stellar_model: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.stellar_model = stellar_model

    def default_backend(self) -> Callable[[str], list[dict[str, Any]]]:
        return _http_tap_backend

    @staticmethod
    def build_adql(max_distance_pc: float | None = None, hostnames: list[str] | None = None) -> str:
        where = ["sy_dist IS NOT NULL"]
        if max_distance_pc is not None:
            where.append(f"sy_dist <= {float(max_distance_pc):.6g}")
        if hostnames:
            quoted = ",".join("'" + h.replace("'", "''") + "'" for h in sorted(set(hostnames)))
            where.append(f"hostname IN ({quoted})")
        return f"SELECT {', '.join(_COLUMNS)} FROM pscomppars WHERE {' AND '.join(where)} ORDER BY hostname"

    def query(self, max_distance_pc: float | None = None, hostnames: list[str] | None = None) -> list[CelestialRecord]:
        """Una sola consulta ADQL (batch) por llamada; los hostnames se trocean por max_batch_size."""
        if hostnames:
            rows: list[dict] = []
            for chunk in chunked_batches(sorted(set(hostnames)), self.config.max_batch_size):
                adql = self.build_adql(max_distance_pc, chunk)
                rows.extend(self.cached_fetch(adql, lambda q=adql: self.backend(q)))
        else:
            adql = self.build_adql(max_distance_pc)
            rows = self.cached_fetch(adql, lambda: self.backend(adql))
        return self.normalize(rows)

    def normalize(self, rows: list[dict[str, Any]]) -> list[CelestialRecord]:
        # Agrupar planetas por estrella anfitriona.
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            for c in self.required_columns:
                if c not in row:
                    raise DataSchemaMismatch(f"[{self.source_name}] falta columna {c}")
            host = str(row["hostname"]).strip()
            entry = grouped.setdefault(host, {**row, "_planets": []})
            if row.get("pl_name"):
                entry["_planets"].append(str(row["pl_name"]))
        out = []
        for host in sorted(grouped):
            rec = self.row_to_record(grouped[host])
            if rec is not None:
                out.append(rec)
        return out

    def row_to_record(self, row: dict[str, Any]) -> CelestialRecord | None:
        if row.get("ra") is None or row.get("dec") is None or not row.get("sy_dist"):
            return None
        confidence = 0.9
        lum_lsun = 10.0 ** float(row["st_lum"]) if row.get("st_lum") is not None else None
        rad_rsun = row.get("st_rad")
        if lum_lsun is None and self.stellar_model is not None and row.get("st_mass"):
            est = self.stellar_model.estimate(float(row["st_mass"]), row.get("st_age"), row.get("st_met"))
            lum_lsun = est["luminosity_lsun"]
            rad_rsun = rad_rsun or est["radius_rsun"]
            confidence *= 0.95
        elif lum_lsun is None:
            confidence *= 0.8  # luminosidad solar por defecto
        return build_normalized_record(
            canonical_id=str(row["hostname"]).strip(),
            source_name=self.source_name,
            source_version=self.dataset_version,
            object_type="star",
            ra_deg=float(row["ra"]),
            dec_deg=float(row["dec"]),
            distance_pc=float(row["sy_dist"]),
            luminosity_lsun=lum_lsun,
            mass_msun=row.get("st_mass"),
            radius_rsun=rad_rsun,
            metallicity_feh=row.get("st_met"),
            age_gyr=row.get("st_age"),
            aliases=tuple(row.get("_planets", ())),
            confidence_score=confidence,
        )
