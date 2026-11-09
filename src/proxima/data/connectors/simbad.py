"""SIMBAD via astroquery.simbad: solo consultas por lote (anti-microconsultas)."""

from __future__ import annotations

from typing import Any, Callable

from proxima.contracts.exceptions import DataProviderUnavailable

from ..governance import SimbadLoopGuard, chunked_batches
from ..normalizer import build_normalized_record
from ..schema import CelestialRecord
from .base import SourceConnector, table_to_rows


def _astroquery_simbad_backend(names: list[str]) -> list[dict[str, Any]]:
    try:
        from astroquery.simbad import Simbad  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DataProviderUnavailable("astroquery no instalado: pip install astroquery") from exc
    simbad = Simbad()
    simbad.add_votable_fields("parallax", "ids", "otype")
    table = simbad.query_objects(names)
    if table is None:
        return []
    rows = table_to_rows(table)
    out = []
    for name, row in zip(names, rows):
        low = {k.lower(): v for k, v in row.items()}
        out.append({
            "query_name": name,
            "main_id": low.get("main_id"),
            "ra": low.get("ra"),
            "dec": low.get("dec"),
            "plx_value": low.get("plx_value", low.get("plx")),
            "ids": low.get("ids"),
            "otype": low.get("otype"),
        })
    return out


class SimbadConnector(SourceConnector):
    source_name = "simbad"
    required_columns = ("main_id", "ra", "dec", "plx_value")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loop_guard = SimbadLoopGuard(self.config.simbad_max_sequential_single_queries)

    def default_backend(self) -> Callable[[list[str]], list[dict[str, Any]]]:
        return _astroquery_simbad_backend

    def query_batch(self, names: list[str]) -> list[CelestialRecord]:
        """Resuelve nombres en lotes (una llamada remota por lote, cacheada)."""
        unique = sorted({n.strip() for n in names if n and n.strip()})
        rows: list[dict] = []
        for chunk in chunked_batches(unique, self.config.max_batch_size):
            self.loop_guard.record_query(len(chunk))
            signature = "objects:" + "|".join(chunk)
            rows.extend(self.cached_fetch(signature, lambda c=chunk: self.backend(c)))
        return self.normalize(rows)

    def query_object(self, name: str) -> CelestialRecord | None:
        """Consulta individual; protegida por SimbadLoopGuard ante loops."""
        res = self.query_batch([name])
        return res[0] if res else None

    def row_to_record(self, row: dict[str, Any]) -> CelestialRecord | None:
        plx = row.get("plx_value")
        if plx is None or float(plx) <= 0.0 or row.get("ra") is None or row.get("dec") is None:
            return None
        aliases = [a.strip() for a in str(row.get("ids") or "").split("|") if a.strip()]
        if row.get("query_name"):
            aliases.append(str(row["query_name"]))
        return build_normalized_record(
            canonical_id=str(row["main_id"]).strip(),
            source_name=self.source_name,
            source_version=self.dataset_version,
            object_type=str(row.get("otype") or "star"),
            ra_deg=float(row["ra"]),
            dec_deg=float(row["dec"]),
            parallax_mas=float(plx),
            aliases=tuple(dict.fromkeys(aliases)),
            confidence_score=0.75,  # sin fotometria fisica: L/M/R solares por defecto
        )
