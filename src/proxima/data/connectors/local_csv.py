"""Catalogos locales CSV: Sharpless (HII), MWSC (cumulos abiertos), SNRs."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Callable

from proxima.contracts.exceptions import DataProviderUnavailable

from ..normalizer import build_normalized_record
from ..schema import CelestialRecord
from .base import SourceConnector

CATALOG_OBJECT_TYPES: dict[str, str] = {
    "sharpless": "HII_region",
    "mwsc": "open_cluster",
    "snrs": "supernova_remnant",
}


def file_checksum(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _csv_backend(path: Path) -> list[dict[str, Any]]:
    try:
        with Path(path).open(newline="", encoding="utf-8") as fh:
            return [dict(r) for r in csv.DictReader(fh)]
    except OSError as exc:
        raise DataProviderUnavailable(f"No se puede leer catalogo local {path}: {exc}") from exc


def _f(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


class LocalCatalogConnector(SourceConnector):
    """Lee un CSV local; la version del dataset es el checksum del archivo (invalidacion)."""

    source_name = "local_csv"
    required_columns = ("object_id", "ra_deg", "dec_deg", "distance_pc")

    def __init__(self, catalog: str, *args: Any, path: str | Path | None = None, **kwargs: Any) -> None:
        if catalog not in CATALOG_OBJECT_TYPES:
            raise ValueError(f"Catalogo desconocido: {catalog}. Opciones: {sorted(CATALOG_OBJECT_TYPES)}")
        super().__init__(*args, **kwargs)
        self.catalog = catalog
        self.path = Path(path) if path else self.config.catalogs_dir / f"{catalog}.csv"
        if not self.path.exists():
            raise DataProviderUnavailable(f"Catalogo local no encontrado: {self.path}")
        if kwargs.get("dataset_version") is None:
            self.dataset_version = f"{catalog}-sha256:{file_checksum(self.path)[:16]}"

    def default_backend(self) -> Callable[[Path], list[dict[str, Any]]]:
        return _csv_backend

    def load(self) -> list[CelestialRecord]:
        rows = self.cached_fetch(f"{self.catalog}:{self.path.name}", lambda: self.backend(self.path))
        return self.normalize(rows)

    def row_to_record(self, row: dict[str, Any]) -> CelestialRecord | None:
        aliases = tuple(a.strip() for a in str(row.get("aliases") or "").split("|") if a.strip())
        return build_normalized_record(
            canonical_id=str(row["object_id"]).strip(),
            source_name=f"{self.source_name}:{self.catalog}",
            source_version=self.dataset_version,
            object_type=CATALOG_OBJECT_TYPES[self.catalog],
            ra_deg=float(row["ra_deg"]),
            dec_deg=float(row["dec_deg"]),
            distance_pc=float(row["distance_pc"]),
            radius_pc=_f(row.get("radius_pc")),
            luminosity_lsun=_f(row.get("luminosity_lsun")),
            mass_msun=_f(row.get("mass_msun")),
            age_gyr=_f(row.get("age_gyr")),
            metallicity_feh=_f(row.get("feh")),
            aliases=aliases,
            confidence_score=0.85,
        )


def load_all_local_catalogs(config=None, cache=None) -> list[CelestialRecord]:
    records: list[CelestialRecord] = []
    for name in sorted(CATALOG_OBJECT_TYPES):
        records.extend(LocalCatalogConnector(name, config, cache=cache).load())
    return records
