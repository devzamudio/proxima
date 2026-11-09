"""Snapshots reproducibles con manifest firmado por checksum SHA-256."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from proxima.contracts.exceptions import DataSchemaMismatch

from .schema import CelestialRecord, validate_record_schema

_TRACKED_DEPENDENCIES = ("numpy", "jax", "pyyaml", "astroquery", "dustmaps", "isochrones", "astropy")


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def records_checksum(records: list[CelestialRecord]) -> str:
    payload = [r.to_dict() for r in sorted(records, key=lambda r: (r.canonical_id, r.source_name))]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def dependency_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in _TRACKED_DEPENDENCIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: str
    created_utc: str
    pipeline_version: str
    unit_system: str
    frame: str
    records_checksum: str
    n_records: int
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    dependencies: dict[str, str] = field(default_factory=dict)
    manifest_checksum: str = ""

    def body(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("manifest_checksum")
        return d

    def compute_checksum(self) -> str:
        return hashlib.sha256(_canonical_json(self.body()).encode("utf-8")).hexdigest()


class SnapshotStore:
    """Persistencia de snapshots: <root>/<snapshot_id>/{records.json, manifest.json}."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        records: list[CelestialRecord],
        *,
        pipeline_version: str,
        unit_system: str = "SI",
        sources: dict[str, dict[str, Any]] | None = None,
        created_utc: str | None = None,
        dependencies: dict[str, str] | None = None,
    ) -> SnapshotManifest:
        for r in records:
            validate_record_schema(r)
        ordered = sorted(records, key=lambda r: (r.canonical_id, r.source_name))
        checksum = records_checksum(ordered)
        # El id depende solo del contenido + version de pipeline => reproducible.
        snapshot_id = hashlib.sha256(f"{checksum}:{pipeline_version}:{unit_system}".encode()).hexdigest()[:16]
        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            created_utc=created_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            pipeline_version=pipeline_version,
            unit_system=unit_system,
            frame="ICRS",
            records_checksum=checksum,
            n_records=len(ordered),
            sources=dict(sorted((sources or {}).items())),
            dependencies=dependencies if dependencies is not None else dependency_versions(),
        )
        manifest = SnapshotManifest(**{**manifest.body(), "manifest_checksum": manifest.compute_checksum()})

        target = self.root / snapshot_id
        target.mkdir(parents=True, exist_ok=True)
        (target / "records.json").write_text(
            json.dumps([r.to_dict() for r in ordered], sort_keys=True, indent=1), encoding="utf-8"
        )
        (target / "manifest.json").write_text(
            json.dumps(asdict(manifest), sort_keys=True, indent=2), encoding="utf-8"
        )
        return manifest

    def load(self, snapshot_id: str) -> tuple[list[CelestialRecord], SnapshotManifest]:
        target = self.root / snapshot_id
        if not (target / "manifest.json").exists():
            raise DataSchemaMismatch(f"Snapshot inexistente: {snapshot_id}")
        manifest = SnapshotManifest(**json.loads((target / "manifest.json").read_text(encoding="utf-8")))
        if manifest.compute_checksum() != manifest.manifest_checksum:
            raise DataSchemaMismatch(f"Manifest alterado en snapshot {snapshot_id}")
        records = [
            CelestialRecord.from_dict(d)
            for d in json.loads((target / "records.json").read_text(encoding="utf-8"))
        ]
        if records_checksum(records) != manifest.records_checksum:
            raise DataSchemaMismatch(f"Checksum de registros no coincide en snapshot {snapshot_id}")
        return records, manifest

    def list_snapshots(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "manifest.json").exists())
