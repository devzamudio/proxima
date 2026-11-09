"""Orquestacion del pipeline de datos: ingestion -> normalizacion -> cross-match -> enriquecimiento -> snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from proxima.contracts.exceptions import DataProviderUnavailable, RateLimitExceeded

from .config import DataLayerConfig
from .crossmatch import CrossMatchReport, IdentityResolver
from .enrichment import ContextEnricher
from .risk_provider import AstronomicalRiskProvider
from .schema import CelestialRecord
from .snapshot import SnapshotManifest, SnapshotStore


@dataclass
class SourceJob:
    """Trabajo de ingestion: `fetch()` devuelve registros normalizados de una fuente."""

    name: str
    fetch: Callable[[], list[CelestialRecord]]
    dataset_version: str = "unversioned"
    required: bool = True


@dataclass
class PipelineResult:
    records: list[CelestialRecord]
    crossmatch: CrossMatchReport
    manifest: SnapshotManifest | None
    source_errors: dict[str, str] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)


class AstronomicalDataPipeline:
    def __init__(
        self,
        jobs: list[SourceJob],
        config: DataLayerConfig | None = None,
        *,
        resolver: IdentityResolver | None = None,
        enricher: ContextEnricher | None = None,
        snapshot_store: SnapshotStore | None = None,
    ) -> None:
        self.config = config or DataLayerConfig.default_config()
        self.jobs = jobs
        self.resolver = resolver or IdentityResolver.from_config(self.config)
        self.enricher = enricher
        self.snapshot_store = snapshot_store

    def run(self, created_utc: str | None = None) -> PipelineResult:
        all_records: list[CelestialRecord] = []
        errors: dict[str, str] = {}
        counts: dict[str, int] = {}
        sources_meta: dict[str, dict[str, Any]] = {}

        for job in self.jobs:
            try:
                recs = job.fetch()
            except (DataProviderUnavailable, RateLimitExceeded) as exc:
                if job.required:
                    raise
                errors[job.name] = f"{type(exc).__name__}: {exc}"
                continue
            all_records.extend(recs)
            counts[job.name] = len(recs)
            sources_meta[job.name] = {"dataset_version": job.dataset_version, "n_records": len(recs)}

        merged, report = self.resolver.resolve(all_records)
        if self.enricher is not None:
            merged = self.enricher.enrich(merged)

        manifest = None
        if self.snapshot_store is not None:
            manifest = self.snapshot_store.save(
                merged,
                pipeline_version=self.config.pipeline_version,
                unit_system=self.config.unit_system,
                sources=sources_meta,
                created_utc=created_utc,
            )
        return PipelineResult(merged, report, manifest, errors, counts)

    def build_risk_provider(self, result: PipelineResult, **kwargs: Any) -> AstronomicalRiskProvider:
        return AstronomicalRiskProvider(result.records, self.config, **kwargs)


def local_catalog_jobs(config: DataLayerConfig | None = None, cache=None) -> list[SourceJob]:
    from .connectors.local_csv import CATALOG_OBJECT_TYPES, LocalCatalogConnector

    jobs = []
    for name in sorted(CATALOG_OBJECT_TYPES):
        conn = LocalCatalogConnector(name, config, cache=cache)
        jobs.append(SourceJob(name=f"local_csv:{name}", fetch=conn.load, dataset_version=conn.dataset_version))
    return jobs


def default_local_risk_provider(config: DataLayerConfig | None = None) -> AstronomicalRiskProvider:
    """Proveedor de riesgo sin red basado en los catalogos locales CSV."""
    cfg = config or DataLayerConfig.default_config()
    result = AstronomicalDataPipeline(local_catalog_jobs(cfg), cfg).run()
    return AstronomicalRiskProvider(result.records, cfg)
