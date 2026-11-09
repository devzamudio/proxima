"""Capa de Datos Astronomicos y Cache (Fase 3)."""

from __future__ import annotations

from .cache import CacheKey, MultiLevelCache
from .config import DataLayerConfig
from .crossmatch import CrossMatchReport, IdentityResolver, angular_separation_arcsec, normalize_alias
from .enrichment import ContextEnricher
from .governance import RateLimiter, SimbadLoopGuard, chunked_batches, execute_with_retry
from .normalizer import build_normalized_record
from .pipeline import (
    AstronomicalDataPipeline,
    PipelineResult,
    SourceJob,
    default_local_risk_provider,
    local_catalog_jobs,
)
from .risk_provider import AstronomicalRiskProvider
from .schema import CelestialRecord, FrameTransformationRecord, validate_record_schema
from .snapshot import SnapshotManifest, SnapshotStore, records_checksum

__all__ = [
    "CacheKey",
    "MultiLevelCache",
    "DataLayerConfig",
    "CrossMatchReport",
    "IdentityResolver",
    "angular_separation_arcsec",
    "normalize_alias",
    "ContextEnricher",
    "RateLimiter",
    "SimbadLoopGuard",
    "chunked_batches",
    "execute_with_retry",
    "build_normalized_record",
    "AstronomicalDataPipeline",
    "PipelineResult",
    "SourceJob",
    "default_local_risk_provider",
    "local_catalog_jobs",
    "AstronomicalRiskProvider",
    "CelestialRecord",
    "FrameTransformationRecord",
    "validate_record_schema",
    "SnapshotManifest",
    "SnapshotStore",
    "records_checksum",
]
