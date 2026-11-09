"""Conectores por servicio de la capa de datos (Fase 3)."""

from __future__ import annotations

from .base import SourceConnector, clean_rows, table_to_rows
from .bayestar import BAYESTAR19_TO_EBV, BayestarExtinctionProvider, ConstantExtinctionProvider
from .exoplanet_tap import ExoplanetArchiveConnector
from .gaia import GaiaDR3Connector
from .local_csv import CATALOG_OBJECT_TYPES, LocalCatalogConnector, load_all_local_catalogs
from .mist import MistStellarModelProvider, main_sequence_backend
from .simbad import SimbadConnector

__all__ = [
    "SourceConnector",
    "clean_rows",
    "table_to_rows",
    "BAYESTAR19_TO_EBV",
    "BayestarExtinctionProvider",
    "ConstantExtinctionProvider",
    "ExoplanetArchiveConnector",
    "GaiaDR3Connector",
    "CATALOG_OBJECT_TYPES",
    "LocalCatalogConnector",
    "load_all_local_catalogs",
    "MistStellarModelProvider",
    "main_sequence_backend",
    "SimbadConnector",
]
