"""Configuracion de la Capa de Datos Astronomicos y Cache (Fase 3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

def _resolve_default_paths() -> tuple[Path, Path]:
    pkg_root = Path(__file__).resolve().parent.parent / "configs" / "phase3"
    if (pkg_root / "data_defaults.yaml").exists():
        return pkg_root / "data_defaults.yaml", pkg_root / "catalogs"
    repo_root = Path(__file__).resolve().parents[3] / "configs" / "phase3"
    return repo_root / "data_defaults.yaml", repo_root / "catalogs"


_DEFAULT_CONFIG, _DEFAULT_CATALOGS_DIR = _resolve_default_paths()


@dataclass(frozen=True)
class DataLayerConfig:
    version: str = "1.0.0"
    based_on_phase0: str = "1.0.0"
    pipeline_version: str = "1.0.0"
    unit_system: str = "SI"
    default_frame: str = "ICRS"

    # Cache
    memory_max_entries: int = 256
    ttl_seconds: dict[str, float] = field(
        default_factory=lambda: {
            "exoplanet_tap": 86400.0,
            "gaia_dr3": 604800.0,
            "simbad": 604800.0,
            "bayestar19": 2592000.0,
            "mist": 2592000.0,
            "local_csv": 31536000.0,
        }
    )

    # Gobernanza y rate-limit
    max_retries: int = 3
    backoff_seconds: float = 0.05
    max_batch_size: int = 500
    simbad_max_sequential_single_queries: int = 3
    rate_limit_calls_per_window: int = 50
    rate_limit_window_seconds: float = 60.0

    # Cross-match
    angular_tolerance_arcsec: float = 5.0
    distance_rel_tolerance: float = 0.15

    # Escalas SI de normalizacion de riesgo
    radiative_flux_scale_j_m2: float = 1.0e6
    h_column_scale_m2: float = 1.0e25
    mass_density_scale_kg_m3: float = 1.0e-18
    tidal_stress_scale_s2: float = 1.0e-10
    risk_weights: dict[str, float] = field(
        default_factory=lambda: {
            "radiative": 0.30,
            "extinction": 0.25,
            "mass_density": 0.25,
            "tidal": 0.20,
        }
    )

    # Versiones de dataset por fuente (parte de la clave de cache)
    dataset_versions: dict[str, str] = field(
        default_factory=lambda: {
            "exoplanet_tap": "pscomppars",
            "gaia_dr3": "gaiadr3",
            "simbad": "simbad-4",
            "bayestar19": "bayestar2019",
            "mist": "MIST-v1.2",
            "local_csv": "auto-checksum",
        }
    )

    # Entorno fisico para el proveedor de riesgo (SI)
    ism_n_h_m3: float = 1.0e5            # densidad numerica de H de fondo (0.1 cm^-3)
    transit_speed_m_s: float = 0.1 * 299792458.0
    gas_object_types: tuple[str, ...] = ("HII_region", "supernova_remnant")

    catalogs_dir: Path = _DEFAULT_CATALOGS_DIR

    @classmethod
    def load(cls, path: str | Path | None = None) -> "DataLayerConfig":
        raw = yaml.safe_load(Path(path or _DEFAULT_CONFIG).read_text(encoding="utf-8"))
        p3 = raw["phase3"]
        cache = raw["cache"]
        gov = raw["governance"]
        cm = raw["cross_match"]
        rs = raw["risk_scales_si"]
        env = raw.get("environment", {})
        defaults = cls()
        return cls(
            dataset_versions={
                **defaults.dataset_versions,
                **{k: str(v) for k, v in raw.get("dataset_versions", {}).items()},
            },
            ism_n_h_m3=float(env.get("ism_n_h_m3", defaults.ism_n_h_m3)),
            transit_speed_m_s=float(env.get("transit_speed_m_s", defaults.transit_speed_m_s)),
            gas_object_types=tuple(env.get("gas_object_types", defaults.gas_object_types)),
            version=str(p3["version"]),
            based_on_phase0=str(p3["based_on_phase0"]),
            pipeline_version=str(p3["pipeline_version"]),
            unit_system=str(p3["unit_system"]),
            default_frame=str(p3["default_frame"]),
            memory_max_entries=int(cache["memory_max_entries"]),
            ttl_seconds={k: float(v) for k, v in cache["ttl_seconds"].items()},
            max_retries=int(gov["max_retries"]),
            backoff_seconds=float(gov["backoff_seconds"]),
            max_batch_size=int(gov["max_batch_size"]),
            simbad_max_sequential_single_queries=int(
                gov.get("simbad_max_Sequential_single_queries", 3)
            ),
            rate_limit_calls_per_window=int(gov["rate_limit_calls_per_window"]),
            rate_limit_window_seconds=float(gov["rate_limit_window_seconds"]),
            angular_tolerance_arcsec=float(cm["angular_tolerance_arcsec"]),
            distance_rel_tolerance=float(cm["distance_rel_tolerance"]),
            radiative_flux_scale_j_m2=float(rs["radiative_flux_j_m2"]),
            h_column_scale_m2=float(rs["h_column_m2"]),
            mass_density_scale_kg_m3=float(rs["mass_density_kg_m3"]),
            tidal_stress_scale_s2=float(rs["tidal_stress_s2"]),
            risk_weights={k: float(v) for k, v in rs["weights"].items()},
            catalogs_dir=_DEFAULT_CATALOGS_DIR,
        )

    @classmethod
    def default_config(cls) -> "DataLayerConfig":
        return cls()
