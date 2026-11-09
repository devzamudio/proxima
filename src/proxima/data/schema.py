"""Esquema normalizado de registros astronomicos en unidades SI y marco ICRS."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from proxima.contracts.exceptions import DataSchemaMismatch
from proxima.contracts.types import Frame


@dataclass(frozen=True)
class FrameTransformationRecord:
    """Metadatos obligatorios de transformacion de marco (Fase 0, Seccion 3)."""

    marco_origen: str
    marco_destino: str
    epoca: str
    precision_esperada: float


@dataclass(frozen=True)
class CelestialRecord:
    """Esquema comun normalizado en SI e ICRS para cualquier fuente astronomica."""

    canonical_id: str
    source_name: str
    source_version: str
    object_type: str
    ra_deg: float
    dec_deg: float
    distance_m: float
    xyz_icrs_m: tuple[float, float, float]
    luminosity_w: float
    mass_kg: float
    radius_m: float
    ebv_mag: float = 0.0
    h_column_m2: float = 0.0
    metallicity_feh: float | None = None
    age_s: float | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    frame: Frame = Frame.ICRS
    frame_transform: FrameTransformationRecord = field(
        default_factory=lambda: FrameTransformationRecord(
            marco_origen="ICRS",
            marco_destino="ICRS",
            epoca="J2000.0",
            precision_esperada=1e-9,
        )
    )
    confidence_score: float = 1.0
    units_declared: dict[str, str] = field(
        default_factory=lambda: {
            "distance_m": "m",
            "xyz_icrs_m": "m",
            "luminosity_w": "W",
            "mass_kg": "kg",
            "radius_m": "m",
            "h_column_m2": "m^-2",
            "age_s": "s",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["frame"] = self.frame.value
        d["xyz_icrs_m"] = list(self.xyz_icrs_m)
        d["aliases"] = list(self.aliases)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CelestialRecord":
        ft = data.get("frame_transform", {})
        return cls(
            canonical_id=str(data["canonical_id"]),
            source_name=str(data["source_name"]),
            source_version=str(data["source_version"]),
            object_type=str(data["object_type"]),
            ra_deg=float(data["ra_deg"]),
            dec_deg=float(data["dec_deg"]),
            distance_m=float(data["distance_m"]),
            xyz_icrs_m=(
                float(data["xyz_icrs_m"][0]),
                float(data["xyz_icrs_m"][1]),
                float(data["xyz_icrs_m"][2]),
            ),
            luminosity_w=float(data["luminosity_w"]),
            mass_kg=float(data["mass_kg"]),
            radius_m=float(data["radius_m"]),
            ebv_mag=float(data.get("ebv_mag", 0.0)),
            h_column_m2=float(data.get("h_column_m2", 0.0)),
            metallicity_feh=(
                float(data["metallicity_feh"])
                if data.get("metallicity_feh") is not None
                else None
            ),
            age_s=float(data["age_s"]) if data.get("age_s") is not None else None,
            aliases=tuple(str(a) for a in data.get("aliases", [])),
            frame=Frame(data.get("frame", "ICRS")),
            frame_transform=FrameTransformationRecord(
                marco_origen=str(ft.get("marco_origen", "ICRS")),
                marco_destino=str(ft.get("marco_destino", "ICRS")),
                epoca=str(ft.get("epoca", "J2000.0")),
                precision_esperada=float(ft.get("precision_esperada", 1e-9)),
            ),
            confidence_score=float(data.get("confidence_score", 1.0)),
            units_declared=dict(data.get("units_declared", {})),
        )


def validate_record_schema(record: CelestialRecord) -> None:
    """Valida que un CelestialRecord cumpla el esquema fisico comun y unidades SI."""
    if not isinstance(record, CelestialRecord):
        raise DataSchemaMismatch(f"Se esperaba CelestialRecord, se recibio {type(record)}")

    if not record.canonical_id or not record.source_name or not record.source_version:
        raise DataSchemaMismatch("canonical_id, source_name y source_version son obligatorios")

    for name, val in [
        ("ra_deg", record.ra_deg),
        ("dec_deg", record.dec_deg),
        ("distance_m", record.distance_m),
        ("luminosity_w", record.luminosity_w),
        ("mass_kg", record.mass_kg),
        ("radius_m", record.radius_m),
        ("ebv_mag", record.ebv_mag),
        ("h_column_m2", record.h_column_m2),
        ("confidence_score", record.confidence_score),
    ]:
        if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            raise DataSchemaMismatch(f"Campo {name} no es finito: {val}")

    if not (0.0 <= record.ra_deg <= 360.0):
        raise DataSchemaMismatch(f"ra_deg fuera de [0, 360]: {record.ra_deg}")
    if not (-90.0 <= record.dec_deg <= 90.0):
        raise DataSchemaMismatch(f"dec_deg fuera de [-90, 90]: {record.dec_deg}")
    if record.distance_m <= 0.0:
        raise DataSchemaMismatch(f"distance_m debe ser > 0: {record.distance_m}")
    if record.luminosity_w < 0.0 or record.mass_kg < 0.0 or record.radius_m <= 0.0:
        raise DataSchemaMismatch("luminosity_w, mass_kg y radius_m deben ser no negativos")
    if not (0.0 <= record.confidence_score <= 1.0):
        raise DataSchemaMismatch(f"confidence_score fuera de [0, 1]: {record.confidence_score}")

    if len(record.xyz_icrs_m) != 3 or not all(math.isfinite(c) for c in record.xyz_icrs_m):
        raise DataSchemaMismatch(f"xyz_icrs_m invalido: {record.xyz_icrs_m}")

    if record.frame != Frame.ICRS:
        raise DataSchemaMismatch(f"Marco normalizado debe ser ICRS, recibido {record.frame}")

    ft = record.frame_transform
    if not ft.marco_origen or not ft.marco_destino or not ft.epoca:
        raise DataSchemaMismatch("Metadatos de transformacion de marco incompletos")
