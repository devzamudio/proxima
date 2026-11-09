"""Homologacion de unidades a SI y transformaciones explicitas de marcos a ICRS."""

from __future__ import annotations

import math

from proxima.contracts.exceptions import DataSchemaMismatch
from proxima.contracts.types import Frame

from .schema import CelestialRecord, FrameTransformationRecord, validate_record_schema

# Constantes de conversion explicitas y congeladas a SI
PARSEC_TO_M: float = 3.0856775814913673e16
AU_TO_M: float = 1.495978707e11
SOLAR_LUMINOSITY_W: float = 3.828e26
SOLAR_MASS_KG: float = 1.98847e30
SOLAR_RADIUS_M: float = 6.957e8
JULIAN_YEAR_S: float = 365.25 * 86400.0
GYR_TO_S: float = 1.0e9 * JULIAN_YEAR_S
# Relacion estandar Bohlin et al. (1978): N_H / E(B-V) = 5.8e21 cm^-2 mag^-1 = 5.8e25 m^-2 mag^-1
EBV_TO_NH_M2: float = 5.8e25

# Matriz de rotacion IAU 1958 / Hipparcos de Galactico (l, b) -> ICRS (x, y, z)
# Filas de A_G2ICRS = transpuesta de A_ICRS2G
_R_GAL_TO_ICRS: tuple[tuple[float, float, float], ...] = (
    (-0.0548755604162154, 0.4941094278755837, -0.8676661490190047),
    (-0.8734370902348850, -0.4448296299600112, -0.1980763734312015),
    (-0.4838350155487132, 0.7469822444972189, 0.4559837761750669),
)


def pc_to_m(distance_pc: float) -> float:
    return float(distance_pc) * PARSEC_TO_M


def parallax_mas_to_m(parallax_mas: float) -> float:
    if parallax_mas <= 0.0 or not math.isfinite(parallax_mas):
        raise DataSchemaMismatch(f"Paralaje invalida (mas): {parallax_mas}")
    distance_pc = 1000.0 / float(parallax_mas)
    return pc_to_m(distance_pc)


def lsun_to_w(luminosity_lsun: float) -> float:
    return float(luminosity_lsun) * SOLAR_LUMINOSITY_W


def msun_to_kg(mass_msun: float) -> float:
    return float(mass_msun) * SOLAR_MASS_KG


def rsun_to_m(radius_rsun: float) -> float:
    return float(radius_rsun) * SOLAR_RADIUS_M


def gyr_to_s(age_gyr: float) -> float:
    return float(age_gyr) * GYR_TO_S


def ebv_to_h_column_m2(ebv_mag: float) -> float:
    return max(0.0, float(ebv_mag)) * EBV_TO_NH_M2


def spherical_icrs_to_xyz_m(
    ra_deg: float, dec_deg: float, distance_m: float
) -> tuple[float, float, float]:
    """Convierte (ra_deg, dec_deg, distance_m) en ICRS a cartesiano (x, y, z) en metros."""
    ra_rad = math.radians(float(ra_deg))
    dec_rad = math.radians(float(dec_deg))
    r = float(distance_m)
    cos_dec = math.cos(dec_rad)
    x = r * cos_dec * math.cos(ra_rad)
    y = r * cos_dec * math.sin(ra_rad)
    z = r * math.sin(dec_rad)
    return (x, y, z)


def xyz_m_to_spherical_icrs(
    xyz_m: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Convierte cartesiano ICRS (x, y, z) en metros a (ra_deg, dec_deg, distance_m)."""
    x, y, z = float(xyz_m[0]), float(xyz_m[1]), float(xyz_m[2])
    r = math.sqrt(x * x + y * y + z * z)
    if r <= 0.0:
        return (0.0, 0.0, 0.0)
    ra_deg = math.degrees(math.atan2(y, x)) % 360.0
    dec_deg = math.degrees(math.asin(max(-1.0, min(1.0, z / r))))
    return (ra_deg, dec_deg, r)


def galactic_to_icrs(
    l_deg: float, b_deg: float, epoch: str = "J2000.0"
) -> tuple[float, float, FrameTransformationRecord]:
    """Transforma coordenadas galacticas (l, b) a ICRS (ra_deg, dec_deg) con trazabilidad."""
    l_rad = math.radians(float(l_deg))
    b_rad = math.radians(float(b_deg))
    ug = (
        math.cos(b_rad) * math.cos(l_rad),
        math.cos(b_rad) * math.sin(l_rad),
        math.sin(b_rad),
    )
    ux = sum(_R_GAL_TO_ICRS[0][j] * ug[j] for j in range(3))
    uy = sum(_R_GAL_TO_ICRS[1][j] * ug[j] for j in range(3))
    uz = sum(_R_GAL_TO_ICRS[2][j] * ug[j] for j in range(3))

    ra_deg = math.degrees(math.atan2(uy, ux)) % 360.0
    dec_deg = math.degrees(math.asin(max(-1.0, min(1.0, uz))))
    transform = FrameTransformationRecord(
        marco_origen="GALACTIC",
        marco_destino="ICRS",
        epoca=epoch,
        precision_esperada=1e-7,
    )
    return ra_deg, dec_deg, transform


def build_normalized_record(
    *,
    canonical_id: str,
    source_name: str,
    source_version: str,
    object_type: str,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    l_deg: float | None = None,
    b_deg: float | None = None,
    distance_m: float | None = None,
    distance_pc: float | None = None,
    parallax_mas: float | None = None,
    luminosity_w: float | None = None,
    luminosity_lsun: float | None = None,
    mass_kg: float | None = None,
    mass_msun: float | None = None,
    radius_m: float | None = None,
    radius_rsun: float | None = None,
    radius_pc: float | None = None,
    ebv_mag: float = 0.0,
    h_column_m2: float | None = None,
    metallicity_feh: float | None = None,
    age_s: float | None = None,
    age_gyr: float | None = None,
    aliases: tuple[str, ...] | list[str] = (),
    source_frame: str = "ICRS",
    epoch: str = "J2000.0",
    confidence_score: float = 1.0,
) -> CelestialRecord:
    """Construye y valida un CelestialRecord homologando todas las unidades a SI y marco ICRS."""
    if source_frame.upper() == "GALACTIC" or (ra_deg is None and l_deg is not None and b_deg is not None):
        if l_deg is None or b_deg is None:
            raise DataSchemaMismatch("Se requieren l_deg y b_deg para marco GALACTIC")
        ra_val, dec_val, ft = galactic_to_icrs(l_deg, b_deg, epoch=epoch)
    else:
        if ra_deg is None or dec_deg is None:
            raise DataSchemaMismatch("Se requieren ra_deg y dec_deg")
        ra_val = float(ra_deg)
        dec_val = float(dec_deg)
        ft = FrameTransformationRecord(
            marco_origen=source_frame.upper(),
            marco_destino="ICRS",
            epoca=epoch,
            precision_esperada=1e-9 if source_frame.upper() == "ICRS" else 1e-7,
        )

    if distance_m is not None:
        dist_val = float(distance_m)
    elif distance_pc is not None:
        dist_val = pc_to_m(distance_pc)
    elif parallax_mas is not None:
        dist_val = parallax_mas_to_m(parallax_mas)
    else:
        raise DataSchemaMismatch("Se requiere distance_m, distance_pc o parallax_mas")

    if luminosity_w is not None:
        lum_val = float(luminosity_w)
    elif luminosity_lsun is not None:
        lum_val = lsun_to_w(luminosity_lsun)
    else:
        lum_val = SOLAR_LUMINOSITY_W

    if mass_kg is not None:
        mass_val = float(mass_kg)
    elif mass_msun is not None:
        mass_val = msun_to_kg(mass_msun)
    else:
        mass_val = SOLAR_MASS_KG

    if radius_m is not None:
        rad_val = float(radius_m)
    elif radius_pc is not None:
        rad_val = pc_to_m(radius_pc)
    elif radius_rsun is not None:
        rad_val = rsun_to_m(radius_rsun)
    else:
        rad_val = SOLAR_RADIUS_M

    ebv_val = float(ebv_mag)
    nh_val = float(h_column_m2) if h_column_m2 is not None else ebv_to_h_column_m2(ebv_val)

    age_val: float | None = None
    if age_s is not None:
        age_val = float(age_s)
    elif age_gyr is not None:
        age_val = gyr_to_s(age_gyr)

    xyz = spherical_icrs_to_xyz_m(ra_val, dec_val, dist_val)

    record = CelestialRecord(
        canonical_id=str(canonical_id),
        source_name=str(source_name),
        source_version=str(source_version),
        object_type=str(object_type),
        ra_deg=ra_val,
        dec_deg=dec_val,
        distance_m=dist_val,
        xyz_icrs_m=xyz,
        luminosity_w=lum_val,
        mass_kg=mass_val,
        radius_m=rad_val,
        ebv_mag=ebv_val,
        h_column_m2=nh_val,
        metallicity_feh=float(metallicity_feh) if metallicity_feh is not None else None,
        age_s=age_val,
        aliases=tuple(str(a).strip() for a in aliases if str(a).strip()),
        frame=Frame.ICRS,
        frame_transform=ft,
        confidence_score=float(confidence_score),
    )
    validate_record_schema(record)
    return record
