"""Backend de condiciones de energia (Hawking-Ellis + opcional Warpax)."""

from .hawking_ellis import classify_energy_conditions, energy_margins
from .observer_search import search_observer_violation
from .report import (
    AggregateEnergyReport,
    EnergyReport,
    aggregate_point_reports,
    aggregate_segment_reports,
)
from .warpax_backend import warpax_energy_conditions

__all__ = [
    "classify_energy_conditions",
    "energy_margins",
    "search_observer_violation",
    "EnergyReport",
    "AggregateEnergyReport",
    "aggregate_point_reports",
    "aggregate_segment_reports",
    "warpax_energy_conditions",
]