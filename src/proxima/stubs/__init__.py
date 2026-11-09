"""
Stubs mínimos de cada contrato de Fase 0.

Estos módulos proveen implementaciones de referencia — numéricamente
triviales pero estructuralmente correctas — para que la suite de contratos
pase en verde desde el primer día y sirva como línea base de regresión.

No contienen física real: su único propósito es satisfacer la forma de los
contratos definidos en ``proxima.contracts``.
"""

from .metric_core import StubMetricCore
from .simulator import StubRelativisticSimulator
from .risk_provider import StubRiskDataProvider
from .fitness_evaluator import StubFitnessEvaluator

__all__ = [
    "StubMetricCore",
    "StubRelativisticSimulator",
    "StubRiskDataProvider",
    "StubFitnessEvaluator",
]
