# Plantilla de Pruebas de Contrato

Estas pruebas validan que las implementaciones reales respeten los contratos definidos en Fase 0.

## Archivos principales

- [conftest.py](conftest.py): fixtures base y puntos de extension.
- [test_contract_templates.py](test_contract_templates.py): pruebas plantilla para las 4 interfaces.
- [../../src/proxima/contracts/testing.py](../../src/proxima/contracts/testing.py): validadores reutilizables.

## Como activar los tests

1. Implementa las fixtures en [conftest.py](conftest.py):
- metric_core_impl
- simulator_impl
- risk_provider_impl
- fitness_evaluator_impl

2. Ejecuta:

```bash
pytest -m contract
```

## Comportamiento por defecto

Si no hay implementaciones conectadas, los tests quedan en `skip` para permitir bootstrap incremental del proyecto.
