# Fase 0 - Contratos de Interfaces

## 1. Objetivo de contratos

Definir entradas, salidas, errores y garantias minimas de interoperabilidad entre capas.

## 2. Contrato: Nucleo Metrico -> Simulador Relativista

## Entrada minima

1. `WarpParameterVector`
2. `MetricStatePoint`
3. `ScientificConventions`

## Salida minima

1. `MetricEvaluation`
2. `EnergyConditionReport`

## Garantias

1. Salidas en unidades SI o explicitamente anotadas.
2. Valores finitos o error tipado.
3. Reporte de dominio invalido cuando corresponda.

## Errores tipados

1. `DomainError`
2. `NumericalStabilityError`
3. `ContractValidationError`

## 3. Contrato: Simulador Relativista -> Optimizador PSO

## Entrada minima

1. `SimulationRequest`
2. `WarpParameterVector`

## Salida minima

1. `SimulationResult`
2. `FitnessBreakdown`

## Garantias

1. Siempre retorna estado terminal explicito: success/failure.
2. Si falla, informa `failure_reason` y metadata de ultimo estado valido.
3. Entrega observables obligatorios cuando `success=true`.

## Errores tipados

1. `IntegrationError`
2. `PhysicalConstraintViolation`
3. `ContractValidationError`

## 4. Contrato: Capa de Datos -> Evaluacion de Riesgos

## Entrada minima

1. `RouteSegment`
2. `SkyCoordinates`
3. `DataQueryContext`

## Salida minima

1. `RiskBreakdown`
2. `DataQualityMetadata`

## Garantias

1. Respuesta trazable con fuente y version de dataset.
2. Normalizacion de unidades previa a devolver resultados.
3. Indicador de confianza por submetrica de riesgo.

## Errores tipados

1. `DataProviderUnavailable`
2. `DataSchemaMismatch`
3. `RateLimitExceeded`

## 5. Reglas de versionado de contratos

1. Cambio mayor (MAJOR): rompe compatibilidad.
2. Cambio menor (MINOR): agrega campos opcionales.
3. Cambio parche (PATCH): corrige definiciones sin romper firma.

Formato de version recomendado: `X.Y.Z`

## 6. Requisitos de validacion automatica

1. Validadores de contrato deben ejecutarse en CI.
2. Toda implementacion concreta debe pasar pruebas de conformidad.
3. Ningun modulo puede consumir otro modulo sin pasar por tipos de contrato.
