# Fase 0 - Resumen Ejecutivo

## Objetivo

Fijar las reglas cientificas, tecnicas y de interoperabilidad que bloquean el inicio de implementacion de las fases 1 a 6.

## Alcance de Fase 0

1. Especificacion cientifica base:
- Convenciones fisicas.
- Unidades y constantes.
- Marcos de referencia.
- Nomenclatura y semantica de variables.

2. Supuestos y restricciones:
- Hipotesis operativas para escenarios subluminales y superluminales.
- Restricciones fisicas duras y blandas.
- Criterios de parada por seguridad.

3. Contratos de interfaces:
- Contrato Nucleo Metrico -> Simulador Relativista.
- Contrato Simulador Relativista -> Optimizador PSO.
- Contrato Capa de Datos -> Evaluacion de Riesgos.

## Entregables implementados

1. Documento de convenciones cientificas: [01_CONVENCIONES_CIENTIFICAS.md](01_CONVENCIONES_CIENTIFICAS.md)
2. Documento de supuestos y restricciones: [02_SUPUESTOS_Y_RESTRICCIONES.md](02_SUPUESTOS_Y_RESTRICCIONES.md)
3. Documento de contratos de interfaces: [03_CONTRATOS_DE_INTERFACES.md](03_CONTRATOS_DE_INTERFACES.md)
4. Criterios de aceptacion y checklist: [04_CRITERIOS_DE_ACEPTACION_FASE0.md](04_CRITERIOS_DE_ACEPTACION_FASE0.md)
5. Configuracion cientifica versionable: [../../configs/phase0/scientific_conventions.yaml](../../configs/phase0/scientific_conventions.yaml)
6. Contratos de codigo en Python: [../../src/proxima/contracts](../../src/proxima/contracts)

## Politica de gobernanza

1. Ningun desarrollo de Fase 1+ puede considerarse valido si rompe estos contratos.
2. Todo cambio en convenciones cientificas exige incremento de version de especificacion.
3. Toda corrida de simulacion debe registrar la version de Fase 0 aplicada.

## Versionado

- Version de fase0: 1.0.0
- Estado: aprobado para iniciar Fase 1
- Fecha de emision: 2026-11-08
