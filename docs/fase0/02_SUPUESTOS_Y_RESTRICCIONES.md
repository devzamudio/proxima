# Fase 0 - Supuestos y Restricciones

## 1. Supuestos cientificos

1. El motor modela escenarios relativistas efectivos bajo Relatividad General clasica.
2. La parametrizacion de warp se inspira en la clase Bobrick-Martire en formulacion ADM.
3. La dinamica se evalua por geodesicas y observables agregados de trayectoria.
4. Los chequeos WEC/NEC/DEC/SEC se consideran parte del criterio de validez fisica.

## 2. Supuestos operativos

1. El sistema funciona en modo de investigacion numerica, no de navegacion en tiempo real.
2. Se prioriza reproducibilidad sobre latencia.
3. Se admite ejecucion batch para optimizacion PSO.
4. Se usa cache para reducir carga en servicios externos.

## 3. Escenarios de mision

1. Escenario subluminal:
- Es el baseline inicial.
- Debe pasar la suite minima de estabilidad y energia.

2. Escenario superluminal:
- Se habilita de forma controlada.
- Requiere trazabilidad extra de violaciones y costo energetico.

## 4. Restricciones fisicas duras

1. Integracion invalida si hay divergencia numerica no recuperable.
2. Simulacion invalida si viola limites de seguridad de marea definidos por experimento.
3. Configuracion invalida si parametros warp salen del dominio permitido.

## 5. Restricciones fisicas blandas (penalizables)

1. Violaciones parciales de condiciones de energia segun politica experimental.
2. Exceso de flujo radiante integrado sobre umbral recomendado.
3. Exceso de columna de H/extincion en rutas alternativas.

## 6. Restricciones de datos y consulta

1. Evitar microconsultas repetitivas en loop a servicios remotos.
2. Preferir consultas por lote y uso de cache.
3. Toda fuente debe versionarse por fecha de extraccion y esquema.

## 7. Criterios de parada de simulacion

1. Error numerico fuera de tolerancia maxima.
2. Condicion de no-fisicidad terminal.
3. Limite computacional alcanzado (tiempo o memoria) segun configuracion.

## 8. Politica de incertidumbre

1. Los resultados se reportan con metadatos de calidad de datos.
2. Cuando aplique, se habilitan corridas ensemble para sensibilidad.
3. Un resultado sin metadata de calidad no puede promoverse a baseline.
