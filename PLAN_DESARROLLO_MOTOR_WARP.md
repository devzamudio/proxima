# Plan Maestro de Desarrollo - Motor Relativista BM + PSO + Datos Astronomicos

## 1. Vision del Sistema

Este proyecto implementa un motor cientifico en Python para planificar, optimizar y simular viajes interestelares a velocidades relativistas, basado en:

- Relatividad General aplicada a metrica de warp.
- Clase formal de soluciones geometricas de Bobrick-Martire.
- Simulacion de geodesicas y observables fisicos de mision.
- Optimizacion por enjambre de particulas (PSO) vectorizada.
- Integracion de datos astronomicos de multiples fuentes con cache robusta.

El sistema se organiza en tres capas principales:

1. Nucleo metrico.
2. Simulador relativista.
3. Optimizador PSO.

Y dos capas transversales:

1. Capa de datos astronomicos.
2. Capa de cache, trazabilidad y reproducibilidad.

## 2. Principios de Arquitectura

## 2.1 Desacoplamiento por contratos

Cada capa expone interfaces claras y no accede a detalles internos de otra capa.

- El simulador consume un contrato del nucleo metrico, no su implementacion interna.
- El optimizador consume un contrato del simulador para evaluar fitness.
- La capa de datos expone proveedores normalizados, independientes de la fuente.

## 2.2 Reproducibilidad cientifica

Toda corrida debe poder repetirse con resultados comparables mediante:

- Versionado de configuracion.
- Semillas controladas.
- Snapshot de datos usados.
- Registro de versiones de catalogos y dependencias.

## 2.3 Seguridad numerica y fisica

Antes de optimizar rendimiento:

- Se valida consistencia matematica.
- Se controlan errores de integracion.
- Se verifican condiciones de energia y restricciones fisicas.

## 3. Fases de Desarrollo

## Fase 0 - Especificacion cientifica y de contratos (bloqueante)

Objetivo: fijar reglas del sistema para evitar deuda tecnica y fisica.

Entregables:

1. Documento de convenciones fisicas:
- Firma metrica.
- Sistema de unidades.
- Marcos de referencia.
- Notacion de parametros warp.

2. Documento de supuestos:
- Escenarios subluminales y superluminales.
- Limites operativos y de seguridad.
- Definicion de restricciones fisicas duras y blandas.

3. Contratos de interfaces:
- Contrato del nucleo metrico para el simulador.
- Contrato del simulador para el optimizador.
- Contrato de proveedores de datos para la evaluacion de riesgos.

Criterio de salida:

- Ninguna implementacion de capa comienza sin contratos aprobados.

---

## Fase 1 - Nucleo Metrico Bobrick-Martire en ADM

Objetivo: construir el fundamento geometrico y tensorial del motor.

## Modulos del nucleo

1. Parametrizacion BM:
- Representacion canonica del vector de parametros warp.
- Validacion de dominios permitidos.

2. Formulacion ADM:
- Lapse.
- Shift vector.
- Shift function.
- Terminos espaciales requeridos por la metrica.

3. Motor tensorial:
- Conexiones.
- Curvaturas e invariantes necesarios.
- Cantidades intermedias para chequeos energeticos.

4. Seguridad numerica:
- Regularizacion en regiones de alta curvatura.
- Guardas ante singularidades numericas.

## Verificacion de Fase 1

1. Pruebas de consistencia de signos y unidades.
2. Pruebas de estabilidad en escenarios de control.
3. Pruebas de sensibilidad sobre parametros BM.

Criterio de salida:

- El nucleo devuelve magnitudes geometricas correctas y estables en casos canonicos.

---

## Fase 2 - Simulador Relativista

Objetivo: integrar geodesicas y calcular observables de mision y seguridad.

## Modulos del simulador

1. Estado dinamico:
- Posicion 4D.
- Momento/velocidad.
- Tiempo propio y tiempo coordenado acumulados.

2. Integrador:
- Integracion de geodesica en tau vs t.
- Paso adaptativo con control de error.

3. Observables fisicos:
- Delta tau.
- Delta t.
- Energia total.
- Magnitudes de marea.

4. Guardas de simulacion:
- Criterios de corte temprano.
- Deteccion de violaciones fisicas.
- Clasificacion de fallas por causa.

## Modulo de condiciones de energia (Warpax)

1. Verificacion de WEC.
2. Verificacion de NEC.
3. Verificacion de DEC.
4. Verificacion de SEC.

Cada chequeo debe poder reportarse:

- Por estado puntual.
- Por tramo de trayectoria.
- Por resumen agregado de corrida.

## Modulo de evaluacion de riesgos

1. Flujo radiante integrado por tramo.
2. Extincion y columna de H por linea de vision.
3. Densidad de masa local por segmento de ruta.
4. Tidal stress por entorno de cumulos.

Salida del modulo:

- Puntaje de riesgo compuesto.
- Subpuntajes trazables para penalizacion en fitness.

## Verificacion de Fase 2

1. Pruebas de convergencia por refinamiento de paso.
2. Pruebas de conservacion/consistencia en casos de referencia.
3. Pruebas de monotonicidad para metricas de riesgo.

Criterio de salida:

- El simulador produce observables reproducibles y fisicamente consistentes.

---

## Fase 3 - Capa de Datos Astronomicos y Cache

Objetivo: construir un pipeline de datos confiable para mapear entorno celeste y soportar riesgos.

## Fuentes de datos previstas

1. NASA Exoplanet Archive (preferencia TAP/ADQL).
2. GAIA DR3 via astroquery.gaia.
3. SIMBAD via astroquery.simbad.
4. dustmaps con Bayestar19.
5. MIST via isochrones.
6. Catalogos locales CSV: Sharpless, MWSC, SNRs.

## Modulos de datos

1. Ingestion por fuente:
- Conectores separados por servicio.
- Manejo de errores y reintentos.

2. Normalizacion:
- Esquema comun de columnas fisicas.
- Homologacion de unidades y marcos.

3. Resolucion de identidades:
- Alias de objetos.
- Unificacion de identificadores cruzados.

4. Enriquecimiento contextual:
- Asociar objetos de interes con medio interestelar y entorno local.

## Estrategia de cache

1. Cache multinivel:
- Memoria para consultas calientes.
- Disco para resultados persistentes.

2. Clave de cache:
- Firma de consulta.
- Version de dataset.
- Version del pipeline.
- Sistema de unidades.

3. Politicas:
- TTL por tipo de fuente.
- Invalidacion por version/checksum.
- Proteccion contra datos obsoletos.

4. Gobernanza de cuota y rate-limit:
- Batch queries.
- Preferencia por asincronia en cargas grandes.
- Evitar loops de microconsultas (especialmente en SIMBAD).

## Verificacion de Fase 3

1. Pruebas de esquema por conector.
2. Pruebas de integridad de cross-match.
3. Pruebas de invalidacion de cache.
4. Pruebas de reproducibilidad de snapshots.

Criterio de salida:

- La capa de datos entrega informacion consistente, trazable y eficiente en costo de consulta.

---

## Fase 4 - Optimizador PSO Vectorizado

Objetivo: buscar configuraciones warp viables maximizando desempeno y minimizando riesgo.

## Modulos PSO

1. Representacion de particulas:
- Vector de parametros warp.
- Restricciones de dominio desde fase de inicializacion.

2. Evaluacion de fitness:
- Ejecuta simulacion completa por particula.
- Calcula objetivo principal + penalizaciones.

3. Vectorizacion:
- PSO por lotes con jax.vmap.
- Diseno para ejecucion determinista con semillas fijas.

4. Dinamica PSO:
- Actualizacion de velocidad y posicion.
- Topologia configurable (global/local).
- Cronograma de inercia y parametros cognitivos/sociales.

## Funcion objetivo recomendada

Objetivo compuesto:

1. Termino de mision:
- Tiempo coordenado.
- Tiempo propio.
- Eficiencia energetica.

2. Penalizaciones fisicas:
- Violaciones WEC/NEC/DEC/SEC.
- Sobretension de marea.
- Excesos de riesgo radiativo y de entorno.

3. Penalizaciones operativas:
- Trayectorias no robustas.
- Regiones de inestabilidad numerica.

## Verificacion de Fase 4

1. Benchmarks de convergencia en funciones sinteticas.
2. Sensibilidad a hiperparametros.
3. Estabilidad entre seeds.

Criterio de salida:

- El PSO converge de forma estable y trazable bajo criterios fisicos definidos.

---

## Fase 5 - Calibracion, Validacion Integral y Performance

Objetivo: consolidar validez cientifica y viabilidad computacional.

## Calibracion

1. Barrido de hiperparametros PSO.
2. Analisis de sensibilidad por parametro warp.
3. Evaluacion de robustez ante incertidumbre de datos.

## Validacion integral

1. Suite de regresion numerica end-to-end.
2. Casos canonicos de no-regresion.
3. Comparacion entre corridas con snapshots identicos.

## Performance

1. Profiling de cuellos de botella.
2. Ajuste de lotes y memoria.
3. Estrategia de paralelizacion escalable.

Criterio de salida:

- El sistema es reproducible, estable y con presupuesto computacional controlado.

---

## Fase 6 - Entrega Cientifico-Operativa

Objetivo: cerrar el sistema para uso continuo y evolucion futura.

Entregables:

1. Documentacion tecnica y cientifica consolidada.
2. Guia de ejecucion de experimentos.
3. Protocolo de release y trazabilidad.
4. Catalogo de riesgos conocidos y mitigaciones.

Criterio de salida:

- Sistema listo para evolucion incremental y trabajo experimental sostenido.

## 4. Riesgos Tecnicos Clave y Mitigaciones

1. Inestabilidad numerica de integracion:
- Mitigar con paso adaptativo, control de error y guardas.

2. Inconsistencias de unidades y marcos:
- Mitigar con normalizador unico y pruebas dimensionales.

3. Costo computacional alto:
- Mitigar con vectorizacion, cache y jerarquizacion de evaluaciones.

4. Datos obsoletos o cache corrupto:
- Mitigar con invalidacion por version y checksum.

5. Sobreuso de APIs externas:
- Mitigar con consultas por lote, asincronia y politicas de cache.

## 5. Recomendaciones de Features para Mejorar el Sistema

1. Optimizacion multiobjetivo tipo Pareto para explorar trade-offs mision-seguridad-energia.
2. Cuantificacion de incertidumbre con Monte Carlo/ensembles.
3. Explainability del fitness para auditar decisiones del optimizador.
4. Planificador jerarquico: prefiltrado rapido + simulacion de alta fidelidad para candidatos top.
5. Snapshots firmados con manifests para reproducibilidad fuerte.
6. Metricas de calidad por fuente de datos para ponderar confiabilidad en riesgo final.
7. Pipeline de ingestion incremental para actualizar catalogos sin recarga total.

## 6. Definicion de MVP Cientifico Recomendado

MVP sugerido (ordenado):

1. Nucleo BM/ADM minimo validado.
2. Simulador con delta tau/delta t/energia/mareas.
3. Chequeos WEC/NEC/DEC/SEC.
4. Riesgo minimo con radiacion y extincion.
5. PSO vectorizado con penalizacion base.
6. Capa de datos inicial: Exoplanet TAP + Gaia DR3 + cache de disco.

Con este MVP se habilita una primera campana de experimentacion fiable y extensible.

## 7. Implementacion de Fase 0 (bloqueante)

Artefactos implementados en repositorio:

1. Documento resumen de fase:
- [docs/fase0/00_RESUMEN_EJECUTIVO.md](docs/fase0/00_RESUMEN_EJECUTIVO.md)

2. Convenciones cientificas:
- [docs/fase0/01_CONVENCIONES_CIENTIFICAS.md](docs/fase0/01_CONVENCIONES_CIENTIFICAS.md)

3. Supuestos y restricciones:
- [docs/fase0/02_SUPUESTOS_Y_RESTRICCIONES.md](docs/fase0/02_SUPUESTOS_Y_RESTRICCIONES.md)

4. Contratos de interfaces:
- [docs/fase0/03_CONTRATOS_DE_INTERFACES.md](docs/fase0/03_CONTRATOS_DE_INTERFACES.md)

5. Criterios de aceptacion de la fase:
- [docs/fase0/04_CRITERIOS_DE_ACEPTACION_FASE0.md](docs/fase0/04_CRITERIOS_DE_ACEPTACION_FASE0.md)

6. Configuracion cientifica versionable:
- [configs/phase0/scientific_conventions.yaml](configs/phase0/scientific_conventions.yaml)

7. Contratos de codigo Python:
- [src/proxima/contracts/types.py](src/proxima/contracts/types.py)
- [src/proxima/contracts/interfaces.py](src/proxima/contracts/interfaces.py)
- [src/proxima/contracts/exceptions.py](src/proxima/contracts/exceptions.py)
- [src/proxima/contracts/__init__.py](src/proxima/contracts/__init__.py)

Estado de salida de Fase 0:

- Aprobada para iniciar Fase 1.
- Version de especificacion: 1.0.0.
- Fecha de aprobacion: 2026-11-08.

## 8. Plantilla de Pruebas de Contrato (implementada)

Objetivo:

- Asegurar que implementaciones reales de Fase 1+ respeten contratos definidos en Fase 0.

Artefactos:

1. Configuracion de pytest:
- [pytest.ini](pytest.ini)

2. Bootstrap de import para `src`:
- [tests/conftest.py](tests/conftest.py)

3. Fixtures plantilla y muestras de entrada:
- [tests/contracts/conftest.py](tests/contracts/conftest.py)

4. Suite plantilla de conformidad de contratos:
- [tests/contracts/test_contract_templates.py](tests/contracts/test_contract_templates.py)

5. Validadores reutilizables de contrato:
- [src/proxima/contracts/testing.py](src/proxima/contracts/testing.py)

6. Guia de activacion:
- [tests/contracts/README.md](tests/contracts/README.md)

Notas operativas:

- Por defecto, la suite corre en modo `skip` hasta conectar implementaciones concretas.
- Comando objetivo: `pytest -m contract`.

## 9. Implementacion de Fase 1 y Fase 2 (completadas)

### Artefactos de Fase 1 (Nucleo Metrico BM en ADM):
- [configs/phase1/core_defaults.yaml](configs/phase1/core_defaults.yaml)
- [src/proxima/core/params.py](src/proxima/core/params.py)
- [src/proxima/core/kernels/shape.py](src/proxima/core/kernels/shape.py)
- [src/proxima/core/kernels/safety.py](src/proxima/core/kernels/safety.py)
- [src/proxima/core/kernels/adm.py](src/proxima/core/kernels/adm.py)
- [src/proxima/core/kernels/geometry.py](src/proxima/core/kernels/geometry.py)
- [src/proxima/core/batch.py](src/proxima/core/batch.py)
- [src/proxima/core/metric_core.py](src/proxima/core/metric_core.py)
- [tests/core/test_metric_core.py](tests/core/test_metric_core.py)

### Artefactos de Fase 2 (Simulador Relativista):
- [configs/phase2/simulator_defaults.yaml](configs/phase2/simulator_defaults.yaml)
- [src/proxima/simulator/config.py](src/proxima/simulator/config.py)
- [src/proxima/simulator/state.py](src/proxima/simulator/state.py)
- [src/proxima/simulator/geodesic.py](src/proxima/simulator/geodesic.py)
- [src/proxima/simulator/integrator.py](src/proxima/simulator/integrator.py)
- [src/proxima/simulator/tidal.py](src/proxima/simulator/tidal.py)
- [src/proxima/simulator/energy_integral.py](src/proxima/simulator/energy_integral.py)
- [src/proxima/simulator/energy/hawking_ellis.py](src/proxima/simulator/energy/hawking_ellis.py)
- [src/proxima/simulator/energy/observer_search.py](src/proxima/simulator/energy/observer_search.py)
- [src/proxima/simulator/energy/report.py](src/proxima/simulator/energy/report.py)
- [src/proxima/simulator/energy/warpax_backend.py](src/proxima/simulator/energy/warpax_backend.py)
- [src/proxima/simulator/risk.py](src/proxima/simulator/risk.py)
- [src/proxima/simulator/simulator.py](src/proxima/simulator/simulator.py)
- [src/proxima/simulator/batch.py](src/proxima/simulator/batch.py)
- Suite de verificacion: `tests/simulator/` y `tests/contracts/`.


## 10. Implementacion de Fase 3 (completada)

### Artefactos de Fase 3 (Capa de Datos Astronomicos y Cache):
- [configs/phase3/data_defaults.yaml](configs/phase3/data_defaults.yaml)
- Catalogos locales: [sharpless.csv](configs/phase3/catalogs/sharpless.csv), [mwsc.csv](configs/phase3/catalogs/mwsc.csv), [snrs.csv](configs/phase3/catalogs/snrs.csv)
- [src/proxima/data/config.py](src/proxima/data/config.py) - configuracion (TTL, gobernanza, escalas de riesgo, entorno ISM).
- [src/proxima/data/schema.py](src/proxima/data/schema.py) - esquema comun `CelestialRecord` (SI + ICRS).
- [src/proxima/data/normalizer.py](src/proxima/data/normalizer.py) - homologacion de unidades y Galactico -> ICRS.
- [src/proxima/data/cache.py](src/proxima/data/cache.py) - cache L1 (LRU memoria) + L2 (disco) con TTL, version y checksum.
- [src/proxima/data/governance.py](src/proxima/data/governance.py) - rate-limit, reintentos con backoff, lotes, guarda anti-microconsultas SIMBAD.
- Conectores ([src/proxima/data/connectors/](src/proxima/data/connectors/)):
  - `exoplanet_tap.py` - NASA Exoplanet Archive TAP/ADQL (`pscomppars`).
  - `gaia.py` - Gaia DR3 via astroquery (jobs asincronos para cargas grandes).
  - `simbad.py` - SIMBAD solo por lotes.
  - `bayestar.py` - dustmaps Bayestar19 (E(B-V) = 0.884 x Bayestar), lotes + cache.
  - `mist.py` - MIST via isochrones para estimar L/R faltantes.
  - `local_csv.py` - Sharpless, MWSC, SNRs (version = checksum del archivo).
- [src/proxima/data/crossmatch.py](src/proxima/data/crossmatch.py) - resolucion de identidades (alias + posicion, union-find determinista).
- [src/proxima/data/enrichment.py](src/proxima/data/enrichment.py) - extincion/columna de H por objeto y entorno local.
- [src/proxima/data/snapshot.py](src/proxima/data/snapshot.py) - snapshots con manifest firmado (SHA-256) y versiones de dependencias.
- [src/proxima/data/risk_provider.py](src/proxima/data/risk_provider.py) - `AstronomicalRiskProvider` (implementa `RiskDataProviderContract`).
- [src/proxima/data/pipeline.py](src/proxima/data/pipeline.py) - orquestacion ingestion -> cross-match -> enriquecimiento -> snapshot.
- Suite de verificacion: `tests/data/` (esquema por conector, cross-match, invalidacion de cache, reproducibilidad de snapshots, riesgo).

### Notas operativas:
- Las librerias externas (astroquery, dustmaps, isochrones, astropy) son opcionales: `pip install -e .[astro]`.
  Se importan de forma perezosa; sin ellas el conector lanza `DataProviderUnavailable`.
- Bayestar19 requiere descargar el mapa una vez (`dustmaps.bayestar.fetch()`).
- Las pruebas usan backends inyectados deterministas: no requieren red.
- El riesgo por segmento es analitico y monotono en la longitud del segmento:
  fluencia radiante (J/m^2), columna de H (m^-2), densidad local maxima (kg/m^3) y marea maxima (1/s^2).
- La plantilla de contratos (`tests/contracts/`) usa ya el proveedor real basado en catalogos locales.


## 11. Implementacion de Fase 4 (completada)

### Artefactos de Fase 4 (Optimizador PSO Vectorizado):
- [configs/phase4/optimizer_defaults.yaml](configs/phase4/optimizer_defaults.yaml) - configuración de hiperparámetros (inercia, cognitivo, social, topología, pesos y penalizaciones).
- [src/proxima/optimizer/config.py](src/proxima/optimizer/config.py) - dataclass `PSOConfig`.
- [src/proxima/optimizer/fitness.py](src/proxima/optimizer/fitness.py) - evaluador de fitness contractual `RelativisticFitnessEvaluator` (cumple `FitnessEvaluatorContract`) y función pura vectorizada `jax_fitness_breakdown`.
- [src/proxima/optimizer/topology.py](src/proxima/optimizer/topology.py) - topologías de comunicación: `get_global_best` (estrella global) y `get_ring_best` (anillo local).
- [src/proxima/optimizer/swarm.py](src/proxima/optimizer/swarm.py) - inicialización de enjambre (`init_swarm`), dinámica de velocidad/posición (`update_particle_dynamics`) con sujeción de velocidad y límites de dominio, y actualización histórica (`update_pbest`).
- [src/proxima/optimizer/optimizer.py](src/proxima/optimizer/optimizer.py) - motor `VectorizedPSO` con `optimize_function` y `optimize_warp` (acoplado a `simulate_batch` de Fase 2).
- [src/proxima/optimizer/__init__.py](src/proxima/optimizer/__init__.py) - interfaz pública del módulo.
- Suite de verificación: `tests/optimizer/`:
  - `test_convergence_benchmarks.py` - convergencia monótona en funciones sintéticas (Sphere, Rosenbrock, Rastrigin) y topología en anillo.
  - `test_hyperparameter_sensitivity.py` - reproducibilidad por semilla (`PRNGKey`), variabilidad entre semillas e impacto del cronograma de inercia y cantidad de partículas.
  - `test_fitness_penalties.py` - conformidad contractual estricta, penalizaciones por violaciones de energía (WEC, NEC, SEC, DEC), sobretensión de marea, riesgo ambiental y fallos de integración.
  - `test_warp_optimization_e2e.py` - optimización end-to-end de los 5 parámetros de la métrica warp acoplada al integrador geodésico relativista.
- Actualizada la plantilla de pruebas de contrato en [tests/contracts/conftest.py](tests/contracts/conftest.py) conectando `RelativisticFitnessEvaluator`.


## 12. Implementacion de Fase 5 (completada)

### Artefactos de Fase 5 (Calibracion, Validacion Integral y Performance):
- [src/proxima/validation/calibration.py](src/proxima/validation/calibration.py) - motor de calibración:
  - `sweep_pso_hyperparameters`: barrido en cuadrícula multivariante promediado sobre múltiples semillas estocásticas con ordenamiento automático por desempeño.
  - `analyze_warp_sensitivity`: cálculo de gradientes de sensibilidad normalizados $\frac{\partial F/F}{\partial \theta_i / \theta_i}$ perturbando cada parámetro warp en $\pm\delta_{rel}$.
  - `evaluate_data_robustness`: análisis Monte Carlo de robustez ante incertidumbre estocástica en datos astronómicos y métrica de estabilidad.
- [src/proxima/validation/integral.py](src/proxima/validation/integral.py) - validador integral end-to-end:
  - `run_canonical_case`: ejecución de casos canónicos completos (Núcleo Métrico BM $\rightarrow$ Integrador Relativista $\rightarrow$ Riesgo Astronómico $\rightarrow$ Evaluador de Fitness).
  - `run_regression_suite`: suite de no-regresión numérica contra líneas base persistidas en formato JSON.
  - `verify_snapshot_reproducibility`: verificación criptográfica de consistencia y reproducibilidad de snapshots de datos.
- [src/proxima/validation/performance.py](src/proxima/validation/performance.py) - análisis y perfilado de rendimiento:
  - `profile_full_cycle`: instrumentación y desglose de tiempos de compilación JIT, integración geodésica por lotes (`simulate_batch`) y dinámicas vectorizadas de la PSO, midiendo throughput (partículas/s).
  - `benchmark_batch_scaling`: barrido de escalabilidad sobre tamaños de lote para determinar el punto óptimo de saturación de CPU/JAX.
- [src/proxima/validation/__init__.py](src/proxima/validation/__init__.py) - interfaz pública unificada de validación.
- Suite de pruebas de Fase 5 en [tests/validation/](tests/validation/):
  - `test_calibration.py` (barrido de hiperparámetros, sensibilidad de parámetros warp y robustez Monte Carlo).
  - `test_integral.py` (ejecución de casos canónicos, no-regresión con tolerancia relativa $\le 10^{-5}$ y reproducibilidad de snapshots).
  - `test_performance.py` (perfilado de desglose porcentual, throughput y escalabilidad de lotes).


## 13. Implementacion de Fase 6 (completada)

### Artefactos de Fase 6 (Entrega Cientifico-Operativa):
- [docs/fase6/MANUAL_TECNICO_CIENTIFICO.tex](docs/fase6/MANUAL_TECNICO_CIENTIFICO.tex) - Documento técnico y científico en \LaTeX{} compilado exitosamente en PDF ([MANUAL_TECNICO_CIENTIFICO.pdf](docs/fase6/MANUAL_TECNICO_CIENTIFICO.pdf)), con fundamentación de la métrica ADM/Bobrick-Martire, tensores de curvatura, integración geodésica, condiciones de Hawking-Ellis, normalización ICRS/SI, modelado analítico de riesgo en travesía, formulación matemática vectorizada del PSO y análisis de rendimiento.
- [docs/fase6/GUIA_EJECUCION_EXPERIMENTOS.md](docs/fase6/GUIA_EJECUCION_EXPERIMENTOS.md) - Guía operativa para la configuración, ejecución de optimizaciones warp, calibración de sensibilidad y auditorías de no-regresión.
- [docs/fase6/PROTOCOLO_RELEASE_TRAZABILIDAD.md](docs/fase6/PROTOCOLO_RELEASE_TRAZABILIDAD.md) - Protocolo formal de control de versiones semántico, inmutabilidad de metadatos de corridas y lista de verificación pre-release.
- [docs/fase6/CATALOGO_RIESGOS_MITIGACIONES.md](docs/fase6/CATALOGO_RIESGOS_MITIGACIONES.md) - Catálogo consolidado de riesgos geométricos, físicos y de conectividad, con sus mitigaciones operativas activas en el código.

### Estado Final del Proyecto:
- **Fases 0 a 6 completadas al 100%**.
- Suite completa de 128 pruebas unitarias, de contratos, de integración y de no-regresión pasando de forma consistente.
- Sistema completamente funcional, desacoplado, trazable y reproducible.


## 14. Empaquetado y Distribución como Paquete Pip (proxima-warp)

### Análisis y Diseño del Empaquetado:
1. **Identificador del Paquete en PyPI / Pip**: `proxima-warp` (versión `1.0.0`), manteniendo el espacio de nombres de importación limpio: `import proxima`.
2. **Estándares Modernos de Python Packaging**:
   - Cumplimiento estricto de **PEP 517 / PEP 518 / PEP 621** mediante `pyproject.toml` con backend `setuptools.build_meta`.
   - Soporte de tipado estático (**PEP 561**) mediante el marcador [src/proxima/py.typed](src/proxima/py.typed).
3. **Inclusión de Activos y Catálogos (Package Data)**:
   - Configuración explícita en `pyproject.toml` para incluir archivos de metadatos `*.yaml` y catálogos astronómicos locales `*.csv` (Sharpless, MWSC, SNRs) dentro del artefacto binario `.whl`.
   - Modificación resiliente de los resolutores de rutas en `src/proxima/core/params.py`, `src/proxima/simulator/config.py`, `src/proxima/data/config.py` y `src/proxima/optimizer/config.py` para priorizar los datos empaquetados en `site-packages` cuando la librería se instala externamente.
4. **Punto de Entrada Unificado de la API**:
   - Enriquecido [src/proxima/__init__.py](src/proxima/__init__.py) exportando directamente las clases fundamentales (`BobrickMartireMetricCore`, `RelativisticSimulator`, `VectorizedPSO`, `AstronomicalRiskProvider`, `WarpParameterVector`, etc.) y la versión `__version__ = "1.0.0"`.
5. **Artefacto de Distribución Generado**:
   - Archivo Wheel multiplataforma generado en `dist/proxima_warp-1.0.0-py3-none-any.whl` (93 KB).
   - Validada la instalación y ejecución desde un entorno y directorio externo sin dependencias del repositorio local.
