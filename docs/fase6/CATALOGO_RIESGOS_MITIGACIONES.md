# Catálogo de Riesgos Técnicos, Físicos y Mitigaciones (Fase 6)

Este documento consolida los riesgos conocidos en el modelado, integración y optimización de métricas warp en **PROXIMA**, junto con sus mecanismos de mitigación activos en el código fuente.

---

## 1. Riesgos Numéricos y Geométricos

### 1.1 Singularidad o Indeterminación en el Origen de la Burbuja ($r_s \to 0$)
- **Descripción**: La derivada euclídea $\partial r_s / \partial x^i$ presenta indeterminación analítica en el centro exacto de la burbuja.
- **Impacto**: Generación de valores `NaN` o `Inf` durante la diferenciación automática en JAX.
- **Mitigación Activa**: Regularización hiperbólica $r_{s, \text{reg}} = \sqrt{r_s^2 + (\epsilon_{\text{rel}} R)^2}$ con $\epsilon_{\text{rel}} = 10^{-9}$ en `src/proxima/core/kernels/safety.py`.

### 1.2 Deriva de la Norma Temporal de la 4-Velocidad
- **Descripción**: En integraciones numéricas prolongadas, el error de truncamiento acumulado puede violar la condición física $g_{\mu\nu} u^\mu u^\nu = -1$.
- **Impacto**: Pérdida del carácter temporal de la geodésica y tiempos propios ficticios.
- **Mitigación Activa**: Normalización analítica forzada en cada paso del integrador mediante `normalize_4velocity` y guarda de tolerancia $\le 10^{-6}$.

---

## 2. Riesgos Físicos y de Seguridad del Vehículo

### 2.1 Gradientes Extremos de Marea (Spaghettification)
- **Descripción**: Valores muy reducidos de espesor de pared $\sigma$ generan curvaturas de Riemann divergentes y fuerzas de marea destructivas.
- **Impacto**: Cizallamiento estructural del navío o carga útil.
- **Mitigación Activa**: Cálculo covariante del tensor de marea espacial $\mathcal{E}_{ij}$ en la tétrada de Fermi-Walker con interrupción inmediata ante $\sigma_{\text{tidal}} > 9.81\text{ s}^{-2}$ (`TERM_TIDAL_LIMIT`).

### 2.2 Violación Severa de Condiciones de Energía (WEC / NEC / DEC / SEC)
- **Descripción**: Métricas superluminales o geometrías con gradientes pronunciados requieren densidades de energía negativas.
- **Impacto**: Soluciones no realizables bajo materia clásica.
- **Mitigación Activa**: Diagonalización de Hawking-Ellis en `src/proxima/simulator/energy/` y penalización cuadrática estricta en el evaluador de fitness.

---

## 3. Riesgos de Datos y Conectividad

### 3.1 Agotamiento de Cuotas y Bloqueo de IP en APIs Remotas
- **Descripción**: Microconsultas masivas o repetitivas hacia SIMBAD, Gaia o NASA Exoplanet Archive.
- **Impacto**: `RateLimitExceeded` y caída de la tubería de datos.
- **Mitigación Activa**: 
  - Caché multinivel (L1 memoria + L2 disco persistente con SHA-256).
  - Consulta obligatoria por lotes (`chunked_batches`).
  - Guarda estricta contra consultas individuales consecutivas (`SimbadLoopGuard`).
  - Reintentos con retroceso exponencial (*exponential backoff*).
