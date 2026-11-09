# Protocolo de Release, Versionado y Trazabilidad (Fase 6)

Este documento fija el estándar formal para versionar el código, publicar entregables y garantizar la reproducibilidad científica estricta de las corridas del sistema **PROXIMA**.

---

## 1. Esquema de Versionado Semántico (SemVer)

El sistema utiliza versionado semántico formal `MAJOR.MINOR.PATCH`:
- **MAJOR**: Cambios en contratos tipados (`src/proxima/contracts/`) que rompen compatibilidad binaria o física.
- **MINOR**: Incorporación de nuevos catálogos, backends de energía, topologías de optimización o modelos estelares sin alterar las interfaces existentes.
- **PATCH**: Correcciones de estabilidad numérica, ajuste de tolerancias o refactorizaciones internas.

---

## 2. Requisitos de Trazabilidad por Corrida Experimental

Toda corrida experimental o simulación debe registrar y persistir los siguientes metadatos inmutables:

1. **Identificador del Snapshot de Datos**: Hash SHA-256 de los catálogos celestes utilizados (`SnapshotManifest.records_checksum`).
2. **Entorno de Dependencias**: Versiones congeladas de Python, JAX, NumPy, SciPy, Astropy y Astroquery.
3. **Semilla Pseudoaleatoria**: Registro explícito de la `PRNGKey` utilizada en la PSO y muestreos estocásticos.
4. **Firma Métrica y Convenciones**: `-+++`, constantes $c$, $G$ y tolerancias absolutas/relativas fijadas en la Fase 0.

---

## 3. Lista de Verificación Pre-Release (Release Checklist)

Antes de etiquetar o promover un release del software:

- [ ] **Conformidad de Contratos**: `pytest -m contract` pasa al 100% sin omitir ninguna interfaz.
- [ ] **Suite Completa de Pruebas**: Todos los tests unitarios e integrales (`pytest`) pasan exitosamente.
- [ ] **Línea Base de No-Regresión**: Validación contra `tests/validation/test_integral.py` con discrepancia relativa $\le 10^{-5}$.
- [ ] **Compilación de Documentación**: El documento LaTeX `docs/fase6/MANUAL_TECNICO_CIENTIFICO.tex` compila limpiamente sin advertencias críticas de desbordamiento.
- [ ] **Integridad de Catálogos**: Checksum de archivos CSV en `configs/phase3/catalogs/` verificado y consistente con el manifest.
