# Guía Operativa de Ejecución de Experimentos (Fase 6)

Esta guía describe el protocolo estándar para configurar, ejecutar y auditar experimentos en el motor relativista **PROXIMA**.

---

## 1. Configuración del Entorno de Ejecución

El sistema opera bajo Python 3.11+ con soporte de compilación JIT acelerada por JAX.

### 1.1 Verificación de Dependencias
```bash
# Activar entorno virtual
source .venv/bin/activate

# Verificar ejecución de pruebas unitarias y de integración
pytest -q
```

---

## 2. Ejecución de Experimentos de Optimización PSO

Para ejecutar una búsqueda y optimización de configuraciones warp sobre un escenario determinado:

### 2.1 Script Básico de Optimización
```python
import jax.numpy as jnp
from proxima.optimizer import VectorizedPSO, PSOConfig
from proxima.contracts.types import WarpParameterVector

# 1. Configurar hiperparámetros
config = PSOConfig(
    n_particles=30,
    n_iterations=50,
    topology="global",  # o "ring" para mayor diversidad
    seed=42,
)

# 2. Inicializar el motor de optimización
pso = VectorizedPSO(config=config, allow_superluminal=False)

# 3. Ejecutar optimización acoplada a la simulación relativista
resultado = pso.optimize_warp(
    seed=42,
    n_particles=30,
    n_iterations=50,
)

print(f"Mejor fitness alcanzado: {resultado.best_fitness:.6e}")
print(f"Parámetros óptimos: {resultado.best_params}")
```

---

## 3. Calibración y Análisis de Sensibilidad

Antes de promover una configuración a línea base (*baseline*), se debe evaluar su sensibilidad:

```python
from proxima.validation import CalibrationEngine

engine = CalibrationEngine()

# Vector de prueba (R, sigma, v/c, A, alpha_0)
nominal = (100.0, 5.0, 0.5, 1.0, 0.01)

def eval_func(vec):
    # Función que evalúa el costo total de la configuración
    return (vec[0] / 100.0) + (vec[2] / 0.5)**2

sensibilidad = engine.analyze_warp_sensitivity(eval_func, nominal, delta_rel=0.05)
for param, grad in sensibilidad.sensitivities.items():
    print(f"Elasticidad de {param}: {grad:.4f}")
```

---

## 4. Auditoría de No-Regresión Numérica

Para verificar que ninguna modificación rompa los casos canónicos del sistema:

```python
from proxima.validation import IntegralValidator

validador = IntegralValidator()
reporte = validador.run_regression_suite()

if reporte.all_passed:
    print("Suite de no-regresión: TODOS LOS CASOS CONFORME A TOLERANCIA.")
else:
    print("Divergencia numérica detectada en los casos canónicos.")
```
