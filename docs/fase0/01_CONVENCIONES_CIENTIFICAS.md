# Fase 0 - Convenciones Cientificas

## 1. Convenciones de metrica y tensores

1. Firma metrica primaria: (-,+,+,+)
2. Coordenadas base: (t, x, y, z)
3. Indices latinos i,j,k para espacio (1..3)
4. Indices griegos mu,nu para espacio-tiempo (0..3)
5. Criterio de suma de Einstein activo

## 2. Sistema de unidades

1. Sistema base de computo: SI
2. Unidades canonicas:
- Tiempo: segundo (s)
- Longitud: metro (m)
- Masa: kilogramo (kg)
- Energia: joule (J)
- Velocidad: m/s

3. Constantes fisicas congeladas:
- c = 299792458.0 m/s
- G = 6.67430e-11 m^3 kg^-1 s^-2

4. Regla operativa:
- Toda variable fisica debe declarar su unidad de entrada y salida.
- Queda prohibida conversion implicita no documentada.

## 3. Marcos de referencia y coordenadas

1. Marco inercial global por defecto: ICRS/BCRS para datos astronomicos.
2. Marco local de simulacion: cartesiano 3+1 centrado en la nave o en la burbuja, segun experimento.
3. Toda transformacion de marco debe registrar:
- marco_origen
- marco_destino
- epoca
- precision esperada

## 4. Convenciones para geodesicas y tiempos

1. Variable de integracion primaria en simulador: tau (tiempo propio)
2. Tiempo coordenado asociado: t
3. Observables obligatorios por corrida:
- delta_tau
- delta_t
- energia_total
- tidal_stress_max

## 5. Convenciones de condiciones de energia

Se evaluan cuatro condiciones:

1. WEC
2. NEC
3. DEC
4. SEC

Regla de reporte:

1. Resultado booleano por punto y por tramo
2. Magnitud de violacion acumulada por condicion
3. Umbrales configurables por experimento

## 6. Convenciones numericas

1. Tipo de punto flotante minimo para simulacion principal: float64
2. Tolerancias base:
- atol = 1e-10
- rtol = 1e-8
3. Si una tolerancia cambia, se registra en metadata de corrida

## 7. Nomenclatura

1. Prefijo delta_ para incrementos (ejemplo: delta_t)
2. Sufijo _si para valores en SI cuando coexistan otros sistemas
3. Sufijo _frame para etiquetas de marcos
4. Sufijo _score para metricas normalizadas [0,1]

## 8. Trazabilidad minima

Cada resultado de simulacion debe guardar:

1. version_fase0
2. firma_metrica
3. constantes_fisicas
4. unidades_activas
5. tolerancias
