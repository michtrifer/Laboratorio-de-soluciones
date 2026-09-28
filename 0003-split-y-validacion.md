# ADR-0003: Estrategia de split y validación

**Estado:** Aceptado
**Fecha:** 2026-08-24

## Contexto
Al ser series de tiempo, no se puede usar k-fold aleatorio (fuga de información del futuro).
Ge et al. (2023) usan 70-15-15 (train-val-test) temporal. Chung et al. (2025) usan 80-20 con
re-estimación recursiva (expanding/rolling) durante el test.

## Opciones consideradas
1. **70-15-15 temporal fijo** (como Ge et al.) — simple, un solo entrenamiento, validación para
   hiperparámetros, test final una sola vez.
2. **80-20 con forecast recursivo** (como Chung et al. 2025) — el modelo se re-estima o se
   desliza en cada paso del test, más realista pero más costoso computacionalmente.

## Decisión
Usar **80-20 temporal**, con:
- 80% para entrenamiento (incluye una porción interna para validación de hiperparámetros en
  los modelos de ML/DL, ej. últimos 15% del 80% como validación).
- 20% final como test out-of-sample, **nunca tocado hasta la evaluación final**.
- Para GARCH: ventana expanding (igual que el benchmark oficial del PRD) — re-estimar en cada
  paso del test es el estándar en la literatura y es lo que lo hace comparable como benchmark real.
- Para ML/DL: entrenamiento único sobre el 80%, forecast sobre el 20% con features actualizadas
  (sin re-entrenar en cada paso, por costo computacional — igual que Chung et al. 2025 hacen
  con LSTM/CNN).

## Consecuencias
- Es fundamental que las features de ML/DL en el set de test **no usen información del futuro**
  (ej. un lag de HV debe ser HV calculada con datos hasta t, no hasta t+21).
- El código de evaluación (`src/evaluation/`) debe compartir la misma función de split para
  todos los modelos, para evitar que cada modelo tenga un test set ligeramente distinto.

## Nota de implementación (issue #15)
- `src/evaluation/backtest.py::make_split` es la ÚNICA función de split del proyecto.
- **Orígenes de test comunes:** todas las fechas del 20% final cuyo objetivo a 20 días existe;
  el horizonte de 1 día se evalúa sobre esas mismas fechas.
- **Purga:** las últimas h filas del train (y de la validación interna) se eliminan al
  entrenar a horizonte h, porque su objetivo HV_{t+h} usa retornos del periodo siguiente.
- GARCH se re-estima en cada día del test por defecto (`GARCH_REFIT_EVERY = 1`); entre
  re-estimaciones la varianza se sigue filtrando con los datos nuevos, sin ver el futuro.
