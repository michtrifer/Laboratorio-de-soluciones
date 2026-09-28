# ADR-0001: Proxy de volatilidad objetivo

**Estado:** Aceptado
**Fecha:** 2026-08-24

## Contexto
Necesitamos definir qué es "la volatilidad" que vamos a pronosticar. Existen varias
definiciones (Ge et al. 2023, sección 2.1): histórica (HV), realizada (RV), implícita (IV).
La IV requiere un índice de volatilidad (tipo VIX) que no existe para muchos activos
mexicanos/latinoamericanos.

## Opciones consideradas
1. **Volatilidad histórica (HV)** a 21 días (std de log-retornos) — usada por Engle (2001) y
   como ground truth de HV en Ge et al. (2023). Fácil de calcular para cualquier activo.
2. **Volatilidad implícita (IV)** — más informativa pero requiere un índice de opciones que no
   siempre existe para activos LatAm.
3. **Retorno al cuadrado como proxy diario** — más ruidoso, usado en el paper de energía como
   proxy visual pero no como target de forecast final.

## Decisión
Usar **HV a 21 días hábiles** (equivalente a 1 mes calendario) como variable objetivo, igual
que Engle (2001) y Ge et al. (2023). Esto permite aplicar el mismo proxy sin depender de datos
de opciones que pueden no existir para el activo elegido.

## Consecuencias
- Los resultados serán comparables directamente con Engle (2001) y con la columna "HV" de
  Ge et al. (2023).
- Si más adelante se consigue un índice de volatilidad implícita del activo elegido, se puede
  agregar como segunda tarea de forecast (como hicieron Ge et al. con HV e IV en paralelo),
  pero no es requisito para el MVP del proyecto.

## Nota de implementación (issue #4)
- `src/features/volatility.py::historical_volatility` calcula
  `HV_t = sqrt(252/21 · Σ_{i=t-20..t} r_i²)` con r en %, es decir, **media cero**. La media
  diaria de los log-retornos es ~0, por lo que la diferencia con la desviación estándar
  muestral es despreciable (disponible con `demean=True`).
- Motivo: con media cero, HV² es un promedio de varianzas diarias, y el pronóstico de
  GARCH (varianza diaria a 1..h pasos) se convierte **exactamente** en pronóstico de HV
  (`hv_from_variance_forecasts`). Así todos los modelos pronostican la misma variable.
- Variable objetivo a horizonte h: `y_{t,h} = HV_{t+h}` con información hasta el cierre de t.
