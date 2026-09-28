# 00 — Overview

## Contexto del curso

Módulo 4 del proyecto: **Modelado y pronóstico de volatilidad** (dificultad media-alta).

- **Gran idea:** la volatilidad cambia en el tiempo y se agrupa en rachas (volatility clustering).
  Anticiparla es clave para gestión de riesgo.
- **Pregunta esencial:** ¿qué pronostica mejor la volatilidad, los modelos clásicos o el machine learning?
- **Qué hay que construir:** un modelo de pronóstico de volatilidad + una comparación honesta
  contra un referente estándar (benchmark).
- **Herramientas:** series de tiempo, GARCH, machine learning, estadística.

## Qué dice la literatura (papers subidos)

| Paper | Aporte clave | Qué tomar de él |
|---|---|---|
| **Ge et al. (2023)** — *Comparing DL Models for Volatility Prediction Using Multivariate Data* | TFT y variantes de TCN superan a GARCH/ANN-GARCH/CNN-LSTM en HV e IV para 5 activos, con testing estadístico riguroso (Shapiro-Wilk + t-test/Kruskal-Wallis) | Metodología de comparación estadística modelo-por-modelo; uso de variables exógenas (macro, otros índices) |
| **Engle (2001)** — *GARCH 101* | Explicación canónica de ARCH/GARCH(1,1), ejemplo de Value-at-Risk | Base teórica y ejemplo de validación (backtesting de VaR) |
| **Chung, S. (2024)** — *Energy Market Volatility: GARCH vs ML* | ML (XGBoost, Ridge, RF) supera a GARCH out-of-sample en la mayoría de commodities; GARCH sobre-predice, ML sub-predice; usa **SHAP** para interpretar y detectar *spillover* entre mercados | Idea de comparar sesgo de sobre/sub-predicción (MMEO/MMEU); SHAP como herramienta de interpretabilidad |
| **Chung, Espinoza & Quispe (2025)** — *Forecasting Volatility Under Structural Breaks (LatAm)* | GARCH ignora quiebres estructurales → sobreestima persistencia; LSTM/CNN dominan en horizontes medianos/largos; usa **ICSS** para detectar quiebres | Detección de quiebres estructurales (ICSS); ventanas expanding/rolling/breaks como variantes de benchmark; métricas MSFE agregado y QLIKE; tests de White y Hansen (SPA) |

## Ángulo de diferenciación (para no repetir lo ya hecho)

La idea de "GARCH vs ML" ya está muy trillada en estos 4 papers. Para que el proyecto sea
**innovador**, conviene elegir un ángulo propio en vez de replicar uno de los papers 1:1.
Ver `docs/PRD.md` sección "Propuesta de valor" para la decisión final, pero las opciones
barajadas son:

1. **Modelo híbrido con conciencia de régimen**: detectar quiebres estructurales (ICSS, como
   en el paper 4) y usar esa segmentación como *feature* (no solo para re-estimar GARCH) dentro
   de un modelo de ML/DL — es decir, el régimen entra como input, no solo como criterio de partición.
2. **Explicabilidad aplicada a mercados emergentes/México**: replicar el enfoque SHAP del paper
   de energía pero sobre activos mexicanos/latinoamericanos (IPC, tipo de cambio, etc.), algo que
   ninguno de los 4 papers hace con SHAP + LatAm a la vez.
3. **Escalera de modelos**: comparar 4-5 niveles de complejidad (naive → GARCH → GARCH-X →
   ML de árboles → DL) contra el mismo benchmark, documentando en qué punto exacto deja de haber
   mejora marginal (esto responde directamente a la "pregunta esencial" del curso de forma más
   completa que solo "GARCH vs ML").
