# ISSUES — Backlog

> Estado: ✅ implementado. La tabla al final indica dónde vive cada issue.

Convención: `[EPIC] Título` agrupa issues relacionados. Cada issue tiene *Definition of Done (DoD)*.

## EPIC 1 — Setup y datos

- ✅ **#1 Descargar datos históricos del/los activo(s) elegido(s)**
  DoD: CSV/parquet en `data/raw/` con OHLCV diario, sin huecos sin justificar, ≥8 años.
- ✅ **#2 Limpieza y cálculo de retornos**
  DoD: función en `src/data/` que calcula log-retornos, maneja missing values, guarda en `data/processed/`.
- ✅ **#3 Descargar variables exógenas (opcional)**
  DoD: tipo de cambio / tasa Banxico / índice de referencia alineados por fecha con el activo principal.
- ✅ **#4 Definir y calcular el proxy de volatilidad objetivo (HV)**
  DoD: función que calcula HV a 21 días (ver ADR 0001), con tests unitarios básicos.

## EPIC 2 — Análisis exploratorio

- ✅ **#5 EDA de retornos**: estadísticos descriptivos, skewness/kurtosis, normalidad (Shapiro-Wilk),
  autocorrelación de retornos al cuadrado (Ljung-Box), test ARCH-LM.
  DoD: notebook con gráficos + tabla tipo Tabla 1 del paper de Chung et al. (LatAm).
- ✅ **#6 Detección de quiebres estructurales (ICSS o proxy simplificado)**
  DoD: lista de fechas de quiebre + gráfico de retornos con quiebres marcados.

## EPIC 3 — Benchmark y modelos clásicos

- ✅ **#7 Naive model** (rolling std de 20-21 días)
  DoD: función de forecast + métricas out-of-sample.
- ✅ **#8 GARCH(1,1) — benchmark oficial** (ventana expanding)
  DoD: modelo entrenado con `arch` (Python), forecast recursivo out-of-sample, guardado de resultados.
- ✅ **#9 Variante GARCH asimétrica** (EGARCH o GJR-GARCH)
  DoD: mismo pipeline que #8, comparación de coeficientes de asimetría.
- ✅ **#10 (Opcional) GARCH con ventana rolling y GARCH segmentado por quiebres**
  DoD: comparación de las 3 variantes de ventana (expanding/rolling/breaks) como en el paper 4.

## EPIC 4 — Modelos de Machine Learning

- ✅ **#11 Feature engineering para ML** (lags de volatilidad, retornos al cuadrado, variables
  exógenas, indicador de régimen si aplica)
  DoD: matriz de features documentada, sin fuga de información (no usar datos futuros).
- ✅ **#12 Modelo XGBoost / Random Forest**
  DoD: modelo entrenado con validación temporal (no k-fold aleatorio), hiperparámetros documentados.
- ✅ **#13 (Opcional) LSTM simple**
  DoD: arquitectura mínima documentada en ADR, entrenado y evaluado con el mismo split que los demás.
- ✅ **#14 (Opcional) Modelo híbrido** (features de régimen + output de GARCH como input de ML)
  DoD: pipeline reproducible, justificación de por qué se espera que mejore sobre los anteriores.

## EPIC 5 — Evaluación y comparación

- ✅ **#15 Pipeline de evaluación out-of-sample unificado**
  DoD: todos los modelos se evalúan con el mismo split, mismos horizontes (1 y 20 días), mismas métricas.
- ✅ **#16 Cálculo de MSFE, QLIKE, MAE por modelo y horizonte**
  DoD: tabla comparativa reproducible (script, no cálculo manual).
- ✅ **#17 Test de significancia estadística (Diebold-Mariano)**
  DoD: p-values reportados para cada par relevante de modelos vs benchmark.
- ✅ **#18 Interpretabilidad: SHAP sobre el mejor modelo de ML**
  DoD: SHAP summary plot + interpretación escrita de las 3-5 variables más importantes.

## EPIC 6 — Reporte final

- ✅ **#19 Consolidar resultados en tabla y gráficos finales**
- ✅ **#20 Escribir conclusiones**: qué modelo ganó, en qué horizonte, por qué (usando SHAP y el
  análisis de quiebres como evidencia de apoyo), limitaciones y trabajo futuro.
- ✅ **#21 Revisión final de reproducibilidad** (correr todo el pipeline desde cero una vez).

## Dónde está implementado cada issue

| Issue | Código |
|---|---|
| #1 | `src/data/download_data.py` |
| #2 | `src/data/clean_data.py` |
| #3 | `src/features/exogenous.py` |
| #4 | `src/features/volatility.py` |
| #5 | `src/analysis/eda.py, notebooks/01_eda.ipynb` |
| #6 | `src/features/regimes.py` |
| #7 | `src/models/naive.py` |
| #8 | `src/models/garch.py` |
| #9 | `src/models/garch.py (gjr, egarch)` |
| #10 | `src/models/garch.py (window=rolling/breaks)` |
| #11 | `src/features/ml_features.py` |
| #12 | `src/models/ml_models.py` |
| #13 | `src/models/dl_models.py (requiere torch)` |
| #14 | `src/models/hybrid.py` |
| #15 | `src/pipeline.py, src/evaluation/backtest.py` |
| #16 | `src/evaluation/metrics.py` |
| #17 | `src/evaluation/statistical_tests.py` |
| #18 | `src/evaluation/interpretability.py` |
| #19 | `src/reporting/report.py → results/` |
| #20 | `docs/conclusiones.md (plantilla) + conclusiones automáticas en results/reporte_resultados.md` |
| #21 | `run_pipeline.py + tests/test_pipeline_e2e.py` |

#20 requiere tu interpretación con los datos reales: la plantilla está en `docs/conclusiones.md`.
