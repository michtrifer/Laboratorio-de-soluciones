# PRD — Modelo de Pronóstico de Volatilidad

## 1. Objetivo

Construir y comparar honestamente varios modelos de pronóstico de volatilidad, de menor a
mayor complejidad, contra un benchmark estándar, para responder: **¿qué predice mejor la
volatilidad, los modelos clásicos (GARCH) o el machine learning — y bajo qué condiciones?**

## 2. Propuesta de valor (el ángulo "innovador")

> **Decisión a tomar en `docs/adr/0002-activos-y-datos.md`**, pero el PRD asume por defecto:

Construir una **"escalera de modelos"** (naive → GARCH(1,1) → GARCH con variantes/exógenas →
ML de árboles → DL) evaluada con un **framework de régimen** (quiebres estructurales vía ICSS,
igual que Chung et al. 2025) aplicado a **uno o más activos mexicanos/latinoamericanos**
(ej. IPC, USD/MXN, o una acción líquida del IPC), y usando **SHAP** para interpretar el mejor
modelo de ML (como en Chung 2024), algo que ningún paper de referencia combina exactamente así.

El benchmark no es un solo modelo: es el **GARCH(1,1) con ventana expanding**, tal como en el
paper 4, para que los resultados sean comparables con literatura reciente.

## 3. Alcance (Scope)

### Incluido (in-scope)
- Un activo principal (o dos si el tiempo lo permite) con al menos 8-10 años de datos diarios.
- Definición de volatilidad objetivo: **volatilidad realizada/histórica (HV)** sobre ventana de
  ~21 días (mismo enfoque que Ge et al. y Engle), calculada como proxy (no volatilidad implícita,
  a menos que se consiga un índice tipo VIX del activo elegido).
- Modelos (mínimo):
  1. Naive (persistencia / rolling std)
  2. GARCH(1,1) — benchmark oficial
  3. Al menos 1 variante GARCH (EGARCH o GJR-GARCH) para capturar asimetría
  4. 1 modelo de ML "de árboles" (XGBoost o Random Forest)
  5. 1 modelo de DL simple (LSTM) — opcional si el tiempo alcanza
  6. (Opcional, si da tiempo) 1 modelo híbrido: features de régimen (ICSS) + GARCH output como
     input del modelo de ML
- Detección de quiebres estructurales (ICSS o versión simplificada) para al menos describir
  el activo y, si el tiempo alcanza, como feature.
- Evaluación out-of-sample con métricas estándar (ver sección 5) + test estadístico de
  significancia (Diebold-Mariano o el test de White/Hansen simplificado).
- Interpretabilidad: SHAP sobre el mejor modelo de ML.
- Reporte final con hallazgos, limitaciones y cuál modelo "gana" y bajo qué métrica/horizonte.

### Excluido (out-of-scope, para no sobre-extender el proyecto)
- Modelos multivariados/BEKK (correlaciones entre activos) — se menciona como trabajo futuro.
- Transformers (TFT) — demasiado costoso para el timeline del curso; se puede citar como
  extensión futura.
- Trading en vivo / implementación en producción.
- Más de 2 activos (a menos que el tiempo lo permita cómodamente).

## 4. Datos

- **Fuente:** Yahoo Finance (gratis, ya usado en 2 de los 4 papers) o Banxico/INEGI si se
  quiere una variable exógena mexicana (tasa de referencia, tipo de cambio, etc.).
- **Frecuencia:** diaria.
- **Periodo sugerido:** 2010–2025 (incluye COVID-19, útil para quiebres estructurales).
- **Variables exógenas candidatas** (opcional, inspiradas en Ge et al. y Chung 2024): tipo de
  cambio USD/MXN, tasa de referencia de Banxico, S&P500 (spillover), volumen.

## 5. Métricas de evaluación

- **MSFE** (Mean Squared Forecast Error) y **QLIKE** (quasi-likelihood loss) — igual que en
  Chung et al. 2025, porque QLIKE penaliza menos los picos que MSFE.
- **MAE** como métrica secundaria/interpretable.
- Sesgo de sobre/sub-predicción (**MMEO/MMEU**, del paper de energía) — opcional pero da
  narrativa rica ("¿el modelo sobreestima o subestima el riesgo?").
- Test de significancia estadística entre modelos (Diebold-Mariano como versión simplificada
  de White/Hansen).
- Horizontes de evaluación: al menos 1 día y 20 días (corto y mediano plazo).

## 6. Entregables

1. Repositorio de código organizado (este mismo repo).
2. Notebook(s) reproducibles con el pipeline completo.
3. Reporte/documento final (puede ser el mismo PRD actualizado + resultados, o un doc aparte).
4. Tabla comparativa final de modelos vs benchmark, por métrica y horizonte.
5. Gráficos: volatilidad real vs pronosticada, quiebres estructurales detectados, SHAP summary
   plot del mejor modelo de ML.

## 7. Criterios de éxito

- Al menos un modelo de la "escalera" supera al benchmark GARCH(1,1) de forma estadísticamente
  significativa en al menos un horizonte.
- El reporte explica **por qué** (o por qué no) el modelo más complejo gana — no solo reporta
  números.
- Todo el pipeline corre de punta a punta sin intervención manual (reproducible).

## 8. Riesgos / supuestos

- Riesgo: no conseguir una buena variable de volatilidad implícita para el activo mexicano →
  mitigación: usar solo HV (volatilidad histórica), como Engle y como fallback en Ge et al.
- Riesgo: LSTM no converge bien con poco tiempo de tuning → mitigación: dejarlo como "nice to
  have", priorizar GARCH + ML de árboles que son más rápidos de iterar.
- Supuesto: se cuenta con Python (statsmodels/arch, scikit-learn, xgboost, shap, tensorflow/pytorch
  opcional).
