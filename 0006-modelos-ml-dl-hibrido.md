# ADR-0006: Diseño de los modelos de ML, LSTM e híbrido

**Estado:** Aceptado
**Fecha:** 2026-09-28

## Contexto
Los issues #12–#14 piden XGBoost/Random Forest, un LSTM simple (opcional) y un modelo
híbrido. Hay que decidir qué pronostica exactamente el ML y cómo evitar sobreajuste y fuga
de información, respetando ADR-0001 (objetivo HV_{t+h}) y ADR-0003 (split 80/20).

## Opciones consideradas
1. **ML pronostica el nivel de HV_{t+h}** — directo, pero los árboles dan salidas
   escalonadas y pierden contra modelos suaves a 1 día, donde 20 de los 21 retornos del
   objetivo ya se conocen.
2. **ML pronostica una corrección sobre un pronóstico base** — log(HV_{t+h} / B_{t,h}).
   El árbol solo aprende lo que la base no captura; con pocos datos es un problema más fácil.
3. **Híbrido "GARCH-ANN"** (Ge et al.) — red neuronal con salidas de GARCH como inputs;
   más costoso, y difícil de interpretar con SHAP.

## Decisión
- **XGBoost / Random Forest (#12):** objetivo log(HV_{t+h} / EWMA_{t,h}). La base EWMA
  (λ=0.94) no tiene parámetros estimados, así que no introduce fuga. Los hiperparámetros se
  eligen con QLIKE en la validación interna (últimos 15% del train, con purga de h días) y
  luego se re-entrena en todo el train. Un **factor de encogimiento** λ ∈ {0, .25, .5, .75, 1}
  elegido en validación escala la corrección: si el ML no ayuda, λ=0 y el modelo regresa a
  la base, en lugar de empeorarla.
- **Escalera:** `xgboost` (base+exógenas) → `xgboost_regimen` (+ features ICSS) →
  `hibrido` (+ salida de GARCH), para medir el aporte marginal de cada bloque de información.
- **Híbrido (#14):** igual que (2), pero la base es el pronóstico de GARCH(1,1) con
  parámetros estimados solo en train; features = base + exógenas + régimen + GARCH.
  Justificación en `src/models/hybrid.py`: GARCH ignora quiebres y sobreestima persistencia
  (Chung et al. 2025); las features de régimen permiten al ML corregir ese sesgo.
- **LSTM (#13, opcional):** 1 capa de 32 unidades, dropout 0.2, secuencias de 21 días de las
  mismas features que `xgboost_regimen`, objetivo log(HV_{t+h}), Adam lr=1e-3, early stopping
  (paciencia 10) en validación y re-entrenamiento en todo el train con las épocas óptimas.
  Solo corre si PyTorch está instalado.

## Consecuencias
- SHAP (#18) explica la **corrección** sobre la base, no el nivel de volatilidad: la
  interpretación es "qué variables hacen que el ML suba o baje el pronóstico de la base".
- Si en los datos reales el encogimiento elegido es 0, eso es un hallazgo en sí mismo: el ML
  no agrega información sobre el modelo base en ese horizonte (útil para las conclusiones).
- Sin re-entrenamiento durante el test (ADR-0003), el ML puede degradarse si el test cae en
  un régimen muy distinto al del train; se discute en limitaciones.
