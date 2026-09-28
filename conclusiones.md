# Conclusiones (issue #20)

> **Cómo llenar este documento:** corre `python run_pipeline.py` y abre
> `results/reporte_resultados.md`. La sección "Conclusiones automáticas" trae los números;
> aquí se escriben con contexto económico. Borra estas notas al terminar.

## 1. ¿Qué modelo ganó?
| Horizonte | Mejor por QLIKE | Ratio vs GARCH(1,1) | ¿Significativo (DM, 5%)? |
|---|---|---|---|
| 1 día  |  |  |  |
| 20 días |  |  |  |

## 2. ¿Por qué? (evidencia)
- **Escalera de modelos:** ¿en qué escalón deja de haber mejora marginal?
  (naive → EWMA → GARCH → GJR → XGBoost → XGBoost+régimen → híbrido). Ver `ratios_vs_garch.csv`.
- **Asimetría:** ¿`gamma[1]` de GJR-GARCH es significativo? ¿GJR mejora a GARCH(1,1)?
  (para USD/MXN, ¿las depreciaciones del peso elevan más la volatilidad que las apreciaciones?)
- **Quiebres estructurales:** fechas detectadas por ICSS y el evento asociado. ¿La variante
  `garch11_breaks` mejora al expanding? (Chung et al. 2025 esperan que sí.)
- **SHAP:** las 3–5 variables más importantes y su lectura económica (p. ej. VIX → aversión
  global al riesgo; S&P500 → spillover).
- **Sesgo:** ¿GARCH sobre-predice y el ML sub-predice, como en Chung (2024)? Ver MME_U / MME_O
  y `pct_sobreprediccion`.

## 3. Limitaciones
- HV21 es un proxy ruidoso de la volatilidad verdadera (no hay datos intradía).
- ML/DL no se re-entrenan durante el test (ADR-0003).
- Un solo activo principal; resultados no necesariamente generalizables.
- Diebold-Mariano compara pares; no controla por comparaciones múltiples (SPA de Hansen sería
  la extensión natural).

## 4. Trabajo futuro
- Volatilidad realizada con datos intradía; volatilidad implícita si existe un índice.
- Modelos multivariados (DCC/BEKK) para el spillover USD/MXN–IPC.
- Transformers (TFT) como en Ge et al. (2023); test SPA/MCS.
