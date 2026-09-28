# ADR-0002: Activo(s) y fuente de datos

**Estado:** Aceptado (configurable en `src/config.py`)
**Fecha:** 2026-08-24

## Contexto
Los 4 papers de referencia ya cubren S&P500, NASDAQ, oro, plata, petróleo, gas natural, y
(el paper de LatAm) IGBVL, BOVESPA, IPSA, IPC. Para diferenciar el proyecto y que no sea una
réplica, conviene elegir un activo/mercado con menos cobertura directa en la literatura
comparada, manteniendo datos accesibles.

## Opciones consideradas
1. **IPC (México)** — ya cubierto por el paper 4 (Chung et al. 2025), pero sin SHAP ni
   framework de escalera de modelos como el que proponemos → sigue siendo diferenciador si el
   ángulo es la escalera + SHAP, no el activo en sí.
2. **USD/MXN (tipo de cambio)** — no cubierto por ninguno de los 4 papers directamente, más
   relevante para el contexto de Michelle (Guadalajara, ITESO), buena fuente de asimetría
   (crisis, políticas de Banxico/Fed).
3. **Una acción líquida del IPC** (ej. GFNORTEO, WALMEX) — más ruido idiosincrático, menos
   comparable con la literatura de índices.

## Decisión
Sugerencia: **USD/MXN** como activo principal (diferenciador claro vs. los 4 papers) + **IPC**
como activo secundario si el tiempo alcanza, para poder comentar sobre spillover entre ambos
(inspirado en el análisis de spillover del paper de energía).
Fuente: Yahoo Finance (`USDMXN=X`, `^MXX`), periodo 2010–2025, frecuencia diaria.

## Consecuencias
- Si se elige USD/MXN, no hay comparación 1:1 directa con ningún paper de referencia — hay que
  ser más cuidadoso documentando la metodología, ya que no hay un benchmark de literatura
  exacto para ese activo.
- Si se prioriza comparabilidad sobre originalidad, la alternativa más segura es replicar IPC
  del paper 4 y diferenciar solo por el framework de modelos (escalera + SHAP).

## Nota de implementación (issues #1–#3)
- Tickers en `src/config.py` (`ASSETS`, `EXOGENOUS`). Cambiar el activo principal es
  cambiar `MAIN_ASSET`; el resto del pipeline no cambia.
- Exógenas: S&P500 (`^GSPC`), VIX (`^VIX`), UST 10Y (`^TNX`), IPC como spillover local y,
  opcionalmente, la tasa objetivo de Banxico (SIE, serie SF61745, requiere `BANXICO_TOKEN`).
- Los datos crudos (`data/raw/`) se versionan en git para que los resultados sean
  reproducibles aunque Yahoo revise datos históricos.
