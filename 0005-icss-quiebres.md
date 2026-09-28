# ADR-0005: Variante de ICSS para detectar quiebres estructurales

**Estado:** Aceptado
**Fecha:** 2026-09-28

## Contexto
El issue #6 pide detectar quiebres estructurales en la varianza (ICSS, como Chung et al.
2025). El estadístico original de Inclán & Tiao (1994) supone retornos i.i.d. normales;
con colas pesadas y efectos GARCH (lo que confirma el EDA, #5) su tamaño real es muy
superior al 5% nominal y detecta "quiebres" que en realidad son clusters de volatilidad.

## Opciones consideradas
1. **IT original** — el de la mayoría de los papers; sobredetecta con datos financieros.
2. **κ2 de Sansó, Aragó & Carrión (2004)** — corrige por curtosis y dependencia con una
   varianza de largo plazo tipo Newey-West; tamaño cercano al nominal.
3. Métodos bayesianos / PELT — fuera del alcance del curso.

## Decisión
Usar **ICSS con κ2** (valor crítico 1.405 al 5%) y segmento mínimo de 63 días. El IT se
conserva en el código (`method="it"`) y se reporta en el notebook como comparación.

## Consecuencias
- Menos quiebres, pero más creíbles: cada uno debe poder asociarse a un evento económico.
- Para usar el régimen como *feature* (#11, #14) sin fuga de información, ICSS se corre en
  ventana expanding cada 21 días (`expanding_regime_features`), nunca sobre toda la muestra.
- Para GARCH segmentado por quiebres (#10) se usa el mismo criterio, también expanding.
