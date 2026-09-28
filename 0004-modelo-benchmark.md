# ADR-0004: Definición del modelo benchmark

**Estado:** Aceptado
**Fecha:** 2026-08-24

## Contexto
El PRD requiere "una comparación honesta contra un referente estándar". Necesitamos fijar
UN modelo benchmark antes de construir el resto, para que todas las comparaciones sean contra
el mismo punto de referencia (y no cambiar de benchmark a mitad de proyecto).

## Opciones consideradas
1. **Naive (persistencia)** — el más simple, usado como piso mínimo en todos los papers.
2. **GARCH(1,1) con ventana expanding** — el estándar de facto en la literatura financiera
   (Engle 2001 lo llama "el más simple y robusto de la familia GARCH"); es el benchmark
   oficial en Chung et al. (2025).
3. **ANN-GARCH** — benchmark más fuerte en Ge et al. (2023), pero es en sí mismo un modelo
   híbrido, no un "referente estándar" simple.

## Decisión
El benchmark oficial del proyecto es **GARCH(1,1) con ventana expanding**. El modelo naive se
mantiene como "piso" de referencia adicional (para poder decir "todos los modelos superan al
naive", que es la barra mínima de sanidad), pero **todas las comparaciones de significancia
estadística (Diebold-Mariano) se hacen contra GARCH(1,1)**, no contra el naive.

## Consecuencias
- Cualquier modelo nuevo que se agregue al proyecto se reporta como "ratio vs GARCH(1,1)"
  (igual que las Tablas 4 y 5 de Chung et al. 2025), lo que facilita leer resultados.
- Si GARCH(1,1) no converge bien para el activo elegido (poco probable, pero posible con
  series muy cortas o muy ruidosas), este ADR debe revisarse y reemplazarse.
