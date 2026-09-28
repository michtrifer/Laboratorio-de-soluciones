# Proyecto: Pronóstico de Volatilidad — GARCH vs Machine Learning (vs Híbrido)

Repositorio estructurado con metodología **DOCS → PRD → ISSUES → ADR → CODE**.

## Cómo navegar este repo

1. **`docs/00-overview.md`** — contexto general, motivación, papers de referencia.
2. **`docs/PRD.md`** — qué se va a construir exactamente (alcance, datos, modelos, métricas, entregables).
3. **`docs/ISSUES.md`** — el PRD partido en tareas concretas y accionables (backlog tipo Kanban).
4. **`docs/adr/`** — decisiones técnicas importantes, una por archivo (ADR = Architecture Decision Record).
5. **`src/`** — código, organizado por etapa del pipeline.
6. **`notebooks/`** — exploración y prototipado (lo que luego se "gradúa" a `src/`).

## Orden de trabajo recomendado

```
1. Leer docs/00-overview.md (contexto)
2. Cerrar docs/PRD.md (decidir alcance ANTES de programar)
3. Convertir PRD en docs/ISSUES.md (backlog)
4. Cada vez que tomes una decisión técnica relevante (ej. qué proxy de volatilidad usar,
   qué split usar, qué arquitectura probar) → escribe un ADR en docs/adr/
5. Programar siguiendo el backlog, issue por issue
```

## Cómo correr el proyecto

Desde la raíz del repo (terminal de PyCharm o Git Bash):

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash en Windows  (en PowerShell: .venv\Scripts\activate)
pip install -r requirements.txt

python -m pytest -m "not slow"       # pruebas rápidas (~10 s)
python run_pipeline.py               # TODO: descarga → EDA → modelos → evaluación → reporte
```

| Comando | Qué hace |
|---|---|
| `python run_pipeline.py` | Pipeline completo desde cero (#21) |
| `python run_pipeline.py --skip-download` | Reusa `data/raw/` (resultados idénticos) |
| `python run_pipeline.py --steps models` | Solo modelos + evaluación + reporte |
| `python run_pipeline.py --refit-every 5` | GARCH re-estimado cada 5 días (más rápido) |
| `python run_pipeline.py --egarch` | Agrega EGARCH (lento: pronóstico por simulación) |
| `python run_pipeline.py --synthetic` | Prueba sin internet con datos simulados (en `synthetic_run/`) |
| `python -m pytest` | Todas las pruebas, incluida la de punta a punta (~2 min) |

Opcionales: `pip install torch` activa el LSTM (#13); definir `BANXICO_TOKEN` (token gratuito
del SIE de Banxico) agrega la tasa objetivo como exógena (#3).

**Salidas** (en `results/`):
- `reporte_resultados.md`: tablas, gráficos y conclusiones automáticas
- `tables/`: `metricas.csv`, `ratios_vs_garch.csv`, `diebold_mariano.csv`, `garch_parametros.csv`,
  `hiperparametros.csv`, `features.csv`, `shap_importancia.csv`, `eda_tabla1.csv`, `quiebres_*.csv`
- `figures/`: pronósticos vs real, ratios, SHAP, EDA, quiebres
- `notebooks/01_eda.ipynb` y `notebooks/02_modelos_y_resultados.ipynb` leen esos archivos.

## Estructura de carpetas

```
proyecto-volatilidad/
├── docs/
│   ├── 00-overview.md
│   ├── PRD.md
│   ├── ISSUES.md
│   └── adr/
│       ├── 0000-template.md
│       ├── 0001-proxy-de-volatilidad.md
│       ├── 0002-activos-y-datos.md
│       ├── 0003-split-y-validacion.md
│       ├── 0004-modelo-benchmark.md
│       ├── 0005-icss-quiebres.md
│       └── 0006-modelos-ml-dl-hibrido.md
├── src/
│   ├── config.py      # parámetros de los ADRs (activos, fechas, split, horizontes)
│   ├── pipeline.py    # pipeline unificado (#15)
│   ├── data/          # descarga, limpieza, datos sintéticos
│   ├── features/      # HV objetivo, exógenas, régimen (ICSS), matriz de ML
│   ├── analysis/      # EDA
│   ├── models/        # naive, GARCH, ML, LSTM, híbrido
│   ├── evaluation/    # split, métricas, Diebold-Mariano, SHAP
│   ├── reporting/     # gráficos y reporte final
│   └── utils/
├── notebooks/         # generados con tools/make_notebooks.py
├── tests/
├── data/raw/          # datos crudos (se versionan)
├── results/           # salidas del pipeline
└── run_pipeline.py
```
