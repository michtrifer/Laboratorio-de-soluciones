"""Genera los notebooks de notebooks/ a partir de celdas definidas aquí (fuente única).

Uso: python tools/make_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = """import sys
from pathlib import Path
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
import pandas as pd, numpy as np, matplotlib.pyplot as plt
pd.set_option("display.width", 160)
%matplotlib inline"""

NOTEBOOKS = {
    "01_eda.ipynb": [
        ("md", "# 01 — EDA de retornos y quiebres estructurales (#5, #6)\n\n"
               "Requiere haber corrido `python -m src.data.download_data --with-exogenous` y "
               "`python -m src.data.clean_data` (o `python run_pipeline.py --steps data`)."),
        ("code", SETUP),
        ("code", "from src import config\nfrom src.utils.io import load_frame\n"
                 "processed = {n: load_frame(config.PROCESSED_DIR / f'{n}.csv') for n in config.ASSETS}\n"
                 "{n: (df.index.min().date(), df.index.max().date(), len(df)) for n, df in processed.items()}"),
        ("md", "## Tabla 1 — estadísticos descriptivos y pruebas\n"
               "`***` p<0.01, `**` p<0.05, `*` p<0.10. Q = Ljung-Box, ARCH-LM = Engle."),
        ("code", "from src.analysis.eda import descriptive_table, format_table\n"
                 "tab = descriptive_table({n: df['log_return'] for n, df in processed.items()})\n"
                 "format_table(tab)"),
        ("md", "**Cómo leerla:** curtosis en exceso > 0 y JB/Shapiro significativos → colas pesadas "
               "(no normalidad). Q²(10) y ARCH-LM significativos → *volatility clustering*, "
               "lo que justifica modelos GARCH."),
        ("code", "from src.analysis.eda import plot_price_returns, plot_distribution, plot_acf_returns\n"
                 "for n, df in processed.items():\n"
                 "    plot_price_returns(df['price'], df['log_return'], n.upper()); plt.show()\n"
                 "    plot_distribution(df['log_return'], n.upper()); plt.show()\n"
                 "    plot_acf_returns(df['log_return'], title=n.upper()); plt.show()"),
        ("md", "## Quiebres estructurales en varianza (ICSS)\n"
               "Se usa el estadístico κ2 de Sansó et al. (2004), robusto a colas pesadas y GARCH. "
               "El IT original se muestra para comparar (suele sobredetectar)."),
        ("code", "from src.features.regimes import break_dates, regime_table\n"
                 "from src.analysis.eda import plot_breaks\n"
                 "for n, df in processed.items():\n"
                 "    r = df['log_return']\n"
                 "    b = break_dates(r, method='kappa2')\n"
                 "    print(n, '→ κ2:', len(b), 'quiebres | IT:', len(break_dates(r, method='it')))\n"
                 "    display(regime_table(r, b))\n"
                 "    plot_breaks(r, b, f'{n.upper()} — ICSS κ2'); plt.show()"),
        ("md", "## Notas para el reporte\n- Relacionar cada fecha de quiebre con eventos "
               "(p. ej. 2016 elecciones EE.UU., 2020 COVID-19, 2022 alzas de tasas)."),
    ],
    "02_modelos_y_resultados.ipynb": [
        ("md", "# 02 — Modelos, evaluación y resultados (#7–#20)\n\n"
               "Este notebook **lee** los resultados que genera `python run_pipeline.py` "
               "(no re-entrena nada), para que sea rápido y reproducible."),
        ("code", SETUP),
        ("code", "from src import config\n"
                 "T = config.TABLES_DIR\n"
                 "metrics = pd.read_csv(T / 'metricas.csv')\nmetrics"),
        ("md", "## Ratios vs GARCH(1,1) (benchmark, ADR-0004)\n< 1 = mejor que el benchmark."),
        ("code", "pd.read_csv(T / 'ratios_vs_garch.csv', index_col=[0, 1])"),
        ("md", "## Diebold-Mariano vs GARCH(1,1) (#17)\n"
               "Estadístico > 0 → el modelo tiene menor pérdida que GARCH."),
        ("code", "pd.read_csv(T / 'diebold_mariano.csv')"),
        ("md", "## Parámetros GARCH y asimetría (#8, #9)"),
        ("code", "pd.read_csv(T / 'garch_parametros.csv', index_col=0)"),
        ("md", "## Gráficos"),
        ("code", "from IPython.display import Image, display\n"
                 "for f in sorted(config.FIGURES_DIR.glob('*.png')):\n"
                 "    if f.name.startswith(('pronostico', 'shap', 'metricas')):\n"
                 "        print(f.name); display(Image(filename=str(f)))"),
        ("md", "## SHAP (#18)"),
        ("code", "pd.read_csv(T / 'shap_importancia.csv').head(10)"),
    ],
}


def build():
    out_dir = ROOT / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
                    for kind, src in cells]
        nbf.write(nb, out_dir / name)
        print("escrito", out_dir / name)


if __name__ == "__main__":
    build()
