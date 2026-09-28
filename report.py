"""Issues #19 y #20 — Tablas, gráficos finales y reporte de resultados en Markdown.

El reporte (`results/reporte_resultados.md`) se genera automáticamente a partir de los CSV
del pipeline, así que siempre refleja la última corrida. Las conclusiones automáticas son un
borrador basado en los números; la versión final para entregar va en
`docs/conclusiones.md` (#20), donde se agrega el contexto económico.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BENCH = "garch11_expanding"


def _md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """DataFrame → tabla Markdown sin depender de `tabulate`."""
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append(format(v, floatfmt) if np.isfinite(v) else "—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------ gráficos
def plot_forecasts(frame: pd.DataFrame, horizon: int, models: list[str], path):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(frame.index, frame["actual"], color="black", lw=1.4, label="HV real")
    for m in models:
        if m in frame:
            ax.plot(frame.index, frame[m], lw=0.9, label=m, alpha=0.85)
    ax.set_title(f"Volatilidad real vs pronosticada — horizonte {horizon} día(s)")
    ax.set_ylabel("HV21 anualizada (%)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_metric_bars(ratios: pd.DataFrame, path, metric: str = "QLIKE"):
    hs = sorted(ratios.index.get_level_values(0).unique())
    fig, axes = plt.subplots(1, len(hs), figsize=(6 * len(hs), 4.5), squeeze=False)
    for ax, h in zip(axes[0], hs):
        r = ratios.loc[h, metric].sort_values()
        colors = ["C2" if v < 1 else "C3" for v in r]
        ax.barh(r.index, r.values, color=colors)
        ax.axvline(1, color="black", lw=1)
        ax.set_title(f"{metric} / GARCH(1,1) — h={h}")
        ax.set_xlabel("< 1 = mejor que el benchmark")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------------------ reporte
def auto_conclusions(metrics: pd.DataFrame, dm: pd.DataFrame, shap_top: dict[int, pd.DataFrame]) -> str:
    lines = []
    for h, g in metrics.groupby("horizonte"):
        g = g.set_index("modelo")
        best_q = g["QLIKE"].idxmin()
        best_m = g["MSFE"].idxmin()
        b = g.loc[BENCH]
        lines.append(f"### Horizonte {h} día(s)")
        lines.append(f"- Mejor por QLIKE: **{best_q}** "
                     f"({g.loc[best_q, 'QLIKE'] / b['QLIKE']:.2f}× el QLIKE de GARCH(1,1)).")
        lines.append(f"- Mejor por MSFE: **{best_m}** "
                     f"({g.loc[best_m, 'MSFE'] / b['MSFE']:.2f}× el MSFE de GARCH(1,1)).")
        sig = dm[(dm["horizonte"] == h) & (dm["conclusion"] == "mejor que benchmark")]
        if len(sig):
            items = ", ".join(sorted({f"{r.modelo} ({r.perdida})" for r in sig.itertuples()}))
            lines.append(f"- Superan a GARCH(1,1) con significancia al 5% (Diebold-Mariano): {items}.")
        else:
            lines.append("- Ningún modelo supera a GARCH(1,1) con significancia estadística al 5%.")
        worse = dm[(dm["horizonte"] == h) & (dm["conclusion"] == "peor que benchmark")]
        if len(worse):
            lines.append("- Significativamente peores que GARCH(1,1): "
                         + ", ".join(sorted(set(worse["modelo"]))) + ".")
        if "naive" in g.index:
            beat_naive = (g["QLIKE"] < g.loc["naive", "QLIKE"]).drop("naive")
            lines.append(f"- Superan al naive (QLIKE): {int(beat_naive.sum())} de {len(beat_naive)} modelos.")
        over = g["pct_sobreprediccion"]
        lines.append(f"- Sesgo: GARCH(1,1) sobre-predice {b['pct_sobreprediccion']:.0f}% de las veces; "
                     f"rango en los demás modelos {over.min():.0f}%–{over.max():.0f}%.")
        if h in shap_top:
            top = ", ".join(shap_top[h]["feature"].head(3))
            lines.append(f"- Variables más importantes (SHAP) del mejor modelo de ML: {top}.")
        lines.append("")
    return "\n".join(lines)


def write_report(out_path, split_info: dict, metrics: pd.DataFrame, ratios: pd.DataFrame,
                 dm: pd.DataFrame, garch_params: pd.DataFrame, shap_info: dict,
                 regimes: pd.DataFrame | None, hyper: pd.DataFrame, synthetic: bool = False):
    parts = ["# Resultados — Pronóstico de volatilidad",
             "",
             "> Generado automáticamente por `python run_pipeline.py`. No editar a mano: "
             "las conclusiones finales van en `docs/conclusiones.md`.",
             ""]
    if synthetic:
        parts += ["> ⚠️ **CORRIDA CON DATOS SINTÉTICOS** (prueba del pipeline). "
                  "Estos números NO son resultados del proyecto.", ""]
    parts += ["## Split (ADR-0003)", ""]
    parts += [f"- **{k}**: {v}" for k, v in split_info.items()] + [""]

    parts += ["## Métricas out-of-sample (#16)", "",
              "HV21 anualizada en %. QLIKE sobre varianza. MME_U castiga sub-predicción, MME_O sobre-predicción.", ""]
    for h, g in metrics.groupby("horizonte"):
        parts += [f"### h = {h}", "", _md(g.drop(columns="horizonte").sort_values("QLIKE")), ""]

    parts += ["## Ratios vs GARCH(1,1) expanding (ADR-0004)", "", "< 1 = mejor que el benchmark.", "",
              _md(ratios.reset_index(), ".3f"), "", "![ratios](figures/metricas_ratios_qlike.png)", ""]

    parts += ["## Diebold-Mariano vs GARCH(1,1) (#17)", "",
              "DM > 0 → el modelo tiene menor pérdida que el benchmark. Varianza HAC (Newey-West, h−1 rezagos) + corrección HLN.", "",
              _md(dm[["horizonte", "perdida", "modelo", "DM", "p_dos_colas", "conclusion"]], ".3f"), ""]

    parts += ["## Parámetros GARCH en el periodo de entrenamiento (#8, #9)", "",
              _md(garch_params.reset_index().rename(columns={"index": "parametro"}), ".4f"), "",
              "`gamma[1]` > 0 y significativo (t > 2) indica efecto asimétrico: los choques negativos "
              "aumentan más la volatilidad que los positivos.", ""]

    if regimes is not None and len(regimes):
        parts += ["## Quiebres estructurales (ICSS κ2, #6)", "", _md(regimes, ".2f"), "",
                  "![quiebres](figures/quiebres_principal.png)", ""]

    parts += ["## Hiperparámetros elegidos (#12)", "", _md(hyper, ".3f"), ""]

    for h, info in shap_info.items():
        parts += [f"## SHAP — mejor modelo de ML a h={h}: `{info['modelo']}` (#18)", "",
                  f"![shap](figures/shap_h{h}.png)", "", info["texto"], ""]

    parts += ["## Gráficos", ""]
    for h in sorted(metrics["horizonte"].unique()):
        parts += [f"![h{h}](figures/pronostico_h{h}.png)", ""]

    parts += ["## Conclusiones automáticas (borrador para #20)", "",
              auto_conclusions(metrics, dm, {h: i["tabla"] for h, i in shap_info.items()})]
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
