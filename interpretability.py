"""Issue #18 — Interpretabilidad con SHAP del mejor modelo de ML.

Se usa `shap.TreeExplainer` (exacto y rápido para XGBoost/Random Forest). Los modelos
pronostican log(HV_{t+h} / base), así que SHAP explica la CORRECCIÓN que el ML aplica sobre
el pronóstico base (EWMA; GARCH en el híbrido). Los valores se multiplican por el factor de
encogimiento elegido en validación, para que reflejen la contribución que realmente se usa.
Se interpretan como efectos multiplicativos: un SHAP de +0.10 ≈ +10% de volatilidad
pronosticada respecto a la base.
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.ml_models import VolModel


def shap_values(model: VolModel, X: pd.DataFrame) -> pd.DataFrame:
    import shap

    X = X[model.feature_names]
    explainer = shap.TreeExplainer(model.model)
    vals = explainer.shap_values(X)
    scale = model.shrink if model.target == "ratio" else 1.0
    return pd.DataFrame(vals * scale, index=X.index, columns=X.columns)


def importance_table(sv: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Importancia media |SHAP| + dirección del efecto (correlación valor-feature vs SHAP)."""
    rows = []
    for c in sv.columns:
        x = X[c].to_numpy(dtype=float)
        s = sv[c].to_numpy()
        corr = np.corrcoef(x, s)[0, 1] if np.std(x) > 0 and np.std(s) > 0 else np.nan
        rows.append({"feature": c, "mean_abs_shap": float(np.abs(s).mean()),
                     "efecto_pct_aprox": float(np.abs(s).mean() * 100),
                     "corr_valor_shap": float(corr)})
    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["direccion"] = np.where(out["corr_valor_shap"] > 0.3, "valor alto → más volatilidad",
                                np.where(out["corr_valor_shap"] < -0.3,
                                         "valor alto → menos volatilidad", "no monótona"))
    return out


def summary_plot(sv: pd.DataFrame, X: pd.DataFrame, title: str, path, max_display: int = 15):
    import shap

    plt.figure()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.summary_plot(sv.to_numpy(), X[sv.columns], max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")


def interpretation_text(imp: pd.DataFrame, top: int = 5) -> str:
    """Borrador de interpretación escrita de las variables más importantes."""
    from src.features.ml_features import feature_dictionary

    docs = feature_dictionary(imp["feature"].head(top)).set_index("feature")
    lines = []
    for i, row in imp.head(top).iterrows():
        d = docs.loc[row["feature"], "descripcion"] or row["feature"]
        lines.append(
            f"{i + 1}. **{row['feature']}** ({d}; grupo *{docs.loc[row['feature'], 'grupo']}*): "
            f"mueve el pronóstico en promedio ±{row['efecto_pct_aprox']:.1f}%; {row['direccion']}."
        )
    return "\n".join(lines)
