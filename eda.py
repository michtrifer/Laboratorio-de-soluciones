"""Issue #5 — Análisis exploratorio de retornos.

`descriptive_table` reproduce el formato de la Tabla 1 de Chung, Espinoza & Quispe (2025):
estadísticos descriptivos, asimetría/curtosis, normalidad (Jarque-Bera y Shapiro-Wilk),
autocorrelación de retornos y retornos al cuadrado (Ljung-Box) y efectos ARCH (ARCH-LM).
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src import config
from src.features.volatility import historical_volatility

SHAPIRO_MAX_N = 5000  # scipy: el p-value de Shapiro-Wilk no es exacto con N > 5000


def _stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def return_stats(r: pd.Series, lags: int = 10) -> dict:
    """Estadísticos y pruebas para una serie de retornos (en %)."""
    r = r.dropna()
    jb = stats.jarque_bera(r)
    sw = stats.shapiro(r.iloc[-SHAPIRO_MAX_N:])
    lb_r = acorr_ljungbox(r, lags=[lags], return_df=True).iloc[0]
    lb_r2 = acorr_ljungbox(r ** 2, lags=[lags], return_df=True).iloc[0]
    with warnings.catch_warnings():  # statsmodels >= 0.15 avisa de un cambio de API futuro
        warnings.simplefilter("ignore", FutureWarning)
        arch_lm = het_arch(r, nlags=lags)
    return {
        "Obs": len(r),
        "Media": r.mean(),
        "Desv. est.": r.std(),
        "Mín": r.min(),
        "Máx": r.max(),
        "Asimetría": stats.skew(r),
        "Curtosis (exceso)": stats.kurtosis(r),
        "Jarque-Bera": jb.statistic, "JB p": jb.pvalue,
        "Shapiro-Wilk W": sw.statistic, "SW p": sw.pvalue,
        f"Q({lags}) r": lb_r["lb_stat"], f"Q({lags}) r p": lb_r["lb_pvalue"],
        f"Q²({lags}) r²": lb_r2["lb_stat"], f"Q²({lags}) r² p": lb_r2["lb_pvalue"],
        f"ARCH-LM({lags})": arch_lm[0], "ARCH-LM p": arch_lm[1],
    }


def descriptive_table(returns: dict[str, pd.Series], lags: int = 10) -> pd.DataFrame:
    """Tabla (filas = estadístico, columnas = activo) con valores numéricos."""
    return pd.DataFrame({name: return_stats(r, lags) for name, r in returns.items()})


def format_table(tab: pd.DataFrame) -> pd.DataFrame:
    """Versión para reporte: estadísticos de prueba con estrellas (* 10%, ** 5%, *** 1%)."""
    pvals = {row: row + " p" for row in tab.index if row + " p" in tab.index}
    pvals.update({"Jarque-Bera": "JB p", "Shapiro-Wilk W": "SW p"})
    pvals.update({row: "ARCH-LM p" for row in tab.index if row.startswith("ARCH-LM(")})
    p_rows = set(pvals.values())
    out = {}
    for col in tab.columns:
        vals = {}
        for row in tab.index:
            if row in p_rows:
                continue
            v = tab.loc[row, col]
            if row in pvals:
                vals[row] = f"{v:,.3f}{_stars(tab.loc[pvals[row], col])}"
            elif row == "Obs":
                vals[row] = f"{int(v):,}"
            else:
                vals[row] = f"{v:,.4f}"
        out[col] = vals
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------ gráficos
def plot_price_returns(price: pd.Series, returns: pd.Series, title: str = ""):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(price.index, price, lw=0.8)
    axes[0].set_ylabel("Precio")
    axes[1].plot(returns.index, returns, lw=0.5)
    axes[1].set_ylabel("Log-retorno (%)")
    hv = historical_volatility(returns)
    axes[2].plot(hv.index, hv, lw=0.8, color="C3")
    axes[2].set_ylabel(f"HV{config.HV_WINDOW} anualizada (%)")
    axes[0].set_title(title)
    fig.tight_layout()
    return fig


def plot_distribution(returns: pd.Series, title: str = ""):
    r = returns.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(r, bins=100, density=True, alpha=0.6, label="Empírica")
    x = np.linspace(r.min(), r.max(), 400)
    axes[0].plot(x, stats.norm.pdf(x, r.mean(), r.std()), "r", label="Normal")
    axes[0].set_title("Histograma vs normal")
    axes[0].legend()
    stats.probplot(r, dist="norm", plot=axes[1])
    axes[1].set_title("QQ-plot")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_acf_returns(returns: pd.Series, lags: int = 40, title: str = ""):
    r = returns.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(r, lags=lags, ax=axes[0], zero=False, title="ACF retornos")
    plot_acf(r ** 2, lags=lags, ax=axes[1], zero=False, title="ACF retornos²")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_breaks(returns: pd.Series, breaks: pd.DatetimeIndex, title: str = ""):
    """Retornos con quiebres marcados + volatilidad por régimen (±2σ)."""
    from src.features.regimes import regime_table

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(returns.index, returns, lw=0.5, color="0.4", label="Log-retorno (%)")
    for d in breaks:
        ax.axvline(d, color="C3", ls="--", lw=1)
    regimes = regime_table(returns, breaks)
    for _, row in regimes.iterrows():
        sd = row["vol_anualizada"] / np.sqrt(config.TRADING_DAYS)
        ix = pd.to_datetime([row["inicio"], row["fin"]])
        ax.plot(ix, [2 * sd] * 2, color="C0", lw=2)
        ax.plot(ix, [-2 * sd] * 2, color="C0", lw=2)
    ax.set_title(title or f"Quiebres estructurales en varianza (ICSS): {len(breaks)} detectados")
    ax.set_ylabel("%")
    ax.legend(["Log-retorno", "Quiebre", "±2σ del régimen"], loc="upper left")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------ runner
def run_eda(processed: dict[str, pd.DataFrame], tables_dir=config.TABLES_DIR,
            figures_dir=config.FIGURES_DIR, method: str = "kappa2") -> dict:
    """Corre #5 y #6 para cada activo y guarda tablas y figuras. Regresa los resultados."""
    from src.features.regimes import break_dates, regime_table
    from src.utils.io import ensure_dir

    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    returns = {name: df["log_return"] for name, df in processed.items()}
    tab = descriptive_table(returns)
    tab.to_csv(tables_dir / "eda_tabla1_numerica.csv")
    format_table(tab).to_csv(tables_dir / "eda_tabla1.csv")

    breaks, regimes = {}, {}
    for name, df in processed.items():
        r = df["log_return"]
        for fig, fname in [
            (plot_price_returns(df["price"], r, name.upper()), f"eda_{name}_serie.png"),
            (plot_distribution(r, name.upper()), f"eda_{name}_distribucion.png"),
            (plot_acf_returns(r, title=name.upper()), f"eda_{name}_acf.png"),
        ]:
            fig.savefig(figures_dir / fname, dpi=120)
            plt.close(fig)
        b = break_dates(r, method=method)
        breaks[name] = b
        regimes[name] = regime_table(r, b)
        regimes[name].to_csv(tables_dir / f"quiebres_{name}.csv", index=False)
        fig = plot_breaks(r, b, f"{name.upper()} — quiebres en varianza (ICSS-{method}): {len(b)}")
        fig.savefig(figures_dir / f"quiebres_{name}.png", dpi=120)
        plt.close(fig)
    return {"tabla1": tab, "breaks": breaks, "regimes": regimes}
