"""Issue #13 (opcional) — LSTM simple para pronóstico de volatilidad.

Requiere PyTorch (`pip install torch`). Si no está instalado, el pipeline lo omite y lo
avisa en el log; el resto de los modelos no se ve afectado.

Arquitectura mínima (ADR-0006):
  entrada  : secuencia de los últimos `seq_len`=21 días de las features (estandarizadas con
             media/desv. del subtrain, sin ver validación ni test)
  LSTM     : 1 capa, 32 unidades
  dropout  : 0.2
  salida   : Linear(32 → 1) que predice log(HV_{t+h})
  pérdida  : MSE sobre log(HV) · optimizador Adam (lr=1e-3) · batch 64
  early stopping: paciencia de 10 épocas sobre la pérdida de validación (máx. 200 épocas)
Mismo split, mismos orígenes de test y misma purga que el resto (ADR-0003).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.evaluation.backtest import Split
from src.features.volatility import target_hv

log = logging.getLogger(__name__)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _sequences(X: np.ndarray, rows: np.ndarray, seq_len: int) -> np.ndarray:
    """Para cada posición p en `rows`, la ventana X[p-seq_len+1 : p+1]."""
    return np.stack([X[p - seq_len + 1: p + 1] for p in rows])


def lstm_forecast(X: pd.DataFrame, returns: pd.Series, split: Split, horizon: int,
                  seq_len: int = 21, hidden: int = 32, dropout: float = 0.2,
                  lr: float = 1e-3, batch_size: int = 64, max_epochs: int = 200,
                  patience: int = 10, seed: int = config.SEED) -> pd.Series:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    y = np.log(target_hv(returns, horizon).reindex(X.index))
    Xf = X.ffill()
    valid = Xf.notna().all(axis=1).to_numpy()
    first_valid = int(np.argmax(valid)) + seq_len - 1  # posición con historia completa

    pos = pd.Series(np.arange(len(X)), index=X.index)
    def rows_for(dates):
        p = pos.reindex(dates).dropna().astype(int).to_numpy()
        return p[(p >= first_valid) & np.isfinite(y.to_numpy()[p])]

    sub = rows_for(split.fit_rows(horizon, "subtrain"))
    val = rows_for(split.fit_rows(horizon, "val"))
    tr = rows_for(split.fit_rows(horizon, "train"))
    te = pos.reindex(split.test_origins).astype(int).to_numpy()

    mu = Xf.iloc[sub].mean()
    sd = Xf.iloc[sub].std().replace(0, 1.0)
    Z = ((Xf - mu) / sd).fillna(0.0).to_numpy(dtype=np.float32)
    yv = y.to_numpy(dtype=np.float32)

    class Net(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x):
            h, _ = self.lstm(x)
            return self.out(self.drop(h[:, -1, :])).squeeze(-1)

    def train(rows_fit, rows_val, n_epochs=None):
        net = Net(Z.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        Xt = torch.from_numpy(_sequences(Z, rows_fit, seq_len))
        yt = torch.from_numpy(yv[rows_fit])
        if rows_val is not None:
            Xv = torch.from_numpy(_sequences(Z, rows_val, seq_len))
            yv_t = torch.from_numpy(yv[rows_val])
        best, best_ep, wait, best_state = np.inf, 0, 0, None
        for ep in range(n_epochs or max_epochs):
            net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), batch_size):
                idx = perm[i: i + batch_size]
                opt.zero_grad()
                loss = loss_fn(net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            if rows_val is None:
                continue
            net.eval()
            with torch.no_grad():
                vl = loss_fn(net(Xv), yv_t).item()
            if vl < best - 1e-6:
                best, best_ep, wait = vl, ep + 1, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_ep

    _, n_epochs = train(sub, val)                  # 1) épocas óptimas con early stopping
    net, _ = train(tr, None, n_epochs=max(n_epochs, 1))  # 2) re-entrena en todo el train
    net.eval()
    with torch.no_grad():
        pred_tr = net(torch.from_numpy(_sequences(Z, tr, seq_len))).numpy()
        smear = float(np.mean(np.exp(yv[tr] - pred_tr)))
        pred = net(torch.from_numpy(_sequences(Z, te, seq_len))).numpy()
    log.info("LSTM h=%d: %d épocas", horizon, n_epochs)
    return pd.Series(np.exp(pred) * smear, index=split.test_origins, name=f"lstm_h{horizon}")
