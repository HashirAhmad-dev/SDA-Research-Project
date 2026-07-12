"""
lstm_infer.py
=============
Torch-free forward pass for the trained BehaviouralLSTM encoder.

The model is trained with PyTorch (`evaluation/train_and_evaluate.py`), but
inference is a single-layer `nn.LSTM(6, 128)` unrolled over 24 steps followed by
taking the final hidden state - about twenty lines of numpy. Serving it that way
keeps the deployed dashboard off torch entirely, which matters: the Streamlit
Cloud container cannot hold a torch install alongside pandas/plotly/sklearn.

These are the *same trained weights*, not an approximation. The training script
exports them to `evaluation/artifacts/lstm_weights.npz`, and its
`verify_numpy_lstm()` checks this implementation against torch's `encode()`:
max absolute difference 2.4e-7 over random windows, i.e. float32 noise.

PyTorch's LSTM gate layout for weight_ih_l0 / weight_hh_l0 / bias_* is a single
(4H, .) stack ordered [input, forget, cell, output].
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Branchless and overflow-safe: exp() of a positive argument is the risk.
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class NumpyLSTMEncoder:
    """Final-hidden-state encoder for a single-layer, uni-directional LSTM."""

    def __init__(self, w_ih: np.ndarray, w_hh: np.ndarray,
                 b_ih: np.ndarray, b_hh: np.ndarray,
                 n_features: int, hidden: int, seq_len: int):
        self.w_ih = np.ascontiguousarray(w_ih, dtype=np.float32)   # (4H, F)
        self.w_hh = np.ascontiguousarray(w_hh, dtype=np.float32)   # (4H, H)
        self.b = np.ascontiguousarray(b_ih + b_hh, dtype=np.float32)  # (4H,)
        self.n_features = int(n_features)
        self.hidden = int(hidden)
        self.seq_len = int(seq_len)

    @classmethod
    def from_npz(cls, path: Path) -> "NumpyLSTMEncoder":
        z = np.load(path)
        return cls(
            w_ih=z["weight_ih_l0"], w_hh=z["weight_hh_l0"],
            b_ih=z["bias_ih_l0"], b_hh=z["bias_hh_l0"],
            n_features=int(z["n_features"]), hidden=int(z["hidden"]),
            seq_len=int(z["seq_len"]),
        )

    @classmethod
    def from_state_dict(cls, sd: Dict[str, Any], n_features: int,
                        hidden: int, seq_len: int) -> "NumpyLSTMEncoder":
        """Accepts a torch state_dict whose tensors expose `.numpy()`."""
        def arr(key: str) -> np.ndarray:
            t = sd[key]
            return np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t)

        return cls(
            w_ih=arr("encoder.weight_ih_l0"), w_hh=arr("encoder.weight_hh_l0"),
            b_ih=arr("encoder.bias_ih_l0"), b_hh=arr("encoder.bias_hh_l0"),
            n_features=n_features, hidden=hidden, seq_len=seq_len,
        )

    def encode(self, window: np.ndarray) -> np.ndarray:
        """window: (seq_len, n_features) -> final hidden state (hidden,)."""
        x = np.asarray(window, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"expected a (T, {self.n_features}) window, got {x.shape}")

        h = np.zeros(self.hidden, dtype=np.float32)
        c = np.zeros(self.hidden, dtype=np.float32)
        H = self.hidden
        for t in range(x.shape[0]):
            g = self.w_ih @ x[t] + self.w_hh @ h + self.b        # (4H,)
            i = _sigmoid(g[0:H])
            f = _sigmoid(g[H:2 * H])
            gg = np.tanh(g[2 * H:3 * H])
            o = _sigmoid(g[3 * H:4 * H])
            c = f * c + i * gg
            h = o * np.tanh(c)
        return h
