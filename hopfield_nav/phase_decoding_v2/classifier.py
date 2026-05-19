"""MLP-with-hidden-layer phase classifier used by Exp 2.

PCA on the hidden-layer activations of a trained classifier gives a
phase-relevant 2D embedding, which is what the user's spec asks for.

Pipeline:
  - StandardScaler on h (fit on train)
  - 2-layer MLP: Linear(in, hidden) -> ReLU -> Linear(hidden, 2)
  - Class-weighted CE loss to handle phase imbalance
  - Adam, ~30 epochs at batch_size 256
  - Returns (model, {"scaler": StandardScaler, "val_balanced_acc": ...})
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPhaseClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.fc1(x))
        logits = self.fc2(hidden)
        return logits, hidden


def _balanced_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


def train_mlp(
    h: np.ndarray,
    phase: np.ndarray,
    *,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 256,
    val_frac: float = 0.1,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str | torch.device | None = None,
    log_every: int = 5,
    log_label: str = "mlp",
) -> tuple[MLPPhaseClassifier, dict]:
    from sklearn.preprocessing import StandardScaler

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    h = np.asarray(h, dtype=np.float32)
    y = np.asarray(phase, dtype=np.int64).reshape(-1)
    if h.shape[0] != y.shape[0]:
        raise ValueError(f"h shape {h.shape} mismatched with phase {y.shape}")
    if len(np.unique(y)) < 2:
        raise ValueError("phase must contain both 0 and 1")
    print(
        f"[classifier:{log_label}] training MLP({h.shape[1]}->{hidden_dim}->2) "
        f"on {h.shape[0]} samples ({int((y==0).sum())} explore, "
        f"{int((y==1).sum())} exploit), epochs={epochs}, batch={batch_size}, "
        f"lr={lr}, device={device}",
        flush=True,
    )

    rng = np.random.RandomState(seed)
    perm = rng.permutation(h.shape[0])
    h, y = h[perm], y[perm]

    n_val = max(1, int(round(val_frac * h.shape[0])))
    h_val, y_val = h[:n_val], y[:n_val]
    h_tr, y_tr = h[n_val:], y[n_val:]

    scaler = StandardScaler().fit(h_tr)
    Xtr = scaler.transform(h_tr).astype(np.float32)
    Xv = scaler.transform(h_val).astype(np.float32)

    torch.manual_seed(seed)
    model = MLPPhaseClassifier(in_dim=h.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    counts = np.bincount(y_tr, minlength=2).astype(np.float32)
    class_weights = (counts.sum() / np.maximum(counts, 1.0))
    class_weights = class_weights / class_weights.mean()
    cw_t = torch.from_numpy(class_weights.astype(np.float32)).to(device)

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(y_tr).to(device)
    Xv_t = torch.from_numpy(Xv).to(device)
    yv_t = torch.from_numpy(y_val).to(device)

    n = Xtr_t.shape[0]
    history: list[dict] = []
    for ep in range(epochs):
        model.train()
        order = torch.randperm(n, device=device)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb, weight=cw_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * idx.numel()
        ep_loss /= max(n, 1)

        model.eval()
        with torch.no_grad():
            logits_v, _ = model(Xv_t)
            pred_v = logits_v.argmax(dim=-1).cpu().numpy()
        val_bal = _balanced_acc(y_val, pred_v)
        history.append({
            "epoch": ep,
            "train_loss": ep_loss,
            "val_balanced_acc": val_bal,
        })
        if log_every and ((ep + 1) % log_every == 0 or ep + 1 == epochs):
            print(
                f"[classifier:{log_label}] epoch {ep + 1}/{epochs} "
                f"train_loss={ep_loss:.4f} val_bal_acc={val_bal:.3f}",
                flush=True,
            )

    metrics = {
        "scaler": scaler,
        "history": history,
        "val_balanced_acc": history[-1]["val_balanced_acc"] if history else float("nan"),
        "hidden_dim": int(hidden_dim),
    }
    return model, metrics


@torch.no_grad()
def extract_hidden(
    model: MLPPhaseClassifier,
    h: np.ndarray,
    scaler,
    device: str | torch.device | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.eval()
    h = np.asarray(h, dtype=np.float32)
    Z = scaler.transform(h).astype(np.float32)
    out: list[np.ndarray] = []
    for i in range(0, Z.shape[0], batch_size):
        chunk = torch.from_numpy(Z[i:i + batch_size]).to(device)
        _, hidden = model(chunk)
        out.append(hidden.cpu().numpy().astype(np.float32))
    if not out:
        return np.zeros((0, model.fc1.out_features), dtype=np.float32)
    return np.concatenate(out, axis=0)
