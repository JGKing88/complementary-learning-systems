"""Associative-memory layers over the grid code: place book and pseudoinverses.

The six public functions here are the whole of what the live code used from
`cls/vectorhash/assoc_utils_np.py` -- 154 lines out of 958, extracted verbatim in
phase 7 together with the six private helpers they call. `VectorHash` builds its
scaffold from them; `analysis/scaffold_experiments/encoder_scaffold.py` builds
its comparison scaffolds the same way.

Nothing here was rewritten. The point of the move is that `cls/` can now be
deleted without taking these with it, not that they are improved.
"""
from __future__ import annotations

import numpy as np
import torch


def relu(x, thresh=0):
    return x * (x > thresh)


def nonlin(x, thresh=2.5):
    #return relu(x, 0)
    return relu(x-thresh, 0)


def _dev():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _to_torch(x, device=None, dtype=torch.float64):
    return torch.as_tensor(x, device=device or _dev(), dtype=dtype)


def _to_numpy(x_t):
    return x_t.detach().cpu().numpy()


def _ij_klj_to_kil_torch(G_t, T_t):
    """
    torch equivalent of einsum('ij,klj->kil', G, T) via GEMM.
    G_t: (I, J)
    T_t: (K, L, J)
    -> (K, I, L)
    """
    I, J = G_t.shape
    K, L, J2 = T_t.shape
    assert J == J2, "Dimension mismatch on J"

    T2 = T_t.reshape(K * L, J)          # (K*L, J)
    out = (T2 @ G_t.T).reshape(K, L, I) # (K, L, I)
    out = out.transpose(1, 2)           # (K, I, L)
    return out


def _ij_kli_to_klj_torch(G_t, T_t):
    """
    torch equivalent of einsum('ij,kli->klj' with G: (J, I), T: (K, L, I)) via GEMM.
    G_t: (J, I)
    T_t: (K, L, I)
    -> (K, L, J)
    """
    J, I = G_t.shape
    K, L, I2 = T_t.shape
    assert I == I2, "Dimension mismatch on I"

    T2 = T_t.reshape(K * L, I)          # (K*L, I)
    out = (T2 @ G_t.T).reshape(K, L, J) # (K, L, J)
    return out


def train_gcpc(pbook, gbook, Npatts):
    """
    NumPy in -> NumPy out, uses GPU if available.
    gbook: (I, J)
    pbook: (K, L, J)  or (L, J)
    Returns:
      if 3D pbook -> (K, I, L)
      if 2D pbook -> (I, L)
    Implements (1/Npatts) * einsum('ij,klj->kil', ...) or (1/Npatts)*einsum('ij,lj->il').
    """
    device = _dev()
    # Slice first to reduce transfers
    G_np = np.asarray(gbook)[:, :Npatts]   # (I, J’)
    if pbook.ndim == 3:
        P_np = np.asarray(pbook)[:, :, :Npatts]  # (K, L, J’)

        G_t = _to_torch(G_np, device)
        P_t = _to_torch(P_np, device)

        out_t = _ij_klj_to_kil_torch(G_t, P_t)   # (K, I, L)
        out_np = _to_numpy(out_t) / float(Npatts)
        return out_np
    else:
        P_np = np.asarray(pbook)[:, :Npatts]     # (L, J’)

        G_t = _to_torch(G_np, device)            # (I, J’)
        PT_t = _to_torch(P_np.T, device)         # (J’, L)
        out_t = G_t @ PT_t                       # (I, L)
        out_np = _to_numpy(out_t) / float(Npatts)
        return out_np


def train_pbook(Wpg, gbook):
    """
    NumPy in -> NumPy out, GPU-accelerated GEMM.
    Wpg:   (J, K)
    gbook: (K, L, M)
    Returns: (Wpg @ gbook.reshape(K, L*M)).reshape(J, L, M), applied in NumPy.
    """
    device = _dev()
    W_np = np.asarray(Wpg)
    G_np = np.asarray(gbook)

    J, K = W_np.shape
    K2, *rest = G_np.shape
    assert K == K2, "Inner dims must match: Wpg[:,K] vs gbook[K,...]"

    W_t = _to_torch(W_np, device)
    Gflat_t = _to_torch(G_np.reshape(K, -1), device)   # (K, L*M)

    P_t = W_t @ Gflat_t                                # (J, L*M)
    P_np = _to_numpy(P_t).reshape(J, *rest)            # back to NumPy for nonlin
    return P_np


def pseudotrain_Wsp(sbook, ca1book, Npatts):
    """
    NumPy in -> NumPy out, GPU-accelerated.
    sbook:   (I, J_full)
    ca1book: (K, L, J_full)  or (L, J_full)
    Returns:
      if 3D ca1book -> (K, I, L)  [einsum('ij, kjl -> kil')]
      if 2D ca1book -> (I, L)     [einsum('ij, jl  -> il')]
    Uses torch.linalg.pinv (batched) for the pseudoinverse.
    """
    device = _dev()
    S_np = np.asarray(sbook)[:, :Npatts]     # (I, J’)

    if ca1book.ndim == 3:
        C_np = np.asarray(ca1book)[:, :, :Npatts]   # (K, L, J’)
        C_t  = _to_torch(C_np, device)
        # pinv over last two dims: (K, L, J’) -> (K, J’, L)
        Cinv_t = torch.linalg.pinv(C_t)
        S_t = _to_torch(S_np, device)              # (I, J’)
        out_t = _ij_klj_to_kil_torch(S_t, Cinv_t)  # (K, I, L)
        return _to_numpy(out_t)

    else:
        C_np = np.asarray(ca1book)[:, :Npatts]     # (L, J’)
        C_t  = _to_torch(C_np, device)
        Cinv_t = torch.linalg.pinv(C_t)            # (J’, L)
        S_t = _to_torch(S_np, device)              # (I, J’)
        out_t = S_t @ Cinv_t                       # (I, L)
        return _to_numpy(out_t)


def pseudotrain_Wps(ca1book, sbook, Npatts):
    """
    NumPy in → NumPy out (GPU if available).
    Original intent:
      sbookinv = pinv(sbook[:, :Npatts])  # (J', I) with J'=Npatts
      3D: einsum('ij, kli -> klj', sbookinv[:Npatts,:], ca1book[:,:,:Npatts])
      2D: einsum('ij, li  -> lj', sbookinv[:Npatts,:], ca1book[:,:Npatts])
    """
    device = _dev()

    # S[:, :Npatts] has shape (I, I') with I' = Npatts
    S_np = np.asarray(sbook)[:, :Npatts]
    S_t  = _to_torch(S_np, device)
    S_inv_t = torch.linalg.pinv(S_t)          # (I', I)  ≡ (J', I)

    if ca1book.ndim == 3:
        # C: (K, L, I') so dims match the 'i' index in einsum
        C_np = np.asarray(ca1book)[:, :, :Npatts]
        C_t  = _to_torch(C_np, device)        # (K, L, I')
        out_t = _ij_kli_to_klj_torch(S_inv_t, C_t)  # (K, L, J')
        return _to_numpy(out_t)
    else:
        # C: (L, I')  and S_inv_t: (I', I)  → (L, I)
        C_np = np.asarray(ca1book)[:, :Npatts]
        C_t  = _to_torch(C_np, device)        # (L, I')
        out_t = C_t @ S_inv_t                 # (L, I)   <<< no transpose
        return _to_numpy(out_t)


def pseudotrain_Wgp(ca1book, gbook, Npatts):
    if len(ca1book.shape) == 3:
        ca1inv = np.linalg.pinv(ca1book[:, :, :Npatts])
        return np.einsum('ij, ljk -> lik', gbook[:,:Npatts], ca1inv[:,:Npatts,:]) 
    else:
        ca1inv = np.linalg.pinv(ca1book[:, :Npatts])
        return np.einsum('ij, jk -> ik', gbook[:,:Npatts], ca1inv[:Npatts,:])


__all__ = [
    "nonlin",
    "pseudotrain_Wgp",
    "pseudotrain_Wps",
    "pseudotrain_Wsp",
    "relu",
    "train_gcpc",
    "train_pbook",
]
