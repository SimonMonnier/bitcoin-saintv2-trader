"""Export SAINTv2 PyTorch → ONNX pour MT5 Strategy Tester.

Produit (pour AGENT="wf1") :
  - saintv2_wf1.onnx           (modèle)
  - norm_stats_mean.bin        (16 doubles, mean des features)
  - norm_stats_std.bin         (16 doubles, std des features)

Ces 3 fichiers doivent être copiés dans :
  <MT5 Data Folder>/MQL5/Files/

Le chemin du Data Folder est accessible via MT5 → Fichier → Ouvrir le dossier de données.
"""

# Fix OpenMP duplicate (DOIT être avant import torch/numpy)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# PATCH MultiheadAttention pour export ONNX
# Le fast-path PyTorch (`_native_multi_head_attention`) n'est PAS supporté
# en ONNX. On remplace .forward par une implémentation pure-torch.
# ============================================================
def manual_mha_forward(mha, query, key, value, **kwargs):
    """Reproduit nn.MultiheadAttention(batch_first=True) en pur torch."""
    B, T, D = query.shape
    h  = mha.num_heads
    dh = D // h

    # Projection in : poids (3D, D) en bloc
    proj = F.linear(query, mha.in_proj_weight, mha.in_proj_bias)  # (B, T, 3D)
    Q, K, V = proj.chunk(3, dim=-1)

    # (B, h, T, dh)
    Q = Q.reshape(B, T, h, dh).transpose(1, 2)
    K = K.reshape(B, T, h, dh).transpose(1, 2)
    V = V.reshape(B, T, h, dh).transpose(1, 2)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / (dh ** 0.5)
    attn   = torch.softmax(scores, dim=-1)
    if mha.dropout > 0 and mha.training:
        attn = F.dropout(attn, p=mha.dropout)
    out = torch.matmul(attn, V)                           # (B, h, T, dh)
    out = out.transpose(1, 2).contiguous().reshape(B, T, D)
    out = F.linear(out, mha.out_proj.weight, mha.out_proj.bias)
    return out, None  # MHA renvoie (output, attn_weights)


def patch_mha(module: nn.Module):
    """Remplace forward de chaque nn.MultiheadAttention par manual_mha_forward."""
    patched = 0
    for m in module.modules():
        if isinstance(m, nn.MultiheadAttention):
            # Bind la méthode patchée à l'instance
            m.forward = manual_mha_forward.__get__(m, nn.MultiheadAttention)
            patched += 1
    return patched

# Import des classes du modèle depuis training.py
from training import (
    SAINTPolicySingleHead,
    OBS_N_FEATURES,
    N_ACTIONS,
)

# Le nom de sortie est DÉRIVÉ du checkpoint : l'ancienne version exportait le
# .pth wf2 sous le nom "saintv2_wf3.onnx", donc l'EA "WF3" tournait sur des
# poids WF2. Ne jamais dissocier les deux.
AGENT      = "wf1"
CHECKPOINT = f"bestprofit_saintv2_loup_duel_{AGENT}_both_{AGENT}.pth"
ONNX_OUT   = f"saintv2_{AGENT}.onnx"
NORM_STATS = "norm_stats_ohlc_indics.npz"
# LOOKBACK LU DEPUIS LA CONFIG, jamais recopie.
#
# Il valait 54 en dur alors que l'entrainement produit desormais des poids sur
# 25 pas de temps. Un modele exporte aurait attendu une entree de forme (1, 54,
# 34) pour des poids entraines en (1, 25, 34) : desaccord silencieux au
# deploiement, decouvert au premier tick en production.
#
# La valeur ne doit exister qu'a UN endroit. Toute duplication finit par
# diverger — c'est deja arrive au plancher d'ATR, au spread et au jeu de
# features au cours de ce projet.
from training import PPOConfig as _PPOConfig
LOOKBACK   = _PPOConfig().lookback

# ============================================================
# 1) Charge le modèle PyTorch
# ============================================================
device = torch.device("cpu")
model = SAINTPolicySingleHead(
    n_features=OBS_N_FEATURES,
    d_model=80,
    num_blocks=2,
    heads=4,
    dropout=0.05,
    ff_mult=2,
    max_len=LOOKBACK,
    n_actions=N_ACTIONS,
).to(device)
state = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(state)
model.eval()
print(f"[OK] Modèle chargé : {CHECKPOINT}")

# ============================================================
# 2) Export ONNX
# ============================================================
dummy_input = torch.randn(1, LOOKBACK, OBS_N_FEATURES, dtype=torch.float32)

# Patch les MultiheadAttention pour éviter le fast-path non-exportable
n_patched = patch_mha(model)
print(f"[OK] {n_patched} MultiheadAttention patchés pour export ONNX")

# Sanity check : le modèle patché produit-il la même chose ?
with torch.no_grad():
    out_patched = model(dummy_input)
print(f"[OK] Forward patché OK : logits.shape={tuple(out_patched[0].shape)}")

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUT,
        input_names=["obs"],
        output_names=["logits", "value"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "obs":    {0: "batch"},
            "logits": {0: "batch"},
            "value":  {0: "batch"},
        },
    )
print(f"[OK] ONNX exporté    : {ONNX_OUT}")

# Vérif rapide ONNX (optionnel — nécessite onnxruntime)
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_OUT)
    test = np.random.randn(1, LOOKBACK, OBS_N_FEATURES).astype(np.float32)
    out = sess.run(None, {"obs": test})
    print(f"[OK] ONNX runtime OK : logits.shape={out[0].shape}, value.shape={out[1].shape}")
    # Compare avec PyTorch
    with torch.no_grad():
        ref_logits, ref_value = model(torch.from_numpy(test))
    diff = np.abs(out[0] - ref_logits.numpy()).max()
    print(f"[OK] Diff max logits torch vs onnx : {diff:.2e}")
except ImportError:
    print("[INFO] onnxruntime non installé, skip vérif (pip install onnxruntime)")

# ============================================================
# 3) Export des stats de normalisation
# ============================================================
from saint_core import FEATURE_COLS as _FEATS

from saint_core import load_model_norm_stats
stats = load_model_norm_stats(CHECKPOINT)
mean = stats["mean"].astype(np.float64)
std  = stats["std"].astype(np.float64)
# Compté depuis saint_core, jamais en dur : un nombre figé ici avait survécu à
# deux changements de jeu de features sans broncher.
assert len(mean) == len(_FEATS), (
    f"{len(_FEATS)} features dans saint_core, {len(mean)} dans {NORM_STATS}. "
    f"Supprimer {NORM_STATS} et relancer le training pour les recalculer."
)

with open("norm_stats_mean.bin", "wb") as f:
    for v in mean:
        f.write(struct.pack("<d", float(v)))  # little-endian double (MQL5 default)
with open("norm_stats_std.bin", "wb") as f:
    for v in std:
        f.write(struct.pack("<d", float(v)))
print(f"[OK] Stats exportées : norm_stats_mean.bin / norm_stats_std.bin")

# Aussi en CSV pour debug humain
with open("norm_stats.csv", "w", encoding="utf-8") as f:
    f.write("idx,name,mean,std\n")
    # Source de vérité unique : évite que ce CSV de debug diverge du modèle.
    for i, (n, m, s) in enumerate(zip(_FEATS, mean, std)):
        f.write(f"{i},{n},{m:.10f},{s:.10f}\n")
print(f"[OK] CSV debug       : norm_stats.csv")

print()
print("=" * 60)
print("PROCHAINE ÉTAPE — copie ces fichiers dans :")
print("  <Terminal>/Common/Files/    (REQUIS pour le Strategy Tester)")
print("Fichiers à copier :")
print(f"  - {ONNX_OUT}")
print(f"  - norm_stats_mean.bin      ({len(mean)} doubles)")
print(f"  - norm_stats_std.bin       ({len(std)} doubles)")
print(f"  - binance_BTCUSD.bin       -> python export_binance_for_mql5.py")
print("    (le déposera lui-même au bon endroit)")
print("=" * 60)
