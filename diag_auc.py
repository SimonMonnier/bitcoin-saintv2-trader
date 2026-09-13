"""PPO a-t-il conserve ou detruit la capacite de CLASSEMENT du modele ?

LA QUESTION. Sur la fenetre de validation, une simple regression logistique sur
les memes 10 colonnes atteint 0.6072 d'AUC. Le modele entraine, lui, affiche une
etendue de conviction de 0.92 — il a des avis tres tranches — mais son winrate
ne s'ameliore PAS quand on resserre le filtre de 44 % a 23 %. Les deux faits ne
peuvent coexister que si ses convictions ne correspondent a rien.

On mesure donc son AUC directement, contre l'etiquette de barriere triple, et on
la compare a celle de la sonde.

PRECEDENT SUR L'OR : PPO y avait fait passer l'AUC de 0.5047 a 0.4848, donc SOUS
le hasard — il avait inverse le classement. C'est ce qu'on cherche a savoir ici.

    python diag_auc.py
"""

import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from saint_core import (FEATURE_COLS, N_POS_FEATURES, MASK_VALUE,
                        ATR_PLANCHER_FRAC, load_norm_stats, safe_normalize,
                        build_policy, get_device, build_mask_from_pos_scalar)

CACHE = "data_cache_BTCUSD_20221215.pkl"
CHK = "bestprofit_saintv2_loup_duel_wf1_both_wf1.pth"
LOOKBACK = 54
SL_MULT, RR = 2.0, 1.4
MAX_HOLD = 240
PAS = 10
BATCH = 512
SPREAD_BPS, SLIP_S_BPS, SLIP_E_BPS, BRUIT_BPS = 2.61, 2.0, 1.0, 0.6


class _Cfg:
    force_cpu = False


def barrieres(hi, lo, cl, atr, idx, sens, demi_sp, slip_e, slip_s):
    n = len(cl)
    dist = SL_MULT * atr[idx]
    entree = cl[idx] + sens * (demi_sp[idx] + slip_e[idx])
    tp, sl = entree + sens * RR * dist, entree - sens * dist
    sortie = np.full(len(idx), np.nan)
    motif = np.zeros(len(idx), np.int8)
    ouvert = np.ones(len(idx), bool)
    for h in range(1, MAX_HOLD + 1):
        j = idx + h
        valide = ouvert & (j < n)
        if not valide.any():
            break
        jj = j[valide]
        h_e = hi[jj] * (1.0 + BRUIT_BPS / 1e4)
        l_e = lo[jj] * (1.0 - BRUIT_BPS / 1e4)
        if sens > 0:
            t_sl, t_tp = l_e <= sl[valide], h_e >= tp[valide]
        else:
            t_sl, t_tp = h_e >= sl[valide], l_e <= tp[valide]
        pos = np.where(valide)[0]
        k_sl, k_tp = pos[t_sl], pos[t_tp & ~t_sl]
        sortie[k_sl], motif[k_sl] = sl[k_sl], -1
        sortie[k_tp], motif[k_tp] = tp[k_tp], 1
        ouvert[k_sl] = ouvert[k_tp] = False
    reste = np.where(ouvert)[0]
    if len(reste):
        sortie[reste] = cl[np.minimum(idx[reste] + MAX_HOLD, n - 1)]
    contre = motif <= 0
    sortie = sortie + sens * np.where(contre, -slip_s[idx], slip_s[idx])
    return (sens * (sortie - entree) - demi_sp[idx]) / dist


def main() -> int:
    df = pd.read_pickle(CACHE)
    n = len(df)
    hi, lo, cl = (df[c].to_numpy(np.float64) for c in ("high", "low", "close"))
    atr = np.maximum(df["atr_14"].to_numpy(np.float64), ATR_PLANCHER_FRAC * cl)
    demi_sp = (SPREAD_BPS / 1e4) * cl / 2.0
    slip_e = (SLIP_E_BPS / 1e4) * cl
    slip_s = (SLIP_S_BPS / 1e4) * cl

    X = safe_normalize(df[FEATURE_COLS].to_numpy(np.float32),
                       load_norm_stats(), clip_sigma=5.0).astype(np.float32)
    vues = np.lib.stride_tricks.sliding_window_view(X, LOOKBACK, axis=0)

    a_va, b_va = int(n * 0.55), int(n * 0.70)
    i_tr = np.arange(LOOKBACK, a_va - MAX_HOLD, PAS)
    i_va = np.arange(a_va, b_va - MAX_HOLD, PAS)
    print(f"{n:,} bougies | train {len(i_tr):,} | val {len(i_va):,}\n")

    r_tr = [barrieres(hi, lo, cl, atr, i_tr, s, demi_sp, slip_e, slip_s) for s in (1, -1)]
    r_va = [barrieres(hi, lo, cl, atr, i_va, s, demi_sp, slip_e, slip_s) for s in (1, -1)]

    # --- sonde logistique, une bougie ---
    print("SONDE LOGISTIQUE (une bougie, les memes 10 colonnes)")
    auc_sonde = []
    for k in (0, 1):
        lr = LogisticRegression(max_iter=500, n_jobs=-1)
        lr.fit(X[i_tr], r_tr[k] > 0)
        p = lr.predict_proba(X[i_va])[:, 1]
        a = roc_auc_score(r_va[k] > 0, p)
        auc_sonde.append(a)
        print(f"  {'BUY ' if k == 0 else 'SELL'} : AUC {a:.4f}")
    print(f"  moyenne : {np.mean(auc_sonde):.4f}\n")

    # --- modele entraine ---
    dev = get_device(_Cfg())
    pol = build_policy(dev, lookback=LOOKBACK)
    pol.load_state_dict(torch.load(CHK, map_location=dev, weights_only=True))
    pol.eval()
    mask = build_mask_from_pos_scalar(0, dev, "both")

    pb, ps = [], []
    with torch.no_grad():
        for d in range(0, len(i_va), BATCH):
            lot = i_va[d:d + BATCH]
            f = np.transpose(vues[lot - LOOKBACK], (0, 2, 1))
            pos = np.zeros((len(lot), LOOKBACK, N_POS_FEATURES), np.float32)
            pos[:, :, 3] = 1.0
            obs = np.concatenate([f, pos], axis=2).astype(np.float32)
            lg, _ = pol(torch.as_tensor(obs, device=dev))
            p = torch.softmax(
                lg.masked_fill(~mask.unsqueeze(0), MASK_VALUE), -1).cpu().numpy()
            pb.append(p[:, 0])
            ps.append(p[:, 1])
    pb, ps = np.concatenate(pb), np.concatenate(ps)

    print(f"MODELE ENTRAINE ({CHK})")
    auc_mod = []
    for k, p in ((0, pb), (1, ps)):
        a = roc_auc_score(r_va[k] > 0, p)
        auc_mod.append(a)
        print(f"  {'BUY ' if k == 0 else 'SELL'} : AUC {a:.4f}   "
              f"conviction {p.min():.3f} a {p.max():.3f}, "
              f"mediane {np.median(p):.3f}")
    print(f"  moyenne : {np.mean(auc_mod):.4f}\n")

    # --- ce que donne la selection, par tranche de conviction ---
    print("ESPERANCE PAR TRANCHE DE CONVICTION (modele entraine)")
    print(f"{'tranche':>12} {'BUY E[R]':>10} {'BUY WR':>8} "
          f"{'SELL E[R]':>10} {'SELL WR':>8}")
    print("-" * 52)
    for lo_q, hi_q in ((0.0, 0.25), (0.25, 0.50), (0.50, 0.75),
                       (0.75, 0.90), (0.90, 0.99), (0.99, 1.0)):
        ligne = f"{f'{100*lo_q:.0f}-{100*hi_q:.0f}%':>12}"
        for k, p in ((0, pb), (1, ps)):
            a, b = np.quantile(p, lo_q), np.quantile(p, hi_q)
            m = (p >= a) & (p <= b)
            if m.sum() < 30:
                ligne += f" {'--':>10} {'--':>8}"
            else:
                ligne += f" {r_va[k][m].mean():>+10.4f} {100*(r_va[k][m] > 0).mean():>7.1f}%"
        print(ligne)

    print()
    print("Si l'AUC du modele est proche de 0.50, ses convictions ne portent")
    print("aucune information. Si elle est SOUS 0.50, PPO a inverse le")
    print("classement — c'est ce qui s'etait produit sur l'or (0.5047 -> 0.4848).")
    print("Et l'esperance doit CROITRE avec la tranche de conviction : c'est la")
    print("seule chose qui justifierait de resserrer le filtre.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
