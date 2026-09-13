"""Pré-entraînement SUPERVISÉ du réseau sur l'issue réelle des trades.

POURQUOI.
Mesuré : après deux mises à jour de l'actor, PPO fait passer l'AUC de conviction
de 0.5047 à 0.4848 — sous le hasard. Le winrate décroît avec la conviction au
lieu de croître. La cause est structurelle : chaque état n'est visité qu'une
fois par epoch, avec UNE seule réalisation d'un trade dont l'issue est aléatoire
à 35 %. L'avantage estimé est dominé par le tirage, pas par la qualité de
l'état, et PPO ajuste le bruit.

Or le même signal est parfaitement accessible en supervisé : l'étiquette « le TP
est-il touché avant le SL ? » est EXACTE, disponible sur 1.9 M bougies, et une
simple régression logistique en tire AUC 0.5442 là où PPO tombe à 0.4848.

On entraîne donc les logits 0 (BUY) et 1 (SELL) de l'actor en binaire sur ces
deux étiquettes, avec le MÊME tronc et la MÊME architecture que la policy. Le
checkpoint produit se recharge tel quel dans PPO, qui démarre alors avec un
réseau sachant déjà classer les occasions.

    python pretrain_supervised.py
"""

import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import MetaTrader5 as mt5
from sklearn.metrics import roc_auc_score

from saint_core import (FEATURE_COLS, N_POS_FEATURES, merge_m1_h1,
                        charge_source_externe, SOURCE_EXT_NOM, load_norm_stats, safe_normalize,
                        build_policy, get_device)

LOOKBACK = 25
SL_MULT, RR = 5.0, 2.0        # doivent rester alignés sur PPOConfig
MAX_HOLD = 240
DEB = datetime(2018, 9, 15)
PAS = 3                        # 1 fenêtre sur 3 : les fenêtres voisines sont
                               # quasi identiques, les garder toutes n'ajoute
                               # que du temps de calcul
BATCH = 512
EPOCHS = 25
LR = 3e-4
PATIENCE = 4
OUT = "pretrain_saintv2.pth"


class _Cfg:
    force_cpu = False


def barrieres(hi, lo, cl, atr, spread, idx, sens, n):
    """1 si le TP est touché avant le SL, 0 sinon, -1 si non résolu.

    Une bougie qui touche les deux est comptée PERDANTE : on ignore l'ordre
    intra-minute, autant que ce soit dans le sens défavorable.
    """
    a = atr[idx]
    px = cl[idx] + sens * spread / 2.0
    tp = px + sens * (RR * SL_MULT * a) + sens * spread / 2.0
    sl = px - sens * (SL_MULT * a) + sens * spread / 2.0
    res = np.full(len(idx), -1, np.int8)
    vivant = np.ones(len(idx), bool)
    for h in range(1, MAX_HOLD + 1):
        j = idx + h
        ok = vivant & (j < n)
        if not ok.any():
            break
        jj = j[ok]
        t_tp = (hi[jj] >= tp[ok]) if sens > 0 else (lo[jj] <= tp[ok])
        t_sl = (lo[jj] <= sl[ok]) if sens > 0 else (hi[jj] >= sl[ok])
        g, pr = t_tp & ~t_sl, t_sl
        pos = np.where(ok)[0]
        res[pos[g]] = 1
        res[pos[pr]] = 0
        v = vivant.copy()
        v[pos[g | pr]] = False
        vivant = v
    return res


def main() -> int:
    if not mt5.initialize():
        print(f"MT5 KO {mt5.last_error()}")
        return 1
    fin = datetime.now()
    m1 = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M1, DEB, fin)
    h1 = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_H1,
                              DEB - timedelta(days=20), fin)
    feats_ext = charge_source_externe(DEB - timedelta(days=1), fin)
    info = mt5.symbol_info("XAUUSD")
    spread = float(pd.DataFrame(m1)["spread"].mean() * info.point)
    mt5.shutdown()

    point = float(info.point)
    df = merge_m1_h1(m1, h1, feats_ext=feats_ext, point=point,
                     dropna_subset=FEATURE_COLS + ["atr_14"])
    n = len(df)
    print(f"{n:,} bougies | spread {spread:.2f}$ | "
          f"SL {SL_MULT}xATR TP {RR * SL_MULT}xATR")

    X = safe_normalize(df[FEATURE_COLS].to_numpy(np.float32),
                       load_norm_stats(), clip_sigma=5.0).astype(np.float32)
    atr = df["atr_14"].to_numpy(np.float64)
    hi, lo, cl = (df[c].to_numpy(np.float64) for c in ("high", "low", "close"))

    # Découpage TEMPOREL identique au walk-forward fold 1 : jamais aléatoire,
    # ce serait mélanger passé et futur.
    win = int(n * 0.8)
    i_tr0, i_tr1 = LOOKBACK, int(win * 0.55)
    i_va0, i_va1 = int(win * 0.55), int(win * 0.70)
    print(f"train [{i_tr0:,}:{i_tr1:,}]  ({df['time'].iloc[i_tr0]:%Y-%m-%d} -> "
          f"{df['time'].iloc[i_tr1]:%Y-%m-%d})")
    print(f"val   [{i_va0:,}:{i_va1:,}]  ({df['time'].iloc[i_va0]:%Y-%m-%d} -> "
          f"{df['time'].iloc[i_va1]:%Y-%m-%d})")

    print("\nEtiquetage par barriere triple...")
    t0 = time.time()
    jeux = {}
    for nom, (a, b) in (("train", (i_tr0, i_tr1)), ("val", (i_va0, i_va1))):
        idx = np.arange(a, b - MAX_HOLD - 1, PAS)
        yl = barrieres(hi, lo, cl, atr, spread, idx, +1, n)
        ys = barrieres(hi, lo, cl, atr, spread, idx, -1, n)
        garde = (yl >= 0) & (ys >= 0)
        jeux[nom] = (idx[garde], yl[garde].astype(np.float32),
                     ys[garde].astype(np.float32))
        print(f"  {nom:<6} {garde.sum():>8,} exemples  "
              f"WR long {100 * yl[garde].mean():.1f}%  "
              f"WR short {100 * ys[garde].mean():.1f}%")
    print(f"  ({time.time() - t0:.0f}s)")

    # Fenetres glissantes en VUE : aucune copie des 1.9 M x 25 x 10 valeurs.
    vues = np.lib.stride_tricks.sliding_window_view(X, LOOKBACK, axis=0)
    dev = get_device(_Cfg())
    pol = build_policy(dev, lookback=LOOKBACK)
    print(f"\ndevice {dev} | "
          f"{sum(p.numel() for p in pol.parameters()):,} parametres")
    # Reprise : une epoch coute ~7 min, inutile de la repayer apres une coupure.
    if os.path.exists(OUT) and "--neuf" not in sys.argv:
        pol.load_state_dict(torch.load(OUT, map_location=dev))
        print(f"reprise depuis {OUT}  (`--neuf` pour repartir de zero)")

    def lot(idx_b):
        # vues[i - LOOKBACK] couvre [i-LOOKBACK, i-1] : passe STRICT, la bougie
        # d'entree elle-meme n'est pas dans la fenetre.
        f = np.transpose(vues[idx_b - LOOKBACK], (0, 2, 1))       # (B, W, F)
        pos = np.zeros((len(idx_b), LOOKBACK, N_POS_FEATURES), np.float32)
        pos[:, :, 3] = 1.0                                        # risk_scale = 1
        return torch.from_numpy(
            np.concatenate([f, pos], axis=2).astype(np.float32)).to(dev)

    opt = torch.optim.AdamW(pol.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    idx_tr, yl_tr, ys_tr = jeux["train"]
    idx_va, yl_va, ys_va = jeux["val"]

    # La ou se joue la rentabilite : le modele ne tradera qu'une petite
    # fraction des instants, donc c'est la QUEUE qui compte, pas la moyenne.
    SELECTIVITES = [0.05, 0.02, 0.01, 0.005, 0.002]

    def wr_queue(y, sc):
        out = []
        for q in SELECTIVITES:
            seuil = np.quantile(sc, 1.0 - q)
            m = sc >= seuil
            out.append(100 * y[m].mean() if m.sum() >= 30 else float("nan"))
        return out

    def top1(y, sc):
        seuil = np.quantile(sc, 0.99)
        m = sc >= seuil
        return 100 * y[m].mean() if m.sum() > 30 else float("nan")

    requis_wr = (SL_MULT + spread / np.median(atr)) / (SL_MULT + RR * SL_MULT)

    best, sans_gain = -1.0, 0
    print(f"\n{'ep':>3} {'loss':>8} {'AUC long':>9} {'AUC short':>10} "
          f"{'AUC moy':>9} {'top1% L':>9} {'top1% S':>9} {'temps':>7}")
    print("-" * 74)
    for ep in range(1, EPOCHS + 1):
        t_ep = time.time()
        pol.train()
        ordre = np.random.permutation(len(idx_tr))
        perte = 0.0
        for d in range(0, len(ordre), BATCH):
            sel = ordre[d:d + BATCH]
            obs = lot(idx_tr[sel])
            cl_l = torch.from_numpy(yl_tr[sel]).to(dev)
            cl_s = torch.from_numpy(ys_tr[sel]).to(dev)
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                logits, _ = pol(obs)
                # logit 0 = BUY, logit 1 = SELL, entraines en binaire
                # INDEPENDANT (sigmoide) et non en softmax : les deux cotes
                # peuvent perdre simultanement, ce qu'un softmax interdirait.
                loss = bce(logits[:, 0], cl_l) + bce(logits[:, 1], cl_s)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            perte += float(loss) * len(sel)
        sched.step()

        pol.eval()
        zs = []
        with torch.no_grad():
            for d in range(0, len(idx_va), 2048):
                obs = lot(idx_va[d:d + 2048])
                with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                    lg, _ = pol(obs)
                zs.append(lg[:, :2].float().cpu())
        z = torch.cat(zs).numpy()
        auc_l = roc_auc_score(yl_va, z[:, 0])
        auc_s = roc_auc_score(ys_va, z[:, 1])
        moy = 0.5 * (auc_l + auc_s)

        print(f"{ep:>3} {perte / len(ordre):>8.4f} {auc_l:>9.4f} {auc_s:>10.4f} "
              f"{moy:>9.4f} {top1(yl_va, z[:, 0]):>8.1f}% "
              f"{top1(ys_va, z[:, 1]):>8.1f}% {time.time() - t_ep:>6.0f}s")
        ql = wr_queue(yl_va, z[:, 0])
        qs_ = wr_queue(ys_va, z[:, 1])
        etq = "  ".join(f"{100*q:g}%" for q in SELECTIVITES)
        print(f"      queue L [{etq}] = "
              + " ".join(f"{v:.1f}" for v in ql))
        print(f"      queue S [{etq}] = "
              + " ".join(f"{v:.1f}" for v in qs_)
              + f"   (requis {100*requis_wr:.1f})")

        if moy > best:
            best, sans_gain = moy, 0
            torch.save(pol.state_dict(), OUT)
        else:
            sans_gain += 1
            if sans_gain >= PATIENCE:
                print(f"\nArret : {PATIENCE} epochs sans progres.")
                break

    requis = (SL_MULT + spread / np.median(atr)) / (SL_MULT + RR * SL_MULT)
    print(f"\nMeilleure AUC moyenne : {best:.4f}  ->  {OUT}")
    print("Reperes : regression logistique 0.5442 | PPO seul 0.4848 | "
          "hasard 0.5000")
    print(f"WR d'equilibre a SL {SL_MULT}xATR : {100 * requis:.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
