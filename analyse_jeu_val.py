# -*- coding: utf-8 -*-
"""Ou le modele du jeu gagne-t-il et perd-il ? Validation du fold 1 seulement.

2026-09-26, run kairos_jeu_btc04. Le PPO recoit le score de l'expert et son
rang, mais fait moins bien que l'expert joue seul sur ses 2 % de signaux
les plus forts. Avant de poser une porte d'entree sur ce rang, on regarde
trade par trade OU le modele entre : a quel rang de l'expert, dans quel
sens par rapport a lui, avec quelles barrieres, a quelle heure — et ce que
chaque groupe rapporte.

VALIDATION DU FOLD 1 SEULEMENT, [n_tr : n_tr + n_va) : elle precede toute
fenetre de test. Celle du fold 2 recouvre le test du fold 1 et n'est pas
analysee ici.

    python analyse_jeu_val.py
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

import jeu_kairos as J
from saint_core import FEATURE_COLS, safe_normalize

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def groupe(nom, m, r, net, brut):
    n = int(m.sum())
    if n == 0:
        return f"  {nom:<34} {0:>5}"
    rr, bb = r[m], brut[m]
    g, p = rr[rr > 0].sum(), -rr[rr < 0].sum()
    t = bb.mean() / (bb.std(ddof=1) / np.sqrt(n) + 1e-12) if n > 1 else float("nan")
    return (f"  {nom:<34} {n:>5}  {rr.mean():+7.3f} R  {10 * rr.sum() / 97:+6.2f} $/jour  "
            f"net {net[m].mean():+7.2f}  brut {bb.mean():+7.2f} (t {t:+.1f})  "
            f"PF {g / p if p > 0 else float('inf'):5.2f}")


def main() -> int:
    cfg = J.JeuConfig()
    t0 = time.time()
    d = pd.read_pickle(cfg.cache)
    cols = list(dict.fromkeys(["time", "open", "high", "low", "close", "atr_14",
                               "spread_bar"] + list(FEATURE_COLS)))
    d = d[cols].reset_index(drop=True)
    N = len(d)
    o, h, l, c = (d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    atr = J.atr_effectif(d["atr_14"].to_numpy(np.float64), c, cfg)
    sp = d["spread_bar"].to_numpy(np.float64)
    n_tr, n_va = int(N * cfg.part_train), int(N * cfg.part_val)
    a_va, a_te = n_tr, n_tr + n_va
    X = d[list(FEATURE_COLS)].to_numpy(np.float32)
    st = {"mean": X[:n_tr].astype(np.float64).mean(0).astype(np.float32),
          "std": X[:n_tr].astype(np.float64).std(0).astype(np.float32)}
    Xn = safe_normalize(X, st).astype(np.float32)
    del X
    R1, D1, S1 = J.table_coups(o, h, l, sp, atr, cfg, 1.0)
    y = J.cibles_expert(R1, cfg)
    pred, _ = J.expert_realiste(Xn, y, n_tr, N, cfg)
    FE = J.features_expert(pred)
    m_e = FE[:n_tr].astype(np.float64).mean(0)
    s_e = FE[:n_tr].astype(np.float64).std(0)
    Xn = np.concatenate([Xn, np.clip((FE - m_e) / (s_e + 1e-8), -5, 5)
                         .astype(np.float32)[:, :cfg.n_expert]], axis=1)
    j_va = J.journees(d["time"], a_va, a_te)
    print(f"preparation {time.time() - t0:.0f} s ; validation du fold 1 "
          f"[{a_va:,} : {a_te:,}), {len(j_va)} journees ; le test n'est pas lu\n",
          flush=True)
    heure_ny = ((d["time"] - pd.Timedelta(hours=7)).dt.hour).to_numpy()
    atr_bps = atr / c * 1e4
    ksl = np.asarray(cfg.sl_atr)

    for nom in ("best", "last"):
        f = f"{nom}_{cfg.prefixe}_wf1.pth"
        ck = torch.load(f, map_location="cpu", weights_only=False)
        pol = J.PolitiqueJeu(cfg)
        pol.load_state_dict(ck["modele"])
        sc, cp, _ = J.joue(pol, j_va, Xn, R1, D1, S1, a_te, cfg, "cpu", explore=False)
        b = J.bilan(sc, cp, c, atr, sp, cfg)
        print(f"=== {f} (epoch {ck.get('epoch')}) : {J.ligne_bilan(b, cfg)}")
        if not cp:
            continue
        a = np.array(cp)
        t, s, i, j = (a[:, k].astype(np.int64) for k in (1, 2, 3, 4))
        r, so = a[:, 5], a[:, 7].astype(np.int64)
        net = r * ksl[j] * atr[t] / c[t] * 1e4
        cout = sp[t] + cfg.glissement_entree_bps + np.where(so != 0, cfg.glissement_sortie_bps, 0.0)
        brut = net + cout
        rang_sens = FE[t, 2 + s]
        rang_max = np.maximum(FE[t, 2], FE[t, 3])
        sens_expert = np.argmax(pred[t], axis=1)
        tous = np.ones(len(t), bool)
        print("  groupe                             coups   R/coup     $/jour    net bps   brut bps          PF")
        print(groupe("tous les coups", tous, r, net, brut))
        print("\n  -- RANG DE L'EXPERT, dans le sens joue --")
        for lo, hi in ((0.98, 1.01), (0.90, 0.98), (0.50, 0.90), (-1, 0.50)):
            print(groupe(f"rang {lo:.2f} a {min(hi, 1):.2f}", (rang_sens >= lo) & (rang_sens < hi),
                         r, net, brut))
        print(groupe("rang max des deux sens >= 0.98", rang_max >= 0.98, r, net, brut))
        print("\n  -- LE SENS PAR RAPPORT A L'EXPERT --")
        print(groupe("meme sens que l'expert", s == sens_expert, r, net, brut))
        print(groupe("sens contraire", s != sens_expert, r, net, brut))
        print(groupe("achat", s == 0, r, net, brut))
        print(groupe("vente", s == 1, r, net, brut))
        print("\n  -- LES BARRIERES (objectif / stop, en ATR) --")
        for ii in range(len(cfg.tp_atr)):
            for jj in range(len(cfg.sl_atr)):
                m = (i == ii) & (j == jj)
                if m.sum() >= 5:
                    print(groupe(f"{cfg.tp_atr[ii]:g} / {cfg.sl_atr[jj]:g}", m, r, net, brut))
        print("\n  -- L'HEURE (New York) --")
        for lo in range(0, 24, 4):
            print(groupe(f"{lo:02d} h - {lo + 4:02d} h", (heure_ny[t] >= lo) & (heure_ny[t] < lo + 4),
                         r, net, brut))
        print("\n  -- LA VOLATILITE A L'ENTREE (ATR des barrieres, bps) --")
        for lo, hi in ((0, 9), (9, 12), (12, 18), (18, 1e9)):
            print(groupe(f"{lo:g} a {hi:g} bps" if hi < 1e8 else f"plus de {lo:g} bps",
                         (atr_bps[t] >= lo) & (atr_bps[t] < hi), r, net, brut))
        print("\n  -- LA SORTIE --")
        for k, nm in enumerate(("objectif", "stop", "temps")):
            print(groupe(nm, so == k, r, net, brut))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
