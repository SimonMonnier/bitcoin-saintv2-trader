# -*- coding: utf-8 -*-
"""Rejoue le TEST d'un bloc d'un run du jeu (BTC seul, validation en blocs)
depuis son meilleur modele sauvegarde.

2026-09-28 : le run kairos_jeu_m5_02 a ete tue par erreur a l'epoch 27/40 du
bloc 1, avant son test (voir la memoire « TaskStop tue l'entrainement »). Son
meilleur modele (`best_<prefixe>_bloc<k>.pth`) est sur le disque. Ce script
refait ce que `main_blocs` fait avant le test — memes blocs, meme purge,
expert reappris sur les memes bougies — et joue le bloc de test avec ce
modele. Le test reste jamais vu : rien n'est choisi d'apres lui.

    python rejoue_test_bloc.py kairos_jeu_m5_02 1
"""
import dataclasses
import json
import sys

import numpy as np
import pandas as pd
import torch

import jeu_kairos as J

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def config(prefixe: str) -> J.JeuConfig:
    brut = json.load(open(f"run_{prefixe}.json", encoding="utf-8"))
    champs = {f.name for f in dataclasses.fields(J.JeuConfig)}
    return J.JeuConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                          for k, v in brut.items() if k in champs})


def main() -> int:
    prefixe, k = sys.argv[1], int(sys.argv[2]) - 1
    cfg = config(prefixe)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = pd.read_pickle(cfg.cache)
    cols = list(dict.fromkeys(["time", "open", "high", "low", "close", "atr_14",
                               "spread_bar"] + J.colonnes_jeu(cfg)))
    d = d[cols].reset_index(drop=True)
    N = len(d)
    o, h, l, c = (d[x].to_numpy(np.float64) for x in ("open", "high", "low", "close"))
    atr = J.atr_effectif(d["atr_14"].to_numpy(np.float64), c, cfg)
    sp = d["spread_bar"].to_numpy(np.float64)
    t_ns = d["time"].values.astype("int64")
    X = d[J.colonnes_jeu(cfg)].to_numpy(np.float32)
    R1, D1, S1 = J.table_coups(o, h, l, sp, atr, cfg, 1.0)
    y_ex = J.cibles_expert(R1)
    toutes = J.parties(d["time"], 0, N, cfg)
    purge = int(cfg.horizon_max + cfg.lookback + cfg.purge_semaines * cfg.barres_par_partie)
    permis, (te0, te1), _ = J.masque_blocs(N, cfg.n_blocs, k, purge)
    fin_tr = J.prochain_exclu(permis)
    j_te = np.array([w for w in toutes if w[0] >= te0 and w[1] <= te1], np.int64).reshape(-1, 2)
    st_m = X[permis].astype(np.float64).mean(0)
    st_s = X[permis].astype(np.float64).std(0)
    Xn = J.safe_normalize(X, {"mean": st_m.astype(np.float32),
                              "std": st_s.astype(np.float32)}).astype(np.float32)
    pred, _ = J.expert_multi(Xn, y_ex, t_ns, permis, fin_tr, cfg)
    FE = J.features_expert(pred, fenetre=10_000 // int(cfg.minutes_par_barre))
    rangs = FE[:, 2:4].copy()
    m_e = FE[permis].astype(np.float64).mean(0)
    s_e = FE[permis].astype(np.float64).std(0)
    FEn = np.clip((FE - m_e) / (s_e + 1e-8), -5.0, 5.0).astype(np.float32)
    Xk = np.concatenate([Xn, FEn[:, :cfg.n_expert]], axis=1)
    src = f"best_{prefixe}_bloc{k + 1}.pth"
    etat = torch.load(src, map_location=device, weights_only=False)
    policy = J.PolitiqueJeu(cfg).to(device)
    policy.load_state_dict(etat["modele"])
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.graine)
    s_t, c_t, _ = J.joue(policy, j_te, Xk, R1, D1, S1, te1, cfg, device, explore=False,
                         gen=gen, rangs=rangs, marge=J.fraction_marge(c, atr, cfg))
    bt = J.bilan(s_t, c_t, c, atr, sp, cfg)
    fmt = lambda i: pd.Timestamp(d["time"].iloc[min(i, N - 1)]).strftime("%Y-%m-%d")
    print(f"{prefixe} bloc {k + 1} : test {fmt(te0)} -> {fmt(te1 - 1)} ({len(j_te)} parties), "
          f"modele {src} (epoch {etat.get('epoch')})")
    print(f"TEST fold {k + 1} (rejoue)  {J.ligne_bilan(bt, cfg)}")
    print(f"TEST fold {k + 1} bilan  {J.ligne_detail(bt, cfg)}")
    with open(f"test_{prefixe}_bloc{k + 1}.json", "w", encoding="utf-8") as fh:
        json.dump({a: v for a, v in bt.items()}, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
