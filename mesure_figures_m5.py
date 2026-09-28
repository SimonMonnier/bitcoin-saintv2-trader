# -*- coding: utf-8 -*-
"""Les FIGURES des chandeliers apportent-elles quelque chose a l'expert M5 ?

2026-09-28, demande du proprietaire : « envoyer au modele les figures des
chandeliers plutot que des chiffres, pour qu'il apprenne a trader les
figures ; un modele qui marcherait sur n'importe quelle unite de temps et
n'importe quel indice ».

LES FIGURES, sans aucun prix : pour chacune des 16 dernieres bougies (j = 0
la bougie qui vient de fermer, j = 15 la plus ancienne),

    corps_j      (cloture - ouverture) / hauteur        de -1 a +1, le signe = la couleur
    haut_j       meche du haut / hauteur                de 0 a 1
    bas_j        meche du bas / hauteur                 de 0 a 1
    taille_j     hauteur / ATR du moment                la bougie contre les recentes
    ecart_j      (ouverture - cloture precedente) / ATR
    niveau_j     (cloture_j - cloture actuelle) / ATR   le dessin des 16 clotures

Le meme marteau donne les memes chiffres sur le BTC en M5 et le DAX en H1.

CE QUI EST MESURE, sur les blocs 1 et 2 du jeu M5 (memes blocs et purge que
`main_blocs`), l'expert LightGBM du jeu (memes reglages), appris sur
l'entrainement et juge sur la VALIDATION, jamais sur le test :

    A  les 40 colonnes actuelles
    B  les 40 colonnes + les figures
    C  les figures seules

Pour chacun : la correlation de rang de la prediction avec le R net moyen
des 16 coups (la cible de l'expert), et le R net moyen des bougies qui
passent la porte (rang glissant >= 0.85 dans le sens joue).

    python mesure_figures_m5.py > mesure_figures_m5.txt

UNE SEULE BOUGIE — 2026-09-28, precision du proprietaire : « une seule
bougie au temps T ; avec le lookback, le modele verrait les autres ». Le
modele du jeu lit `lookback` = 4 lignes : la bougie T sur chaque ligne, il
voit donc les bougies T a T-3. Versions mesurees :

    A     les 40 colonnes actuelles
    A+T   les 40 colonnes + la forme de la bougie T (5 colonnes)
    T     la bougie T seule
    T-3   les bougies T a T-3 (ce que le lookback de 4 lui montrerait)

    python mesure_figures_m5.py une > mesure_figures_m5_une.txt
"""
import sys
import time

import numpy as np
import pandas as pd

import jeu_kairos as J

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

K = 16


def figures(d: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = (d[x].to_numpy(np.float64) for x in ("open", "high", "low", "close"))
    atr = d["atr_14"].to_numpy(np.float64)
    atr = np.where(atr > 0, atr, np.nan)
    rg = h - l
    rg = np.where(rg > 0, rg, np.nan)
    pc = np.r_[np.nan, c[:-1]]
    base = {"corps": (c - o) / rg, "haut": (h - np.maximum(o, c)) / rg,
            "bas": (np.minimum(o, c) - l) / rg}
    out = {}
    for j in range(K):
        dec = lambda x: pd.Series(x).shift(j).to_numpy()
        for nom, v in base.items():
            out[f"{nom}_{j}"] = dec(v)
        out[f"taille_{j}"] = dec(rg) / atr
        out[f"ecart_{j}"] = dec(o - pc) / atr
        if j:
            out[f"niveau_{j}"] = (dec(c) - c) / atr
    f = pd.DataFrame(out)
    return f.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def expert(X, y, idx_tr, idx_va):
    import lightgbm as lgb
    pred = np.full((len(idx_va), 2), np.nan)
    for s_ in range(2):
        u = idx_tr[np.isfinite(y[idx_tr, s_])]
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31,
                              min_child_samples=400, subsample=0.7, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=5.0, n_jobs=4, verbose=-1)
        m.fit(X[u], y[u, s_])
        pred[:, s_] = m.predict(X[idx_va])
    return pred


def juge(pred, y, jour):
    ic = [pd.Series(pred[:, s_]).corr(pd.Series(y[:, s_]), method="spearman") for s_ in range(2)]
    rang = np.column_stack([pd.Series(pred[:, s_]).rolling(2000, min_periods=200)
                            .rank(pct=True).fillna(0.5).to_numpy() for s_ in range(2)])
    v, jr = [], []
    for s_ in range(2):
        ok = (rang[:, s_] >= 0.85) & np.isfinite(y[:, s_])
        v.append(y[ok, s_])
        jr.append(jour[ok])
    v, jr = np.concatenate(v), np.concatenate(jr)
    pj = pd.Series(v).groupby(jr).mean()
    t = pj.mean() / (pj.std(ddof=1) / np.sqrt(len(pj))) if len(pj) > 2 else float("nan")
    return ic, float(v.mean()), float(t), len(v) / max(len(np.unique(jour)), 1)


def main() -> int:
    cfg = J.JeuConfig()
    d = pd.read_pickle(cfg.cache)
    cols = J.colonnes_jeu(cfg)
    N = len(d)
    o, h, l, c = (d[x].to_numpy(np.float64) for x in ("open", "high", "low", "close"))
    atr = J.atr_effectif(d["atr_14"].to_numpy(np.float64), c, cfg)
    sp = d["spread_bar"].to_numpy(np.float64)
    R1, _, _ = J.table_coups(o, h, l, sp, atr, cfg, 1.0)
    y = J.cibles_expert(R1)
    del R1
    jour = d["time"].dt.floor("D").values.astype("int64")
    XA = d[cols].to_numpy(np.float32)
    F = figures(d)
    XC = F.to_numpy(np.float32)
    XB = np.concatenate([XA, XC], axis=1)
    versions = (("A  actuelles", XA), ("B  actuelles + figures", XB), ("C  figures seules", XC))
    if len(sys.argv) > 1 and sys.argv[1] == "une":
        t0c = [f"{x}_0" for x in ("corps", "haut", "bas", "taille", "ecart")]
        t4c = [col for col in F.columns if int(col.rsplit("_", 1)[1]) <= 3]
        XT = F[t0c].to_numpy(np.float32)
        versions = (("A    actuelles", XA),
                    ("A+T  actuelles + bougie T", np.concatenate([XA, XT], axis=1)),
                    ("T    la bougie T seule", XT),
                    ("T-3  bougies T a T-3", F[t4c].to_numpy(np.float32)))
        print(f"UNE SEULE BOUGIE, BTC M5 — {N:,} bougies ; la bougie T = {len(t0c)} colonnes, "
              f"T a T-3 = {len(t4c)} colonnes (ce que montre un lookback de 4)")
    else:
        print(f"FIGURES DES CHANDELIERS, BTC M5 — {N:,} bougies ; A = {XA.shape[1]} colonnes "
              f"actuelles, B = A + {XC.shape[1]} colonnes de figures ({K} bougies), C = figures seules")
    print("jugement sur la VALIDATION de chaque bloc ; le bloc de test n'est pas lu\n")
    purge = int(cfg.horizon_max + cfg.lookback + cfg.purge_semaines * cfg.barres_par_partie)
    H = int(cfg.horizon_max)
    fmt = lambda i: pd.Timestamp(d["time"].iloc[min(i, N - 1)]).strftime("%Y-%m-%d")
    for k in (0, 1):
        permis, _te, (va0, va1) = J.masque_blocs(N, cfg.n_blocs, k, purge)
        fin_tr = J.prochain_exclu(permis)
        idx = np.flatnonzero(permis)
        idx_tr = idx[idx + 1 + H < fin_tr[idx]]
        idx_va = np.arange(va0, va1)
        print(f"=== BLOC {k + 1} : validation {fmt(va0)} -> {fmt(va1 - 1)} ===")
        print("   version                    correlation achat / vente   R net porte >= 0.85   "
              "t (jours)   bougies/jour a la porte")
        for nom, X in versions:
            t0 = time.time()
            pred = expert(X, y, idx_tr, idx_va)
            ic, r, t, n = juge(pred, y[idx_va], jour[idx_va])
            print(f"   {nom:24s}   {ic[0]:+.3f} / {ic[1]:+.3f}             {r:+.4f} R          "
                  f"{t:+5.1f}        {n:5.1f}      ({time.time() - t0:.0f} s)", flush=True)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
