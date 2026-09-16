"""Est-ce que QUOI QUE CE SOIT peut classer cette cible ? La question qui decide.

POURQUOI ELLE PASSE AVANT TOUT LE RESTE. Le run en cours gagne en validation
mais son `rho` reste dans le bruit : le profit vient de la geometrie, pas du
modele. Deux diagnostics opposes expliquent cela, et ils appellent des
remedes contraires :

  A. L'INFORMATION EST LA, mais PPO ne l'extrait pas. Il optimise le rendement
     de ses actions ; personne ne lui demande que l'ORDRE de ses probabilites
     soit juste, et c'est pourtant tout ce dont la selectivite se sert. Le
     plus vieux chiffre du depot dit exactement cela : une regression
     logistique obtient 0.6271 d'AUC contre 0.5707 pour la politique.

  B. L'INFORMATION N'Y EST PAS pour cette cible. Alors changer d'apprenant ne
     servirait a rien, et il faudrait revoir la cible ou les features.

Un modele supervise simple tranche. S'il classe, c'est A. S'il ne classe pas,
c'est B — et toute l'energie mise sur l'architecture depuis des semaines
aurait ete mal placee.

CE QUI EST COMPARE. Trois apprenants entraines sur le TRAIN a predire la cible
que la tete auxiliaire vise — "ce trade touchera-t-il 2 R avant -1 R" — puis
mesures sur la VALIDATION, jamais ailleurs. Leur rho de rang est directement
comparable a celui que le journal affiche a chaque epoch.

LE TEST N'EST PAS OUVERT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import cibles as C
import training as T
from saint_core import FEATURE_COLS

PAS = 12          # une occasion par heure


def rang(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float)
    xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def rho(a, b):
    ra, rb = rang(np.asarray(a, float)), rang(np.asarray(b, float))
    ra -= ra.mean()
    rb -= rb.mean()
    d = ra.std() * rb.std() * len(ra)
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def err_bloc(a, b, nb=40):
    """Incertitude par blocs contigus : les barres voisines partagent l'avenir."""
    t = max(len(a) // nb, 2)
    v = [rho(a[i:i + t], b[i:i + t]) for i in range(0, len(a) - t + 1, t)]
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 2 else float("nan")


def main() -> int:
    cfg = T.PPOConfig()
    df = T.load_mt5_data(cfg)
    n = len(df)
    tr = int(n * 0.55)
    va = tr + int(n * 0.15)
    print(f"train [0, {tr:,})   validation [{tr:,}, {va:,})   test intouche")
    print(f"cible : touche-t-on {cfg.aux_tp_mult / cfg.atr_sl_mult:.0f} R avant "
          f"-1 R, stop {cfg.atr_sl_mult:g}xATR\n")

    def jeu(a, b):
        d = df.iloc[a:b].reset_index(drop=True)
        idx = np.arange(cfg.lookback, len(d) - C.BORNE_DEFAUT - 2, PAS)
        ra, rv = C.rendements(d, idx, cfg, indicateur=True)
        ok = np.isfinite(ra) & np.isfinite(rv)
        X = d[FEATURE_COLS].to_numpy(np.float32)[idx[ok]]
        y = ((ra[ok] - rv[ok]) / 2.0).astype(np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), y

    Xtr, ytr = jeu(0, tr)
    Xva, yva = jeu(tr, va)
    print(f"{len(ytr):,} occasions d'entrainement, {len(yva):,} de validation\n")

    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-8)
    Ztr = np.clip((Xtr - mu) / sd, -10, 10)
    Zva = np.clip((Xva - mu) / sd, -10, 10)

    print(f"{'apprenant':<34} {'rho validation':>15} {'+/-':>8} {'sigma':>7}")
    print("-" * 68)

    # 1. Moindres carres regularises — le plus simple qui soit.
    lam = 1e3
    A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1])
    w = np.linalg.solve(A, Ztr.T @ ytr)
    p = Zva @ w
    r, e = rho(p, yva), err_bloc(p, yva)
    print(f"{'moindres carres regularises':<34} {r:>+15.4f} {e:>8.4f} "
          f"{r / max(e, 1e-9):>+7.1f}")

    # 2. Gradient boosting — non lineaire, sans reglage fin.
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=200, max_depth=4,
                                          learning_rate=0.05,
                                          random_state=0)
        m.fit(Ztr, ytr)
        p = m.predict(Zva)
        r, e = rho(p, yva), err_bloc(p, yva)
        print(f"{'gradient boosting':<34} {r:>+15.4f} {e:>8.4f} "
              f"{r / max(e, 1e-9):>+7.1f}")
    except Exception as exc:
        print(f"{'gradient boosting':<34}   indisponible ({type(exc).__name__})")

    # 3. Temoin : des poids au hasard. Il doit rendre zero, sinon la mesure
    #    elle-meme fabrique de la correlation et rien de ce qui precede ne
    #    veut dire quoi que ce soit.
    rng = np.random.default_rng(0)
    p = Zva @ rng.normal(size=Zva.shape[1])
    r, e = rho(p, yva), err_bloc(p, yva)
    print(f"{'TEMOIN poids au hasard':<34} {r:>+15.4f} {e:>8.4f} "
          f"{r / max(e, 1e-9):>+7.1f}")

    print("\nLECTURE. Si un apprenant simple depasse nettement zero, "
          "l'information")
    print("EST dans les features et c'est PPO qui ne l'extrait pas : le remede")
    print("est de trier avec la tete auxiliaire plutot qu'avec les")
    print("probabilites de la politique. Si aucun n'y arrive, le probleme est")
    print("la cible ou les features, et changer d'apprenant ne servirait a rien.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
