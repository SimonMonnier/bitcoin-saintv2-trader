"""Combien de trades une selectivite peut-elle produire, au maximum ?

CETTE BORNE NE DEPEND PAS DU MODELE. Une seule position est tenue a la fois :
tout signal emis pendant qu'un trade court est ignore, par l'environnement
comme par le live. Le nombre de trades sature donc a
*duree de la fenetre / duree d'un trade*, et elargir la selectivite au-dela
de ce point n'ajoute plus rien — les barres nouvellement eligibles tombent
presque toutes pendant qu'une position est deja ouverte.

C'est devenu le fait dominant le 16 septembre : en passant le stop de 8 a
12xATR avec sortie au stop suiveur, la duree d'un trade a a peu pres double.
Le plafond a donc ete divise par deux le jour ou la geometrie a change, sans
que ce soit mesure.

CE FICHIER NE CHARGE PAS DE MODELE. Il tire le signal au hasard a un taux
donne, ce qui isole exactement l'effet d'occupation : la courbe obtenue est
le PLAFOND que la selectivite peut atteindre, quel que soit le modele. Un
vrai modele ne peut que rester dessous, jamais au-dessus.

LE TEST N'EST PAS OUVERT : seules train et validation du fold sont lues.
"""

from __future__ import annotations

import numpy as np

import cibles as C
import training as T

T_TRAIN, T_VAL, T_TEST = 0.55, 0.15, 0.10
PAS = 3                  # une decision candidate toutes les 15 minutes
TAUX = (1.00, 0.50, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01)
TIRAGES = 12             # phases, pour que le resultat ne tienne pas au hasard


def enchaine(garde, pos, dur):
    """Indices des trades REELLEMENT pris : une position a la fois."""
    pris, libre = [], -1.0
    for k in range(len(pos)):
        if pos[k] < libre or not garde[k] or not np.isfinite(dur[k]):
            continue
        pris.append(k)
        libre = pos[k] + dur[k]
    return np.array(pris, dtype=int)


def main() -> int:
    cfg = T.PPOConfig()
    df = T.load_mt5_data(cfg)
    n = len(df)
    depart = 0
    b_tr = depart + int(n * T_TRAIN)
    tr, va = b_tr, b_tr + int(n * T_VAL)
    dv = df.iloc[tr:va].reset_index(drop=True)
    del df
    jours = len(dv) * 5 / 60 / 24
    print(f"fold 1, VALIDATION [{tr:,}, {va:,}) — {jours:.0f} jours")
    print(f"sortie : stop {cfg.atr_sl_mult:g}xATR, trailing "
          f"{cfg.atr_trail_mult / cfg.atr_sl_mult:.2f} R, objectif "
          f"{'aucun' if not cfg.use_tp else str(cfg.atr_tp_mult) + 'xATR'}\n")

    idx = np.arange(cfg.lookback, len(dv) - C.BORNE_DEFAUT - 2, PAS)
    ra, rv, da, dvv = C.rendements(dv, idx, cfg, durees=True)
    # Sens tire a pile ou face : on mesure l'occupation, pas une direction.
    rng = np.random.default_rng(0)
    sens = rng.integers(0, 2, len(idx)) * 2 - 1
    rend = np.where(sens > 0, ra, rv)
    dur = np.where(sens > 0, da, dvv)
    fini = np.isfinite(dur)
    med = float(np.nanmedian(dur))
    moy = float(np.nanmean(dur))
    print(f"{len(idx):,} decisions candidates, {int(fini.sum()):,} resolues")
    print(f"duree d'un trade : mediane {med*5/60:.1f} h, moyenne {moy*5/60:.1f} h")
    print(f"PLAFOND THEORIQUE : {jours*24/(moy*5/60):.0f} trades sur la fenetre "
          f"(duree fenetre / duree moyenne)\n")

    print(f"{'garde':>7} {'trades pris':>12} {'+/-':>6} {'x vs 5%':>9} "
          f"{'occupation':>11} {'E[R] hasard':>12}")
    print("-" * 64)
    base = None
    for q in TAUX:
        ns, ers = [], []
        for t in range(TIRAGES):
            r2 = np.random.default_rng(100 + t)
            garde = r2.random(len(idx)) < q if q < 1.0 else np.ones(len(idx), bool)
            k = enchaine(garde, idx, dur)
            ns.append(len(k))
            if len(k) >= 10:
                ers.append(float(np.nanmean(rend[k])))
        m, e = float(np.mean(ns)), float(np.std(ns, ddof=1))
        occ = float(np.mean([np.nansum(dur[enchaine(
            (np.random.default_rng(100 + t).random(len(idx)) < q)
            if q < 1.0 else np.ones(len(idx), bool), idx, dur)])
            for t in range(4)])) / len(dv)
        if q == 0.05:
            base = m
        rel = f"{m / base:.2f}x" if base else "-"
        print(f"{100*q:>6.0f}% {m:>12.0f} {e:>6.0f} {rel:>9} "
              f"{100*occ:>10.0f}% {np.mean(ers):>+12.4f}")

    print("\nLECTURE. 'occupation' est la part du temps ou une position est")
    print("ouverte. Des qu'elle approche 100 %, la colonne 'trades pris' cesse")
    print("de monter : les barres nouvellement eligibles tombent pendant qu'un")
    print("trade court deja. C'est le plafond, et aucun modele ne le franchit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
