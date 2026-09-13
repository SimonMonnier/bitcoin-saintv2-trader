"""Quel SL / R:R sur BTCUSD, maintenant que le plancher d'ATR est corrige ?

POURQUOI REFAIRE LA MESURE. Le reglage precedent (SL 3xATR, R:R 1.4) a ete
choisi alors que le plancher d'ATR valait 0.15 % du prix. Mesure sur BTCUSD :
ce plancher vaut 99.9 $ contre un ATR(14) M1 median de 34.5 $, donc il
l'emportait sur la quasi-totalite des bougies et un stop annonce a 3xATR etait
en realite pose a ~7.7xATR. Le chiffre retenu ne decrit pas la configuration
qu'il pretend decrire. A 0.01 % le plancher ne mord plus que sur 2.94 % des
bougies, et « 3xATR » veut enfin dire 3xATR.

CE QU'ON MESURE, ET LES DEUX PIEGES.

1. SURVIE. Un jeu de barrieres lointaines parait excellent tant qu'on ne compte
   que les trades RESOLUS : ceux qui n'ont atteint aucune barriere sont les
   trainards, et les jeter est un biais de survie. Mesure passee sur cette meme
   base : 10xATR / R:R 0.7 sortait en tete a +5.2 avec 83.1 % de resolution,
   puis devenait PERDANT (-0.0774 ATR/trade) une fois les non-resolus cloture au
   marche. On les cloture donc au marche, toujours.

2. NI L'AUC NI L'ESPERANCE BRUTE NE REPONDENT A LA QUESTION.

   L'esperance mesuree ici est celle d'une entree AU HASARD, sur chaque bougie
   candidate. Elle est donc negative partout, et ce qu'elle chiffre est le COUT
   DE LA FRICTION : -0.28 R a 2xATR, -0.06 R a 10xATR, la ou le spread fixe se
   dilue dans un stop plus large. C'est l'obstacle a franchir, pas un verdict.

   L'AUC, elle, va DANS L'AUTRE SENS : 0.615 au plus serre contre 0.523 au plus
   large (mesure, contrairement a ce que supposait la version precedente de ce
   commentaire). Les barrieres serrees se jouent sur quelques minutes, ou le
   retour a la moyenne est reellement previsible.

   Le bon critere croise les deux : de quelle marge le winrate obtenu EN
   SELECTIONNANT depasse-t-il le winrate d'equilibre. C'est ce que mesure le
   second tableau, et c'est lui qui decide.

Les frictions sont celles mesurees chez ce broker : spread moyen 2.607 bps,
paye des DEUX cotes.

    python mesure_sltp.py
"""

import sys
import time
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from saint_core import (FEATURE_COLS, ATR_PLANCHER_FRAC, load_norm_stats,
                        safe_normalize)

CACHE = "data_cache_BTCUSD_20221215.pkl"
MAX_HOLD = 240          # meme plafond que cfg.max_holding_bars
PAS = 10                # une bougie sur 10 : un trade dure des dizaines de
                        # minutes, echantillonner plus fin n'ajoute que du
                        # recouvrement entre observations
SPREAD_BPS = 2.61       # mesure, cf. training.PPOConfig.spread_bps

GRILLE_SL = (2.0, 3.0, 5.0, 8.0, 10.0)
GRILLE_RR = (1.0, 1.4, 2.0, 3.0)


def barrieres(hi, lo, cl, atr, idx, sl_mult, rr, sens, demi_spread):
    """Course TP/SL sur MAX_HOLD bougies, non-resolus CLOTURES AU MARCHE.

    Renvoie (pnl_en_R, resolu, motif_tp). `sens` vaut +1 pour un long.
    """
    n = len(cl)
    a = atr[idx]
    entree = cl[idx] + sens * demi_spread[idx]
    dist_sl = sl_mult * a
    tp = entree + sens * rr * dist_sl
    sl = entree - sens * dist_sl

    sortie = np.full(len(idx), np.nan)
    motif_tp = np.zeros(len(idx), bool)
    ouvert = np.ones(len(idx), bool)

    for h in range(1, MAX_HOLD + 1):
        j = idx + h
        valide = ouvert & (j < n)
        if not valide.any():
            break
        jj = j[valide]
        if sens > 0:
            touche_sl = lo[jj] <= sl[valide]
            touche_tp = hi[jj] >= tp[valide]
        else:
            touche_sl = hi[jj] >= sl[valide]
            touche_tp = lo[jj] <= tp[valide]

        pos = np.where(valide)[0]
        # Une bougie touchant les DEUX est comptee perdante : on ignore l'ordre
        # intra-minute, autant que ce soit dans le sens defavorable.
        k_sl = pos[touche_sl]
        k_tp = pos[touche_tp & ~touche_sl]
        sortie[k_sl] = sl[k_sl]
        sortie[k_tp] = tp[k_tp]
        motif_tp[k_tp] = True
        ouvert[k_sl] = False
        ouvert[k_tp] = False

    # Non resolus : cloture au marche a la derniere bougie disponible.
    reste = np.where(ouvert)[0]
    if len(reste):
        j = np.minimum(idx[reste] + MAX_HOLD, n - 1)
        sortie[reste] = cl[j]

    pnl_prix = sens * (sortie - entree) - demi_spread[idx]
    return pnl_prix / dist_sl, ~ouvert, motif_tp


def main() -> int:
    df = pd.read_pickle(CACHE)
    n = len(df)
    print(f"{n:,} bougies  {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

    hi, lo, cl = (df[c].to_numpy(np.float64) for c in ("high", "low", "close"))
    atr_brut = df["atr_14"].to_numpy(np.float64)
    atr = np.maximum(atr_brut, ATR_PLANCHER_FRAC * cl)
    demi_spread = (SPREAD_BPS / 1e4) * cl / 2.0

    X = safe_normalize(df[FEATURE_COLS].to_numpy(np.float32),
                       load_norm_stats(), clip_sigma=5.0).astype(np.float32)

    # Meme decoupe que le fold 1 de l'entrainement. La fenetre de TEST
    # (au-dela de 0.70) n'est jamais touchee ici.
    a_tr, b_tr = 0, int(n * 0.55)
    a_va, b_va = int(n * 0.55), int(n * 0.70)
    print(f"train [{a_tr:,} : {b_tr:,}]   val [{a_va:,} : {b_va:,}]   "
          f"test au-dela de {int(n*0.70):,}, INTOUCHE\n")

    i_tr = np.arange(a_tr, b_tr - MAX_HOLD, PAS)
    i_va = np.arange(a_va, b_va - MAX_HOLD, PAS)
    bon_tr = np.isfinite(atr[i_tr]) & (atr[i_tr] > 0)
    bon_va = np.isfinite(atr[i_va]) & (atr[i_va] > 0)
    i_tr, i_va = i_tr[bon_tr], i_va[bon_va]
    print(f"{len(i_tr):,} candidats d'entrainement, {len(i_va):,} de validation\n")

    spread_med = float(np.median(demi_spread) * 2)
    atr_med = float(np.median(atr))

    print(f"{'SL':>5} {'R:R':>5} {'WR eq':>7} {'resolu':>7} {'TP':>6} "
          f"{'WR':>6} {'E[R]':>8} {'t':>7} {'AUC':>7}")
    print("-" * 70)
    lignes = []
    for sl_mult in GRILLE_SL:
        for rr in GRILLE_RR:
            t0 = time.time()
            # Deux cotes, puis on empile : le spread rend les deux legerement
            # asymetriques, les moyenner serait faux.
            r_tr, res_tr, tp_tr, y_tr, xs_tr = [], [], [], [], []
            for sens in (+1, -1):
                r, res, tpm = barrieres(hi, lo, cl, atr, i_tr, sl_mult, rr,
                                        sens, demi_spread)
                r_tr.append(r); res_tr.append(res); tp_tr.append(tpm)
                y_tr.append(r > 0); xs_tr.append(X[i_tr])
            r_va, res_va, tp_va, y_va, xs_va = [], [], [], [], []
            for sens in (+1, -1):
                r, res, tpm = barrieres(hi, lo, cl, atr, i_va, sl_mult, rr,
                                        sens, demi_spread)
                r_va.append(r); res_va.append(res); tp_va.append(tpm)
                y_va.append(r > 0); xs_va.append(X[i_va])

            # Le probe apprend a predire « ce trade LONG finit-il gagnant ? ».
            # On ne melange pas les deux cotes dans un meme modele : ce sont
            # deux questions differentes.
            auc, p_va = [], []
            for k in (0, 1):
                lr = LogisticRegression(max_iter=400, n_jobs=-1)
                lr.fit(xs_tr[k], y_tr[k])
                p = lr.predict_proba(xs_va[k])[:, 1]
                p_va.append(p)
                if len(np.unique(y_va[k])) > 1:
                    auc.append(roc_auc_score(y_va[k], p))
            auc_moy = float(np.mean(auc)) if auc else np.nan

            # --- SELECTION : ce que donne le probe quand il CHOISIT ---
            sel = {}
            for q in (0.01, 0.05, 0.10):
                wr_s, er_s, n_s = [], [], 0
                for k in (0, 1):
                    if k >= len(p_va):
                        continue
                    seuil = np.quantile(p_va[k], 1.0 - q)
                    m = p_va[k] >= seuil
                    if m.sum() < 30:
                        continue
                    wr_s.append((r_va[k][m] > 0).mean())
                    er_s.append(r_va[k][m].mean())
                    n_s += int(m.sum())
                sel[q] = (float(np.mean(wr_s)) if wr_s else np.nan,
                          float(np.mean(er_s)) if er_s else np.nan, n_s)

            r_all = np.concatenate(r_va)
            res_all = np.concatenate(res_va)
            tp_all = np.concatenate(tp_va)
            esp = float(r_all.mean())
            t_stat = esp / (r_all.std() / np.sqrt(len(r_all)) + 1e-12)
            wr = float((r_all > 0).mean())
            wr_eq = (sl_mult * atr_med + spread_med) / (sl_mult * atr_med * (1 + rr))

            print(f"{sl_mult:>5.1f} {rr:>5.1f} {100*wr_eq:>6.1f}% "
                  f"{100*res_all.mean():>6.1f}% {100*tp_all.mean():>5.1f}% "
                  f"{100*wr:>5.1f}% {esp:>+8.4f} {t_stat:>+7.2f} {auc_moy:>7.4f}"
                  f"   ({time.time()-t0:.0f}s)")
            sys.stdout.flush()
            lignes.append((sl_mult, rr, wr_eq, res_all.mean(), esp, t_stat,
                           auc_moy, sel))

    print()
    print("E[R] = esperance par unite de risque, non-resolus clotures au marche.")
    print("t    = E[R] en ecarts-types : au-dela de +2, distinguable de zero.")
    print("AUC  = pouvoir de classement d'une regression logistique sur les 10")
    print("       features. Elle monte mecaniquement avec la distance des")
    print("       barrieres et ne doit JAMAIS servir seule a choisir.")
    print()
    # =================================================================
    # LE TABLEAU QUI DECIDE : ce que donne la SELECTION, contre l'equilibre
    # =================================================================
    print("=" * 78)
    print("EN SELECTIONNANT — winrate obtenu contre winrate d'equilibre")
    print("=" * 78)
    print(f"{'SL':>5} {'R:R':>5} {'WR eq':>7} | "
          + " | ".join(f"{f'top {100*q:g}%':^22}" for q in (0.01, 0.05, 0.10)))
    print(f"{'':>5} {'':>5} {'':>7} | "
          + " | ".join(f"{'WR':>6} {'ecart':>7} {'E[R]':>7}" for _ in range(3)))
    print("-" * 78)
    classement = []
    for sl_mult, rr, wr_eq, res, esp, t, auc, sel in lignes:
        ligne = f"{sl_mult:>5.1f} {rr:>5.1f} {100*wr_eq:>6.1f}% |"
        for q in (0.01, 0.05, 0.10):
            wr_s, er_s, n_s = sel[q]
            if np.isnan(wr_s):
                ligne += f" {'--':>6} {'--':>7} {'--':>7} |"
            else:
                ligne += (f" {100*wr_s:>5.1f}% {100*(wr_s-wr_eq):>+6.1f}pt "
                          f"{er_s:>+7.4f} |")
        print(ligne)
        for q in (0.01, 0.05, 0.10):
            wr_s, er_s, n_s = sel[q]
            if not np.isnan(er_s):
                classement.append((er_s, sl_mult, rr, q, wr_s, wr_eq, n_s, auc))

    print()
    print("`ecart` = winrate obtenu MOINS winrate d'equilibre. C'est la seule")
    print("colonne qui dise si la configuration peut gagner : positive, la")
    print("selection paie les frais ; negative, elle ne les paie pas.")
    print()
    classement.sort(key=lambda x: -x[0])
    print("Meilleures esperances APRES selection :")
    for er, sl_mult, rr, q, wr_s, wr_eq, n_s, auc in classement[:6]:
        print(f"  SL {sl_mult:>4.1f}xATR  R:R {rr:>3.1f}  top {100*q:>4.1f}%  "
              f"E[R] {er:+.4f}  WR {100*wr_s:.1f}% contre {100*wr_eq:.1f}% "
              f"requis ({100*(wr_s-wr_eq):+.1f} pt, {n_s} trades)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
