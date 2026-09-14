"""Combien de features, et quel SL/R:R — avec la friction COMPLETE.

DEUX CORRECTIONS PAR RAPPORT A mesure_sltp.py.

1. FRICTION. La premiere version ne facturait que le spread. L'environnement
   d'entrainement facture aussi le slippage d'entree (1.0 bps) et de sortie
   (2.0 bps), plus un bruit de tick qui etend high/low de 0.6 bps en moyenne.
   Sur BTCUSD a 66 583 $ cela fait ~20 $ de plus, soit 29 % d'un stop a 2xATR
   (69 $) — et c'est justement la configuration la plus serree, donc la plus
   sensible, qui sortait gagnante. Le classement peut s'inverser.

   On reproduit le modele EXACT de l'environnement, y compris son slippage
   FAVORABLE sur TP : un take-profit est un ordre limite, qui remplit au prix
   ou mieux si le marche traverse. C'est defendable, mais c'est flatteur, et il
   faut le savoir en lisant le resultat.

2. LARGEUR DU JEU. Le jeu est passe de 21 a 10 features parce que chaque
   feature retiree mesurait un apport individuel quasi nul. Mais une mesure
   ulterieure a montre que LES APPORTS MARGINAUX NE SE COMPOSENT PAS : retirer
   9 features individuellement nulles a coute 0.0024 d'AUC, plus que la
   meilleure feature du jeu n'en apporte. L'elagage n'etait donc pas gratuit.
   On compare donc plusieurs largeurs sur la meme sonde.

BIAIS DE SELECTION — a garder en tete. On choisit la meilleure case d'une
grille evaluee sur la MEME fenetre de validation. Le maximum de N estimations
bruitees est biaise vers le haut. Les chiffres sont des bornes optimistes, et
seule la fenetre de test (jamais touchee ici) peut les confirmer.

    python mesure_features.py
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from saint_core import FEATURE_COLS, ATR_PLANCHER_FRAC

CACHE = "data_cache_BTCUSD_20221215.pkl"
MAX_HOLD = 240
PAS = 10

SPREAD_BPS = 2.61
SLIP_SORTIE_BPS = 2.0
SLIP_ENTREE_BPS = 1.0
BRUIT_TICK_BPS = 0.0      # l'env ne bruite plus les meches : les bougies MT5
                          # portent deja les extremes reels
# Pas de commission : ce courtier n en facture pas sur BTCUSD CFD.
# Le cout est entierement dans le spread, deja facture ci-dessus.
FEE_RATE = 0.0
TP_SLIPPAGE_FAVORABLE = ('--tp-neutre' not in sys.argv)

# --------------------------------------------------------------- jeux testes
JEU_M1_BASE = ["close_ema_dev", "returns", "range_norm", "mom_5", "high_vol_regime"]
JEU_H1_BASE = ["close_h1_dev", "rsi_14_h1", "returns_h1"]
JEU_EXT = ["taker_ratio", "ls_ratio_top"]

JEU_M1_LARGE = ["rsi_14", "returns", "vol_20", "range_norm", "open_rel",
                "high_rel", "low_rel", "close_ema_dev", "mom_5", "rsi_ok",
                "vol_rank", "high_vol_regime"]
JEU_H1_LARGE = ["rsi_14_h1", "returns_h1", "vol_20_h1", "range_norm_h1",
                "open_rel_h1", "high_rel_h1", "low_rel_h1", "close_ema_dev_h1",
                "mom_5_h1", "rsi_ok_h1", "vol_rank_h1", "high_vol_regime_h1",
                "close_h1_dev"]
JEU_LIQ_TEMPS = ["spread_rel", "heure_sin", "heure_cos"]

JEUX = {
    "10 actuel":       JEU_M1_BASE + JEU_H1_BASE + JEU_EXT,
    "8 sans Binance":  JEU_M1_BASE + JEU_H1_BASE,
    "25 M1+H1 larges": JEU_M1_LARGE + JEU_H1_LARGE,
    "27 + Binance":    JEU_M1_LARGE + JEU_H1_LARGE + JEU_EXT,
    "30 tout":         JEU_M1_LARGE + JEU_H1_LARGE + JEU_EXT + JEU_LIQ_TEMPS,
}

GRILLE = [(2.0, 1.4), (3.0, 1.4), (5.0, 1.4), (5.0, 2.0),
          (8.0, 1.4), (8.0, 2.0), (10.0, 1.4), (10.0, 2.0)]
SELECTIVITES = (0.01, 0.05)


def barrieres(hi, lo, cl, atr, idx, sl_mult, rr, sens, demi_sp, slip_e, slip_s):
    """Course TP/SL, friction complete, non-resolus CLOTURES AU MARCHE."""
    n = len(cl)
    dist_sl = sl_mult * atr[idx]
    # Entree : on paie le demi-spread ET le slippage d'entree, toujours contre.
    entree = cl[idx] + sens * (demi_sp[idx] + slip_e[idx])
    tp = entree + sens * rr * dist_sl
    sl = entree - sens * dist_sl

    sortie = np.full(len(idx), np.nan)
    motif = np.zeros(len(idx), np.int8)          # 1 = TP, -1 = SL, 0 = temps
    ouvert = np.ones(len(idx), bool)

    for h in range(1, MAX_HOLD + 1):
        j = idx + h
        valide = ouvert & (j < n)
        if not valide.any():
            break
        jj = j[valide]
        # Bruit de tick : etend high vers le haut et low vers le bas, comme
        # dans l'env. Les wicks intra-minute que l'agregation M1 masque
        # declenchent des barrieres que la bougie ne montre pas.
        h_e = hi[jj] * (1.0 + BRUIT_TICK_BPS / 1e4)
        l_e = lo[jj] * (1.0 - BRUIT_TICK_BPS / 1e4)
        if sens > 0:
            t_sl, t_tp = l_e <= sl[valide], h_e >= tp[valide]
        else:
            t_sl, t_tp = h_e >= sl[valide], l_e <= tp[valide]

        pos = np.where(valide)[0]
        k_sl = pos[t_sl]                     # les deux touches -> perdant
        k_tp = pos[t_tp & ~t_sl]
        sortie[k_sl], motif[k_sl] = sl[k_sl], -1
        sortie[k_tp], motif[k_tp] = tp[k_tp], 1
        ouvert[k_sl] = ouvert[k_tp] = False

    reste = np.where(ouvert)[0]
    if len(reste):
        sortie[reste] = cl[np.minimum(idx[reste] + MAX_HOLD, n - 1)]

    # Slippage de sortie, modele EXACT de l'env : defavorable sur SL et sur
    # sortie au temps, FAVORABLE sur TP (ordre limite traverse par le marche).
    contre = motif <= 0
    if TP_SLIPPAGE_FAVORABLE:
        sortie = sortie + sens * np.where(contre, -slip_s[idx], slip_s[idx])
    else:
        # Variante prudente : un ordre limite remplit AU PRIX, jamais mieux.
        # L'env accorde le gain de traversee ; on mesure ce qu'il vaut.
        sortie = sortie - sens * np.where(contre, slip_s[idx], 0.0)
    # Demi-spread paye en fermant, PUIS la commission.
    #
    # La commission est en dollars sur la position : fee = taux x prix x taille.
    # Le PnL etant ici par UNITE de prix, on retranche taux x prix_de_sortie —
    # c'est l'equivalent par unite, et il ne depend pas de la taille.
    pnl = sens * (sortie - entree) - demi_sp[idx] - FEE_RATE * sortie
    return pnl / dist_sl, ~ouvert


def main() -> int:
    df = pd.read_pickle(CACHE)
    n = len(df)
    hi, lo, cl = (df[c].to_numpy(np.float64) for c in ("high", "low", "close"))
    atr = np.maximum(df["atr_14"].to_numpy(np.float64), ATR_PLANCHER_FRAC * cl)
    demi_sp = (SPREAD_BPS / 1e4) * cl / 2.0
    slip_e = (SLIP_ENTREE_BPS / 1e4) * cl
    slip_s = (SLIP_SORTIE_BPS / 1e4) * cl

    a_tr, b_tr = 0, int(n * 0.55)
    a_va, b_va = int(n * 0.55), int(n * 0.70)
    i_tr = np.arange(a_tr, b_tr - MAX_HOLD, PAS)
    i_va = np.arange(a_va, b_va - MAX_HOLD, PAS)
    i_tr = i_tr[np.isfinite(atr[i_tr]) & (atr[i_tr] > 0)]
    i_va = i_va[np.isfinite(atr[i_va]) & (atr[i_va] > 0)]

    print(f"{n:,} bougies  {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    print("slippage sur TP : " + ("FAVORABLE (modele de l'env)" if TP_SLIPPAGE_FAVORABLE else "NEUTRE (ordre limite au prix, variante prudente)"))
    print(f"train {len(i_tr):,} candidats | val {len(i_va):,} | "
          f"test au-dela de {int(n*0.70):,}, INTOUCHE")
    fr = 2 * float(np.median(demi_sp)) + float(np.median(slip_e)) + float(np.median(slip_s))
    print(f"friction totale mediane {fr:.2f} $ "
          f"(spread A/R {2*float(np.median(demi_sp)):.2f} + entree "
          f"{float(np.median(slip_e)):.2f} + sortie {float(np.median(slip_s)):.2f})")
    am = float(np.median(atr))
    cl_med = float(np.median(cl))
    print(f"ATR median {am:.2f} $ -> friction = "
          + ", ".join(f"{100*fr/(m*am):.0f}% d'un stop {m:g}xATR" for m in (2, 3, 5))
          + "\n")

    # Normalisation locale : chaque jeu a ses propres colonnes, on ne peut pas
    # reutiliser le npz global qui ne couvre que les 10 actuelles.
    stats = {}
    for c in set(sum(JEUX.values(), [])):
        v = df[c].to_numpy(np.float32)
        stats[c] = (float(np.nanmean(v)), float(np.nanstd(v)) + 1e-8)

    def mat(cols, idx):
        X = np.empty((len(idx), len(cols)), np.float32)
        for k, c in enumerate(cols):
            m, s = stats[c]
            X[:, k] = np.clip((df[c].to_numpy(np.float32)[idx] - m) / s, -5, 5)
        return np.nan_to_num(X)

    # Les resultats de barrieres ne dependent PAS du jeu de features : on les
    # calcule une fois par (SL, R:R), pas une fois par jeu.
    cache_b = {}
    for sl_mult, rr in GRILLE:
        for sens in (+1, -1):
            for nom, idx in (("tr", i_tr), ("va", i_va)):
                cache_b[(sl_mult, rr, sens, nom)] = barrieres(
                    hi, lo, cl, atr, idx, sl_mult, rr, sens, demi_sp, slip_e, slip_s)

    ent = "  ".join(f"{f'top {100*q:g}%':>16}" for q in SELECTIVITES)
    print(f"{'jeu':<18} {'SL':>4} {'R:R':>4} {'WR eq':>6} {'AUC':>7}  {ent}")
    print("-" * (18 + 26 + 18 * len(SELECTIVITES)))

    resultats = []
    for nom_jeu, cols in JEUX.items():
        Xtr, Xva = mat(cols, i_tr), mat(cols, i_va)
        for sl_mult, rr in GRILLE:
            t0 = time.time()
            auc, p_va, r_va = [], [], []
            for sens in (+1, -1):
                r_t, _ = cache_b[(sl_mult, rr, sens, "tr")]
                r_v, _ = cache_b[(sl_mult, rr, sens, "va")]
                lr = LogisticRegression(max_iter=500, n_jobs=-1)
                lr.fit(Xtr, r_t > 0)
                p = lr.predict_proba(Xva)[:, 1]
                p_va.append(p)
                r_va.append(r_v)
                if len(np.unique(r_v > 0)) > 1:
                    auc.append(roc_auc_score(r_v > 0, p))
            auc_m = float(np.mean(auc))
            # Seuil d'equilibre, CORRIGE.
            #
            # La version precedente faisait (SL + friction) / (SL x (1+R:R)),
            # ce qui comptait la friction DEUX FOIS : elle est deja absorbee
            # dans le placement des barrieres, puisque celles-ci sont posees a
            # partir du prix d'entree deja degrade. Entrer plus cher ne reduit
            # pas le gain au TP, cela rend le TP plus dur a atteindre — la
            # friction se paie en TAUX DE REUSSITE, pas en montant. D'ou des
            # lignes affichant une esperance positive sous un seuil pretendu
            # inatteignable, ce qui etait impossible.
            #
            # On raisonne donc sur les gains REELS en R, tels que la simulation
            # les produit : un TP rapporte R:R plus le solde de slippage, un SL
            # coute 1 plus le spread et le slippage de sortie.
            d = sl_mult * am
            r_tp = rr + (SLIP_SORTIE_BPS - SPREAD_BPS / 2) / 1e4 * cl_med / d
            r_sl = 1.0 + (SLIP_SORTIE_BPS + SPREAD_BPS / 2) / 1e4 * cl_med / d
            wr_eq = r_sl / (r_tp + r_sl)

            bouts = ""
            for q in SELECTIVITES:
                ers, wrs, nn = [], [], 0
                for k in (0, 1):
                    m = p_va[k] >= np.quantile(p_va[k], 1.0 - q)
                    ers.append(r_va[k][m].mean())
                    wrs.append((r_va[k][m] > 0).mean())
                    nn += int(m.sum())
                er, wr = float(np.mean(ers)), float(np.mean(wrs))
                # t de Student sur les deux cotes empiles
                tous = np.concatenate([r_va[k][p_va[k] >= np.quantile(p_va[k], 1-q)]
                                       for k in (0, 1)])
                t = tous.mean() / (tous.std() / np.sqrt(len(tous)) + 1e-12)
                bouts += f"  {er:>+7.4f} t{t:>+5.1f}"
                resultats.append((er, t, nom_jeu, sl_mult, rr, q, wr, wr_eq, nn, auc_m))
            print(f"{nom_jeu:<18} {sl_mult:>4.1f} {rr:>4.1f} {100*wr_eq:>5.1f}% "
                  f"{auc_m:>7.4f} {bouts}   ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

    print()
    print("E[R] = esperance par unite de risque APRES selection, friction complete.")
    print("t    = en ecarts-types. Au-dela de +2 le resultat est distinguable de")
    print("       zero ; en deca, il ne l'est pas, quelle que soit sa valeur.")
    print()
    resultats.sort(key=lambda x: -x[0])
    print("CLASSEMENT")
    for er, t, nom_jeu, sl_mult, rr, q, wr, wr_eq, nn, auc_m in resultats[:10]:
        marque = " *" if t > 2 else "  "
        print(f"{marque}{nom_jeu:<18} SL {sl_mult:>4.1f} R:R {rr:>3.1f} top {100*q:>4.1f}%  "
              f"E[R] {er:+.4f} (t {t:+.1f})  WR {100*wr:.1f}% / {100*wr_eq:.1f}% requis  "
              f"{nn} trades  AUC {auc_m:.4f}")
    print()
    print("* = distinguable de zero. Sans etoile, le chiffre est du bruit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
