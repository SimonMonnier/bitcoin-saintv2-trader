"""Les 47 colonnes Ichimoku apportent-elles quelque chose ? Protocole purge.

CE QUI EST COMPARE. Deux jeux de colonnes sur EXACTEMENT les memes lignes, les
memes occasions, les memes blocs, les memes phases :

    reference   56 colonnes   prix H1, contexte H4, flux, temps, range
    complet    103 colonnes   les memes + 47 couvrant toutes les strategies
                              Ichimoku : Chikou-Span, replis sur plat Kijun,
                              cassures de plus hauts, espace tradable, boussole
                              du nuage, 14 structures en chandeliers.

POURQUOI CE PROTOCOLE ET PAS LA VALIDATION. Mesure du 2026-09-15, repetee sur
deux methodes sans rapport : tout ce qu'on choisit sur la fenetre de validation
se transfere NEGATIVEMENT au test. Le checkpoint choisi sur elle coute 3.3
points, la selectivite de TabM choisie sur elle coute 4 points. Valider un jeu
de features de la meme facon repeterait la faute.

Le seul protocole de ce depot dont une conclusion ait tenu est celui du banc :
walk-forward par blocs, entrees espacees de la duree de detention pour qu'aucun
resultat ne chevauche le suivant, moyenne sur les phases, et surtout LECTURE
BLOC PAR BLOC. Un gain qui n'apparait que dans une periode sur six est un
accident — c'est ainsi que le gain apparent de TabM s'est effondre la premiere
fois.

CE QUE LA COMPARAISON APPARIEE APPORTE. Les deux jeux voient les memes barres
aux memes instants, donc la difference entre eux ne contient pas la difference
de periode. C'est la seule facon de lire un ecart de 0.02 R quand l'ecart-type
entre blocs en vaut 0.05.

LE RISQUE QU'ON MESURE ICI, ET QUI N'EST PAS THEORIQUE. Passer de 56 a 103
colonnes sur 55 000 barres d'entrainement peut tres bien DEGRADER le resultat :
plus de colonnes, c'est plus de facons de trouver une regularite qui n'existe
pas. Un resultat negatif serait donc une reponse, pas un echec.

    python mesure_features_ichimoku.py
"""

import warnings

import numpy as np
import pandas as pd

import banc_rendement_net as B
import mesure_features as MF
from features_ichimoku import ajoute_features_ichimoku
from saint_core import FEATURE_COLS, ATR_PLANCHER_FRAC

warnings.filterwarnings("ignore")

CACHE = "data_cache_BTCUSD_H1.pkl"   # le jeu "complet" a fusionne avec lui
SL_MULT, RR, PAS = 2.0, 2.0, 24
N_BLOCS = 6
PHASES = list(range(0, 24, 3))          # 8 phases
FRAC_CALIB = 0.30
MARGES = [0.0, 0.10, 0.20]
MIN_TR, MIN_VA = 500, 80


def prepare():
    df = pd.read_pickle(CACHE)
    cols_ich = [c for c in df.columns if c.startswith("ich_")]
    jeux = {"reference": list(FEATURE_COLS),
            "complet": list(FEATURE_COLS) + cols_ich}
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=jeux["complet"] + ["atr_14"]).reset_index(drop=True)
    n = len(df)
    fin = int(n * 0.70)
    d = {"n": n, "fin": fin,
         "hi": df["high"].to_numpy(np.float64),
         "lo": df["low"].to_numpy(np.float64),
         "cl": df["close"].to_numpy(np.float64)}
    d["atr"] = np.maximum(df["atr_14"].to_numpy(np.float64),
                          ATR_PLANCHER_FRAC * d["cl"])
    d["ds"] = (B.SPREAD_BPS / 1e4) * d["cl"] / 2.0
    d["se"] = (B.SLIP_ENTREE_BPS / 1e4) * d["cl"]
    d["ss"] = (B.SLIP_SORTIE_BPS / 1e4) * d["cl"]
    d["bornes"] = np.linspace(int(fin * 0.35), fin, N_BLOCS + 1).astype(int)
    d["X"] = {k: np.nan_to_num(np.column_stack(
        [df[c].to_numpy(np.float32) for c in v])) for k, v in jeux.items()}
    d["jeux"] = {k: len(v) for k, v in jeux.items()}
    return d


def cibles(d, idx):
    MF.MAX_HOLD = PAS
    out = []
    for sens in (1, -1):
        r, _ = MF.barrieres(d["hi"], d["lo"], d["cl"], d["atr"], idx,
                            SL_MULT, RR, sens, d["ds"], d["se"], d["ss"])
        out.append(r)
    return out[0], out[1]


def main() -> int:
    d = prepare()
    print(f"{d['n']:,} barres  |  {N_BLOCS} blocs  |  {len(PHASES)} phases  |  "
          f"entrees tous les {PAS} barres")
    for k, v in d["jeux"].items():
        print(f"  {k:>10} : {v} colonnes")
    print("  fenetre de test (au-dela de 70 %) JAMAIS touchee\n")

    fab = B.modele_tabm()
    # (jeu, marge, bloc) -> liste des E[R] par phase, et le repere au hasard.
    res = {j: {m: {b: [] for b in range(N_BLOCS)} for m in MARGES}
           for j in d["jeux"]}
    hasard = {b: [] for b in range(N_BLOCS)}

    for ph in PHASES:
        for b in range(N_BLOCS):
            a_va, b_va = d["bornes"][b], d["bornes"][b + 1]
            i_all = np.arange(ph, a_va - 2 * PAS, PAS)
            i_all = i_all[np.isfinite(d["atr"][i_all]) & (d["atr"][i_all] > 0)]
            if len(i_all) < MIN_TR:
                continue
            coupe = int(len(i_all) * (1 - FRAC_CALIB))
            i_tr, i_ca = i_all[:coupe], i_all[coupe:]
            i_va = np.arange(a_va + ph, b_va - PAS, PAS)
            i_va = i_va[np.isfinite(d["atr"][i_va]) & (d["atr"][i_va] > 0)]
            if len(i_va) < MIN_VA:
                continue

            rb_tr, rs_tr = cibles(d, i_tr)
            rb_va, rs_va = cibles(d, i_va)
            hasard[b].append(0.5 * (rb_va.mean() + rs_va.mean()))

            for jeu, X in d["X"].items():
                p_ca, p_va = {}, {}
                for cle, y in (("b", rb_tr), ("s", rs_tr)):
                    mod = fab().fit(X[i_tr], y)
                    p_ca[cle] = mod.predict(X[i_ca])
                    p_va[cle] = mod.predict(X[i_va])
                gain = np.where(p_va["b"] >= p_va["s"], rb_va, rs_va)
                best = np.maximum(p_va["b"], p_va["s"])
                for m in MARGES:
                    sel = best >= m
                    res[jeu][m][b].append(gain[sel].mean() if sel.sum() >= 10
                                          else np.nan)

    base_b = [np.mean(v) for b, v in hasard.items() if v]
    print(f"repere au hasard, par bloc : "
          + "  ".join(f"{x:+.3f}" for x in base_b))
    print(f"                   ensemble : {np.mean(base_b):+.4f} R\n")

    print(f"{'marge':>7} {'jeu':>11} {'E[R]':>9} {'err-type':>9} "
          f"{'blocs +':>9}   {'ecart au reference':>19} {'blocs mieux':>12}")
    print("-" * 86)
    for m in MARGES:
        moy = {}
        for jeu in d["jeux"]:
            par_bloc = [np.nanmean(res[jeu][m][b]) for b in range(N_BLOCS)
                        if res[jeu][m][b] and not np.all(np.isnan(res[jeu][m][b]))]
            moy[jeu] = par_bloc
        for jeu in ("reference", "complet"):
            v = moy[jeu]
            if not v:
                continue
            err = np.std(v, ddof=1) / np.sqrt(len(v))
            ligne = (f"{m:+7.2f} {jeu:>11} {np.mean(v):+9.4f} {err:9.4f} "
                     f"{sum(1 for x in v if x > 0):5d}/{len(v):<3d}")
            if jeu == "complet" and moy["reference"]:
                # DIFFERENCE APPARIEE : son erreur-type se calcule sur les
                # differences elles-memes, pas sur les deux moyennes separement.
                # Les deux jeux voient les memes barres, donc la variance de
                # periode — qui domine tout le reste ici — se soustrait. La lire
                # sur les moyennes separees donnerait une barre plusieurs fois
                # trop large et masquerait un effet reel.
                paires = [c - r for c, r in zip(v, moy["reference"])]
                e = np.std(paires, ddof=1) / np.sqrt(len(paires))
                ligne += (f"   {np.mean(paires):+10.4f} +/- {e:.4f} "
                          f"{sum(1 for x in paires if x > 0):5d}/{len(paires):<5d}")
            print(ligne)
        print()

    print("La colonne 'ecart au reference' est APPARIEE bloc a bloc : les deux")
    print("jeux voient les memes barres, donc la difference de periode ne s'y")
    print("melange pas. 'blocs mieux' est le test des signes — 6/6 vaut 0.016.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
