"""L'equite barre par barre, dans l'ordre, avec toutes les contraintes reelles.

POURQUOI CE FICHIER EXISTE. Tous les chiffres de geometrie de ce depot sont
des R PAR TRADE additionnes : "+12.5 R par an" suppose des trades independants
pris a 1 R chacun. Sous concurrence ils ne le sont pas — deux positions de meme
sens ouvertes a une heure d'ecart correlent a 0.72 — et le budget de risque
force a reduire la taille. Le rendement annuel REEL est donc inferieur a ce que
l'addition suggere, et le creux, lui, est superieur.

Additionner des R est une mesure de la GEOMETRIE. Ce fichier mesure le
PORTEFEUILLE : une seule equite, qui compose, qui subit les creux, et qui peut
mourir.

IL N'Y A PAS DE SECOND SIMULATEUR. Le trade est joue par
`BTCTradingEnvDiscrete` lui-meme, en un unique episode long — donc le lot
minimum, la marge, l'appel de marge, la ruine et le stop suiveur sont ceux qui
tourneront. Ecrire un simulateur de portefeuille a cote aurait cree deux
descriptions du meme trade, qui doivent s'accorder par convention : c'est la
faute que ce depot passe son temps a payer.

LES ENTREES SONT NEUTRES. Sens tire a pile ou face, plusieurs graines, aucun
modele : on mesure ce que la structure produit, pas ce qu'un reseau y ajoute.
Un modele ne peut que deplacer ce resultat, pas en changer la nature.

LE TEST N'EST PAS OUVERT : on s'arrete a `cibles.borne_etude`.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import cibles as C
import training as T
from saint_core import FEATURE_COLS

BUDGETS = (0.03, 0.05, 0.10, 0.20)
GRAINES = (1, 2)
TAUX_ENTREE = 1.0      # on tente une entree a chaque barre ; le solde arbitre


def parcours(data, cfg, graine, n_bars):
    """Une equite continue sur toute la fenetre, en un seul episode."""
    np.random.seed(graine)
    env = T.BTCTradingEnvDiscrete(data, cfg)
    T.reset_au_depart(env, cfg.lookback + 1)
    env.end_idx = min(cfg.lookback + 1 + n_bars, data.length - 2)
    rng = np.random.default_rng(graine)

    eq = np.empty(n_bars, dtype=np.float64)
    ouv = np.empty(n_bars, dtype=np.int32)
    k = 0
    raison = None
    while k < n_bars:
        u = rng.random()
        if u < TAUX_ENTREE / 2:
            a = 0
        elif u < TAUX_ENTREE:
            a = 1
        else:
            a = 2
        _, _, done, _, info = env.step(a)
        eq[k] = env.capital + env._latent_at_bid(
            env.data.close[max(env.idx - 1, 0)])
        ouv[k] = info["n_positions"]
        k += 1
        if done:
            raison = info.get("done_reason")
            break
    return eq[:k], ouv[:k], raison


def creux(eq):
    pic = np.maximum.accumulate(eq)
    return float((1.0 - eq / np.maximum(pic, 1e-9)).max())


def main() -> int:
    cfg0 = T.PPOConfig()
    df = T.load_mt5_data(cfg0)
    n = len(df)
    va = C.borne_etude(n)
    t = pd.to_datetime(df["time"])
    ans = (t.iloc[va - 1] - t.iloc[0]).days / 365.25
    stats = T.compute_and_save_global_norm_stats(df.iloc[:va], FEATURE_COLS,
                                                 path=None)
    data = T.MarketData(df.iloc[:va].reset_index(drop=True), FEATURE_COLS, stats)
    del df

    n_bars = min(int(sys.argv[1]) if len(sys.argv) > 1 else data.length - 10,
                 data.length - 10)
    annees = n_bars * 5 / 60 / 24 / 365.25
    print(f"{t.iloc[0]:%Y-%m-%d} -> {t.iloc[va-1]:%Y-%m-%d}, test intouche")
    print(f"stop {cfg0.atr_sl_mult:g}xATR, trailing "
          f"{cfg0.atr_trail_mult/cfg0.atr_sl_mult:.1f} R, lot min "
          f"{cfg0.lot_min}, capital {cfg0.initial_capital:.0f}$")
    print(f"{n_bars:,} barres parcourues = {annees:.1f} ans\n")

    print(f"{'budget':>7} {'graine':>7} {'positions':>10} {'gain':>9} "
          f"{'par an':>8} {'creux max':>10} {'fin':>18}")
    print("-" * 76)
    for b in BUDGETS:
        cfg = T.PPOConfig()
        cfg.budget_risque = b
        cfg.episode_length = n_bars + 10
        for g in GRAINES:
            t0 = time.perf_counter()
            eq, ouv, raison = parcours(data, cfg, g, n_bars)
            if len(eq) < 10:
                print(f"{100*b:>6.0f}% {g:>7}   (episode vide)")
                continue
            a_reel = len(eq) * 5 / 60 / 24 / 365.25
            gain = eq[-1] / cfg.initial_capital - 1.0
            par_an = (1.0 + gain) ** (1.0 / max(a_reel, 1e-9)) - 1.0 if gain > -1 else -1.0
            print(f"{100*b:>6.0f}% {g:>7} {int(np.median(ouv)):>10} "
                  f"{100*gain:>+8.0f}% {100*par_an:>+7.0f}% {100*creux(eq):>9.0f}% "
                  f"{(raison or 'fenetre finie'):>18}   [{time.perf_counter()-t0:.0f}s]")

    print("\nLECTURE. 'creux max' est le pire recul de l'equite depuis son")
    print("sommet, dans l'ordre chronologique — pas un tirage melange. C'est")
    print("lui qui dit si le systeme est jouable, et il ne se deduit pas des")
    print("R par trade additionnes. Une fin en 'appel_de_marge' ou en")
    print("'solde_insuffisant' veut dire que le compte est mort avant la fin")
    print("de la fenetre.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
