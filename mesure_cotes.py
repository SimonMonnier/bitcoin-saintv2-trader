# -*- coding: utf-8 -*-
"""Reintroduire les ventes changerait-il quelque chose ? Le hasard, par cote.

POURQUOI CETTE MESURE. Le journal affiche depuis toujours un « sommet du
hasard » a +0.569 R : une occasion prise AU HASARD, a l'achat, rapporte plus
d'un demi R. Ce n'est pas du talent, c'est la DERIVE — l'or est passe
d'environ 1 500 a 4 376 $ sur la fenetre. Le modele, lui, sort a +1.113 :
il ajoute +0.54 R par-dessus un +0.57 R gratuit.

LA QUESTION QUE CELA POSE. Si la moitie du rendement vient de la tendance,
que reste-t-il quand elle s'inverse ? Les ventes sont la seule reponse — mais
elles partent peut-etre avec un handicap d'un R complet, et il faut le savoir
AVANT de les reintroduire.

CE QUE LES SHORTS N'APPORTENT PAS, et c'est ce qui les distingue d'un second
instrument : AUCUNE occasion independante. Sur la meme barre, l'achat et la
vente sont le meme evenement vu a l'envers — `R_vente ~ -R_achat` a la
friction pres, et la cible du depot le dit deja en prenant la part symetrique
`(r_achat - r_vente) / 2`. Le BTC fait passer 930 occasions a 1 515 ; les
ventes les laissent a 930.

CE QU'ELLES APPORTENT PEUT-ETRE : de quoi continuer a gagner quand la derive
change de signe. Un argument de ROBUSTESSE, pas de rendement.

CE QUE LA MESURE EXISTANTE NE DIT PAS. Le tableau des trois folds — shorts a
-286.9, -108.5, -435.8 $ — vient d'un modele JAMAIS ENTRAINE A VENDRE : le
bug de masque de cote, ou `decide_avec_barres` laissait passer des ventes
dans un run long-only. Il mesure des ventes au hasard, pas une strategie.

    python mesure_cotes.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import cibles as CIB
import training as T

PAS = 12          # une occasion par heure, comme la grille de classement
SELECTIVITE = 0.05


def main() -> int:
    cfg = T.PPOConfig()
    df = pd.read_pickle("data_cache_XAUUSD_M5.pkl")
    print(f"\n{len(df):,} barres M5   "
          f"{pd.to_datetime(df['time'].iloc[0]):%Y-%m-%d} -> "
          f"{pd.to_datetime(df['time'].iloc[-1]):%Y-%m-%d}")
    prix0 = float(df["close"].iloc[0])
    prix1 = float(df["close"].iloc[-1])
    print(f"or : {prix0:,.0f} -> {prix1:,.0f} $   "
          f"soit x{prix1/prix0:.2f} sur la fenetre")

    idx = np.arange(cfg.lookback,
                    len(df) - CIB.BORNE_DEFAUT - 2, PAS)
    print(f"{len(idx):,} occasions, une toutes les {PAS*5} min")
    print("\ncalcul des deux cotes sous la regle deployee "
          f"(stop {cfg.atr_sl_mult:.0f} x ATR)...", flush=True)
    ra, rv = CIB.rendements(df, idx, cfg)
    ok = np.isfinite(ra) & np.isfinite(rv)
    ra, rv = ra[ok], rv[ok]
    print(f"{ok.sum():,} occasions resolues dans la borne\n")

    print("=" * 76)
    print("LE HASARD, PAR COTE")
    print("=" * 76)
    for nom, r in (("ACHAT", ra), ("VENTE", rv)):
        print(f"  {nom:<6} moyenne {np.mean(r):>+7.3f} R   "
              f"mediane {np.median(r):>+7.3f} R   "
              f"part gagnante {100*np.mean(r > 0):>5.1f} %")

    # LA SYMETRIE EST-ELLE EXACTE ? Si R_vente valait exactement -R_achat, la
    # somme serait nulle. Les barrieres ne sont pas symetriques — le trailing
    # suit dans un seul sens et le stop se declenche sur des meches
    # differentes — donc l'ecart mesure ce que la geometrie ajoute ou retire.
    print()
    print(f"  somme des deux moyennes : {np.mean(ra) + np.mean(rv):>+7.3f} R")
    print(f"  correlation achat/vente : {np.corrcoef(ra, rv)[0,1]:>+7.3f}")
    print("  (a -1.000 les deux cotes sont le meme evenement a l'envers,")
    print("   donc une vente n'ajoute AUCUNE occasion independante)")

    print()
    print("=" * 76)
    print("LE HANDICAP, ET CE QU'IL FAUDRAIT POUR LE COMBLER")
    print("=" * 76)
    ecart = float(np.mean(ra) - np.mean(rv))
    print(f"  un achat au hasard bat une vente au hasard de {ecart:+.3f} R")
    print()
    print("  Le modele long-only sort a +1.113 R sur son sommet, contre")
    print(f"  {np.mean(ra):+.3f} R pour un achat au hasard : il ajoute")
    print(f"  {1.113 - float(np.mean(ra)):+.3f} R de selection.")
    print()
    print(f"  Pour qu'une strategie SHORT egale ce resultat, il lui faudrait")
    print(f"  ajouter {1.113 - float(np.mean(rv)):+.3f} R par-dessus son propre")
    print("  hasard — soit bien davantage, et sur la meme quantite de donnees.")

    # PAR SOUS-PERIODE : la derive n'est pas uniforme, et c'est tout l'enjeu.
    # Si les ventes ne sont perdantes QUE dans les periodes de hausse, elles
    # redeviennent utiles ailleurs — et c'est un argument de robustesse.
    print()
    print("=" * 76)
    print("PAR SOUS-PERIODE — la derive n'est pas uniforme")
    print("=" * 76)
    t = pd.to_datetime(df["time"].to_numpy()[idx][ok])
    n_bl = 6
    bornes = np.linspace(0, len(ra), n_bl + 1).astype(int)
    print(f"  {'periode':<26} {'or':>8} {'achat':>9} {'vente':>9} "
          f"{'la vente bat-elle':>18}")
    for i in range(n_bl):
        a, b = bornes[i], bornes[i + 1]
        if b - a < 50:
            continue
        deb, fin = t[a], t[b - 1]
        # La variation de l'or sur la sous-periode, pour lire le reste.
        i0 = idx[ok][a]
        i1 = idx[ok][b - 1]
        var = float(df["close"].iloc[i1] / df["close"].iloc[i0] - 1.0)
        ma, mv = float(np.mean(ra[a:b])), float(np.mean(rv[a:b]))
        print(f"  {deb:%Y-%m} a {fin:%Y-%m}          {100*var:>+7.0f}% "
              f"{ma:>+9.3f} {mv:>+9.3f} {'OUI' if mv > ma else 'non':>18}")

    print()
    print("=" * 76)
    print("""
CE QUE CELA VEUT DIRE.

  Si la vente perd PARTOUT, y compris dans les sous-periodes ou l'or baisse,
  alors la geometrie elle-meme lui est defavorable — stop, trailing et
  borne de resolution ne sont pas symetriques — et la reintroduire coute
  sans rien rapporter.

  Si la vente gagne dans les sous-periodes baissieres, alors le long-only
  paie sa robustesse : il ne peut rien faire quand la derive s'inverse, et
  les +0.57 R gratuits qu'il encaisse aujourd'hui ne sont pas acquis.

  DANS LES DEUX CAS, LES VENTES N'AJOUTENT PAS D'OCCASIONS. Elles changent
  ce que le modele peut faire, pas ce qu'on peut mesurer.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
