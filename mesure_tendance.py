# -*- coding: utf-8 -*-
"""QUEL DETECTEUR DE TENDANCE DIT VRAIMENT QUEL COTE PRIVILEGIER ?

LE PIEGE, ET IL EST ENORME SUR CETTE FENETRE. L'or a monte : acheter au hasard
rapporte +0.338 R, vendre au hasard coute -0.419 R. Un detecteur qui
repondrait "haussier" en permanence paraitrait donc excellent sans rien
detecter du tout. Comparer le rendement des achats en regime haussier a zero,
ou meme a la moyenne des ventes, ne prouve RIEN.

CE QU'IL FAUT MESURER A LA PLACE : l'ECART entre les deux sens, et comment il
bouge d'un regime a l'autre.

    ecart(regime) = R moyen des ACHATS - R moyen des VENTES, dans ce regime

Sur toute la fenetre cet ecart vaut +0.757 R par construction (0.338 + 0.419),
et il refletent la derive, pas un tri. Un detecteur utile est un detecteur ou
l'ecart CHANGE : large en haussier, et surtout retreci — voire inverse — en
baissier. C'est la difference des ecarts qui porte l'information, et c'est
elle qu'on teste.

SI L'ECART EST LE MEME DANS LES DEUX REGIMES, le detecteur n'a rien detecte :
il a decoupe la fenetre en deux morceaux ou la meme derive s'applique.

LE BRUIT. La fenetre porte ~930 occasions INDEPENDANTES (duree mediane d'un
trade : 371 barres). Deux occasions espacees d'une heure partagent presque
toute leur fenetre de resultat ; les compter comme independantes surestimerait
la certitude d'un facteur dix. Les sigma affiches tiennent compte de cela.

ET SEPT DETECTEURS, C'EST SEPT ESSAIS : le maximum de sept tirages de bruit
pur vaut deja ~1.9 sigma. Le seuil est rappele en bas de table.

LA FENETRE DE TEST N'EST PAS LUE.

    python mesure_tendance.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

import cibles as C
import instruments as I
import training as T

PAS = 12


def main() -> int:
    cfg = T.PPOConfig()
    df = pd.read_pickle(I.INSTRUMENTS[cfg.symbol]["cache"])
    n_fin = int(len(df) * 0.55) + int(len(df) * 0.15)
    idx = np.arange(cfg.lookback + 1, n_fin - C.BORNE_DEFAUT - 2, PAS)

    ra, rv, ta, _ = C.rendements(df, idx, cfg, indicateur=False, durees=True)
    a = np.asarray(ra, float)
    v = np.asarray(rv, float)
    duree = float(np.nanmedian(np.asarray(ta, float)))
    n_indep = (n_fin - cfg.lookback) / max(duree, 1.0)
    ok = np.isfinite(a) & np.isfinite(v)

    close = df["close"].to_numpy(float)
    g = df.iloc[idx].reset_index(drop=True)

    def mom(n):
        prec = np.maximum(idx - n, 0)
        return (close[idx] - close[prec]) / np.maximum(close[prec], 1e-9)

    def sma(n):
        m = pd.Series(close).rolling(n).mean().to_numpy()
        return (close[idx] - m[idx]) / np.maximum(m[idx], 1e-9)

    detecteurs = [
        ("nuage : orientation (le livre)", g["ich_regime"].to_numpy(float)),
        ("nuage H4 : orientation", g["ich_regime_h4"].to_numpy(float)),
        ("prix vs nuage (au-dessus/dessous)", g["ich_pos_kumo"].to_numpy(float)),
        ("pente de Kijun", g["ich_pente_kijun"].to_numpy(float)),
        ("momentum 1 jour (288 barres)", mom(288)),
        ("momentum 1 semaine (2 016)", mom(2016)),
        ("momentum 1 mois (8 640)", mom(8640)),
        ("prix vs moyenne 1 semaine", sma(2016)),
        ("prix vs moyenne 1 mois", sma(8640)),
    ]

    e_global = float(a[ok].mean() - v[ok].mean())
    print(f"{len(idx):,} occasions, ~{n_indep:,.0f} independantes")
    print(f"ACHAT au hasard {float(a[ok].mean()):+.3f} R  |  "
          f"VENTE au hasard {float(v[ok].mean()):+.3f} R")
    print(f"ECART GLOBAL entre les deux sens : {e_global:+.3f} R "
          f"— c'est la derive de l'or, pas un tri")
    print(f"La fenetre de test n'est pas lue.\n")

    print(f"{'detecteur':<36} {'regime':>9} {'n':>7} {'achat':>8} "
          f"{'vente':>8} {'ecart':>8} {'vs global':>10} {'sigma':>7}")
    print("-" * 100)
    meilleurs = []
    for nom, x in detecteurs:
        fini = ok & np.isfinite(x)
        # Terciles : le detecteur decoupe en haussier / neutre / baissier,
        # sans qu'on ait a choisir un seuil — donc rien a sur-ajuster.
        q1, q2 = np.nanquantile(x[fini], [1 / 3, 2 / 3])
        lignes = []
        for lab, m in (("haussier", fini & (x >= q2)),
                       ("baissier", fini & (x < q1))):
            if m.sum() < 200:
                continue
            ma, mv = float(a[m].mean()), float(v[m].mean())
            e = ma - mv
            k = max(m.sum() * n_indep / max(fini.sum(), 1), 1.0)
            se = np.sqrt(a[m].std() ** 2 + v[m].std() ** 2) / np.sqrt(k)
            lignes.append((lab, int(m.sum()), ma, mv, e, (e - e_global) / se))
        if len(lignes) < 2:
            continue
        # CE QUI COMPTE : l'ecart entre les deux regimes, pas chaque regime.
        sep = lignes[0][4] - lignes[1][4]
        k = max(min(lignes[0][1], lignes[1][1]) * n_indep / max(fini.sum(), 1), 1.0)
        se_sep = np.sqrt(sum(a[fini].std() ** 2 + v[fini].std() ** 2
                             for _ in (0, 1))) / np.sqrt(k)
        sg = sep / se_sep if se_sep > 0 else 0.0
        meilleurs.append((abs(sg), nom, sep))
        for i, (lab, n, ma, mv, e, s) in enumerate(lignes):
            t = nom if i == 0 else ""
            print(f"{t:<36} {lab:>9} {n:>7,} {ma:>+8.3f} {mv:>+8.3f} "
                  f"{e:>+8.3f} {e-e_global:>+10.3f} {s:>+7.1f}")
        print(f"{'':36} {'SEPARATION haussier - baissier':>45} "
              f"{sep:>+8.3f} {'':>10} {sg:>+7.1f}")
        print()
    print("-" * 100)
    if meilleurs:
        m = max(meilleurs)
        print(f"meilleure separation : {m[1]}  ->  {m[2]:+.3f} R a {m[0]:.1f} sigma")
    print(f"ce qu'un bruit pur produirait sur {len(detecteurs)} essais : "
          f"~1.9 sigma")
    print("\nLECTURE. `ecart` est ce que l'achat rapporte de plus que la vente")
    print("DANS ce regime. `SEPARATION` est la difference entre les deux")
    print("regimes : c'est la seule ligne qui dise si le detecteur detecte")
    print("quelque chose, parce qu'elle est insensible a la derive de l'or.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
