"""Le stop, compare TRADE PAR TRADE sur les memes dates d'entree.

POURQUOI C'EST LE BON TEST. Deux largeurs de stop jouees aux MEMES dates
subissent le meme marche : la variance du marche, qui est enorme, s'annule
dans la difference. Comparer deux moyennes independantes, c'est mesurer un
ecart de 0.05 R sous un bruit de 2 R ; comparer les differences appariees,
c'est mesurer le meme ecart sous le bruit de ce que la largeur change
vraiment. Le gain de puissance est d'un ordre de grandeur.

L'INCERTITUDE SE CALCULE SUR LES OCCASIONS, pas sur les phases. Les entrees
sont espacees de PAS barres, choisi au-dessus de la duree mediane du stop le
plus large, pour que deux trades consecutifs ne se recouvrent pas. Chaque
entree est alors une occasion, et l'ecart-type des differences divise par la
racine de leur nombre est une vraie barre d'erreur.

LE TEST N'EST PAS OUVERT : on s'arrete a 85 % de l'historique.
"""
import numpy as np
import pandas as pd

import cibles as C
import training as T

STOPS = (6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0)
REF = 8.0
PAS = 2016          # 7 jours de M5, au-dessus de la duree mediane du 20x


class Cfg:
    def __init__(self, sl):
        self.atr_sl_mult = sl
        self.use_tp = False
        self.atr_tp_mult = 0.0
        self.use_be_trail = True
        self.atr_be_mult = 1e9
        self.atr_trail_mult = 1.5 * sl
        self.atr_trail_dist = 1.5 * sl


def main() -> int:
    df = T.load_mt5_data(T.PPOConfig())
    n = len(df)
    va = C.borne_etude(n)
    t = pd.to_datetime(df["time"])
    print(f"fenetre : {t.iloc[0]:%Y-%m-%d} -> {t.iloc[va-1]:%Y-%m-%d}   "
          f"test intact a partir du {t.iloc[va]:%Y-%m-%d}")

    idx = np.arange(2000, va - C.BORNE_DEFAUT - 2, PAS)
    an = pd.to_datetime(df["time"]).dt.year.to_numpy()[idx]
    print(f"{len(idx):,} dates d'entree espacees de {PAS*5/60/24:.1f} jours\n")

    # Part symetrique par date : (achat + vente) / 2, la derive deduite.
    sym, duree = {}, {}
    for sl in STOPS:
        ra, rv = C.rendements(df, idx, Cfg(sl))
        sym[sl] = (ra + rv) / 2.0
    ok = np.all([np.isfinite(sym[s]) for s in STOPS], axis=0)
    print(f"{int(ok.sum()):,} dates ou TOUTES les largeurs se resolvent "
          f"— seules celles-la sont comparables\n")
    an = an[ok]

    print(f"{'stop':>6} {'E[R] sym':>10} {'+/-':>8} {'sigma':>7}   "
          f"{'ecart vs 8x':>12} {'+/-':>8} {'sigma':>7} {'ann +':>7}")
    print("-" * 78)
    ref = sym[REF][ok]
    for sl in STOPS:
        y = sym[sl][ok]
        se = y.std(ddof=1) / np.sqrt(len(y))
        if sl == REF:
            print(f"{sl:>5g}x {y.mean():>+10.4f} {se:>8.4f} "
                  f"{y.mean()/se:>+7.1f}   {'(reference)':>12}")
            continue
        d = y - ref
        sed = d.std(ddof=1) / np.sqrt(len(d))
        pa = sum(1 for a in sorted(set(an)) if d[an == a].mean() > 0)
        print(f"{sl:>5g}x {y.mean():>+10.4f} {se:>8.4f} {y.mean()/se:>+7.1f}   "
              f"{d.mean():>+12.4f} {sed:>8.4f} {d.mean()/sed:>+7.1f} "
              f"{pa:>4}/{len(set(an))}")

    print("\nLECTURE. La colonne de gauche compare des moyennes independantes :")
    print("son bruit est celui du marche. La colonne de droite compare les")
    print("MEMES dates : son bruit est celui de la largeur seule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
