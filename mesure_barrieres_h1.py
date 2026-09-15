"""Le taux de base des barrieres, avant toute decision.

CE QUE CE FICHIER MESURE, ET POURQUOI IL ARRIVE SI TARD. Mesure du 2026-09-15
sur neuf ans de H1 : TabM bat le hasard de +0.085 R dans SIX periodes sur six
(test des signes p = 0.016), mais son E[R] absolu reste a +0.034 +/- 0.031,
donc indistinguable de zero. La ligne du dessus explique tout :

    entrer au hasard, SL 2xATR / TP 4xATR / 24 h :  -0.051 R

Le modele recupere +0.085 R sur un jeu qui en coute -0.051 avant qu'il ait
ouvert la bouche. Ce -0.051 n'est pas de la friction — elle ne vaut que
0.073 R par ATR ici, soit 0.037 R sur un stop de 2 ATR aller-retour. Le reste
vient de la GEOMETRIE : un TP a 4 ATR n'est pas atteint deux fois moins
souvent qu'un SL a 2 ATR, et une course non resolue au bout de 24 h est
cloturee au marche, ce qui a son propre biais.

Aucun modele ne repare une geometrie defavorable ; il ne fait que la
compenser. On cherche donc la configuration dont le taux de base est le plus
proche de zero, pour que l'avantage mesure du modele tombe entier dans le
resultat au lieu d'etre absorbe.

CE QUI EST MESURE ICI EST ENTIEREMENT SANS MODELE : on entre a chaque
occasion, un coup a l'achat un coup a la vente, et on lit ce que la geometrie
rend. Il n'y a donc rien a surajuster, et la fenetre de test reste intouchee.

    python mesure_barrieres_h1.py
"""

import warnings

import numpy as np

import banc_rendement_net as B
import mesure_features as MF

warnings.filterwarnings("ignore")

SL_MULTS = (1.0, 1.5, 2.0, 3.0)
RRS = (1.0, 1.5, 2.0, 3.0)
HOLDS = (12, 24, 48, 96)


def base(d, sl_mult, rr, hold, phases):
    """E[R] au hasard, part resolue, et E[R] par bloc — sans aucun modele."""
    MF.MAX_HOLD = hold
    par_bloc = [[] for _ in range(B.N_BLOCS)]
    resolus, tous = [], []
    for ph in phases:
        for b in range(B.N_BLOCS):
            a, bb = d["bornes"][b], d["bornes"][b + 1]
            idx = np.arange(a + ph, bb - hold, max(hold, 1))
            idx = idx[np.isfinite(d["atr"][idx]) & (d["atr"][idx] > 0)]
            if len(idx) < 30:
                continue
            rs = []
            for sens in (1, -1):
                r, motif = MF.barrieres(d["hi"], d["lo"], d["cl"], d["atr"],
                                        idx, sl_mult, rr, sens,
                                        d["ds"], d["se"], d["ss"])
                rs.append(r)
                resolus.append(float((motif != 0).mean()))
            moy = 0.5 * (rs[0].mean() + rs[1].mean())
            par_bloc[b].append(moy)
            tous.append(moy)
    moy_b = [np.mean(v) for v in par_bloc if v]
    return (float(np.mean(tous)) if tous else np.nan,
            float(np.mean(resolus)) if resolus else np.nan,
            moy_b)


def main() -> int:
    B.configure("h1")
    d = B.prepare()
    phases = list(range(0, 12, 4))       # 3 decalages : le taux de base est
                                         # stable, inutile d'en payer douze
    print(f"{d['n']:,} bougies H1  |  {B.N_BLOCS} blocs  |  "
          f"fenetre de test intouchee")
    print("entree a CHAQUE occasion, moyenne des deux sens, aucun modele\n")
    print(f"{'SL':>5} {'R:R':>5} {'hold':>5} {'E[R] hasard':>12} "
          f"{'resolus':>9} {'blocs +':>9} {'ecart-type blocs':>17}")
    print("-" * 70)

    lignes = []
    for hold in HOLDS:
        for sl in SL_MULTS:
            for rr in RRS:
                er, res, moy_b = base(d, sl, rr, hold, phases)
                if not moy_b:
                    continue
                pos = sum(1 for x in moy_b if x > 0)
                sd = float(np.std(moy_b, ddof=1))
                lignes.append((abs(er), er, sl, rr, hold, res, pos,
                               len(moy_b), sd))
                print(f"{sl:5.1f} {rr:5.1f} {hold:5d} {er:+12.4f} "
                      f"{100*res:8.1f}% {pos:>5}/{len(moy_b)} {sd:17.4f}")

    print("\n--- les six geometries les plus neutres ---")
    print(f"{'SL':>5} {'R:R':>5} {'hold':>5} {'E[R] hasard':>12} "
          f"{'resolus':>9} {'blocs +':>9}")
    for _, er, sl, rr, hold, res, pos, n, sd in sorted(lignes)[:6]:
        print(f"{sl:5.1f} {rr:5.1f} {hold:5d} {er:+12.4f} "
              f"{100*res:8.1f}% {pos:>5}/{n}")

    print("\n'resolus' = part des courses qui touchent une barriere avant la")
    print("fin du delai. Une part basse veut dire que la plupart des trades")
    print("sont cloturees au marche : le R:R affiche n'est alors qu'un decor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
