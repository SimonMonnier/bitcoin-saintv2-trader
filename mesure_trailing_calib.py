"""Le stop suiveur est-il calibre ? Il ne l'a jamais ete.

POURQUOI CE FICHIER EXISTE. Les reglages en place — declenchement a 1.5 R,
distance 1.5 R — viennent de `mesure_trailing`, dont le resultat a ete RETIRE
le 2026-09-16 : elle comparait un stop FIXE de 94 points de base a un
environnement qui utilise un stop ADAPTATIF en multiples d'ATR, et le +0.0829 R
annonce valait +0.0001 une fois corrige. La mesure est tombee, les parametres
sont restes.

Depuis, trois choses ont change et aucune ne laisse ce reglage intact :
le stop est passe de 8 a 6 fois l'ATR, l'occupation ne plafonne plus le
nombre de trades, et le critere de decision est devenu le rendement ANNUEL
plutot que le rendement par trade.

CE QUI EST BALAYE. Le declenchement (a partir de quel gain le stop se met a
suivre) et la distance (a quelle distance du plus haut il se place). Les deux
en unites de RISQUE, pas d'ATR : exprimes en ATR ils changeraient de sens a
chaque fois que le stop change de largeur, ce qui est exactement l'erreur que
le test d'alignement a attrapee cet apres-midi.

CE QUI DECIDE. Le rendement par an, et sa tenue quand la friction est plus
chere que supposee — c'est ce critere qui a fait preferer 6xATR a 3xATR.

LE TEST N'EST PAS OUVERT : on s'arrete a `cibles.borne_etude`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import cibles as C
import training as T

DECLENCHE = (0.5, 1.0, 1.5, 2.0, 3.0)      # en R
DISTANCE = (0.5, 1.0, 1.5, 2.0)            # en R
PAS = 6                                     # une entree candidate par 30 min


class Cfg:
    """Le stop de production, avec un trailing variable."""
    def __init__(self, sl, declenche, distance):
        self.atr_sl_mult = sl
        self.use_tp = False
        self.atr_tp_mult = 0.0
        self.use_be_trail = True
        self.atr_be_mult = 1e9
        self.atr_trail_mult = declenche * sl
        self.atr_trail_dist = distance * sl


def main() -> int:
    cfg0 = T.PPOConfig()
    sl = float(cfg0.atr_sl_mult)
    df = T.load_mt5_data(cfg0)
    n = len(df)
    va = C.borne_etude(n)
    t = pd.to_datetime(df["time"])
    ans = (t.iloc[va - 1] - t.iloc[0]).days / 365.25
    dv = df.iloc[:va].reset_index(drop=True)
    del df
    fric = C.SPREAD_BPS + C.SLIP_ENTREE_BPS + C.SLIP_SORTIE_BPS
    risque_bps = sl * float(np.median(dv["atr_14"] / dv["close"])) * 1e4
    fric_R = fric / risque_bps
    print(f"stop {sl:g}xATR ({risque_bps:.0f} points de base), friction "
          f"{fric_R:.3f} R par trade")
    print(f"{t.iloc[0]:%Y-%m-%d} -> {t.iloc[va-1]:%Y-%m-%d} ({ans:.1f} ans) — "
          f"test intouche\n")

    idx = np.arange(100, len(dv) - C.BORNE_DEFAUT - 2, PAS)

    print("RENDEMENT PAR AN, en unites de risque")
    print(f"{'declenche':>10} " + " ".join(f"{'d=' + str(d) + 'R':>10}"
                                           for d in DISTANCE))
    print("-" * (11 + 11 * len(DISTANCE)))
    table = {}
    for dec in DECLENCHE:
        ligne = []
        for dist in DISTANCE:
            cfg = Cfg(sl, dec, dist)
            ra, rv, da, _ = C.rendements(dv, idx, cfg, durees=True)
            ok = np.isfinite(ra) & np.isfinite(rv) & np.isfinite(da)
            if ok.sum() < 200:
                ligne.append(None)
                continue
            sym = (ra[ok] - rv[ok]) / 2.0
            dur = da[ok]
            pos = idx[ok]
            # Entrees NON CHEVAUCHANTES : chaque reglage recoit le nombre
            # d'occasions que SA duree permet.
            garde, libre = [], -1
            for k in range(len(pos)):
                if pos[k] < libre:
                    continue
                garde.append(k)
                libre = pos[k] + dur[k]
            g = np.array(garde, dtype=int)
            y = sym[g]
            par_an = len(y) / ans
            table[(dec, dist)] = (par_an, y.mean(),
                                  y.std(ddof=1) / np.sqrt(len(y)),
                                  float(np.median(dur)) * 5 / 60)
            ligne.append(par_an * y.mean())
        print(f"{dec:>9.1f}R " + " ".join(
            "      n/a " if v is None else f"{v:>+10.1f}" for v in ligne))

    print("\nLE MEME, AVEC UNE FRICTION UNE FOIS ET DEMIE PLUS CHERE")
    print(f"{'declenche':>10} " + " ".join(f"{'d=' + str(d) + 'R':>10}"
                                           for d in DISTANCE))
    print("-" * (11 + 11 * len(DISTANCE)))
    for dec in DECLENCHE:
        ligne = []
        for dist in DISTANCE:
            v = table.get((dec, dist))
            ligne.append(None if v is None else v[0] * (v[1] - 0.5 * fric_R))
        print(f"{dec:>9.1f}R " + " ".join(
            "      n/a " if v is None else f"{v:>+10.1f}" for v in ligne))

    print("\nDETAIL DES TROIS MEILLEURS (friction supposee)")
    best = sorted(table.items(), key=lambda kv: -kv[1][0] * kv[1][1])[:3]
    print(f"{'declenche':>10} {'distance':>9} {'trades/an':>10} "
          f"{'E[R]':>9} {'+/-':>8} {'duree':>8} {'R/an':>8}")
    for (dec, dist), (par_an, m, se, duree) in best:
        print(f"{dec:>9.1f}R {dist:>8.1f}R {par_an:>10.0f} {m:>+9.4f} "
              f"{se:>8.4f} {duree:>7.1f}h {par_an*m:>+8.1f}")

    print("\nLECTURE. Un declenchement bas fait sortir tot : beaucoup de")
    print("trades courts, chacun rapportant peu, et la friction pese lourd.")
    print("Un declenchement haut laisse courir : peu de trades, chacun")
    print("rapportant plus. C'est le produit qui decide, et la seconde table")
    print("dit lequel des deux tient quand le spread est moins favorable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
