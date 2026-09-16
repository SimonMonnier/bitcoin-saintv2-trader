"""Un stop plus court paie-t-il, maintenant que la concurrence existe ?

LA QUESTION A CHANGE. Le balayage du matin comparait le rendement PAR TRADE
sous la contrainte d'une position a la fois : le nombre de trades y etait
plafonne par l'occupation, donc raccourcir la duree n'achetait presque rien
et seule l'esperance comptait. 12xATR gagnait.

Avec plusieurs positions simultanees, le plafond d'occupation saute et trois
effets s'opposent :

  1. UN STOP PLUS COURT DONNE PLUS DE TRADES. La duree tombe, donc il en
     rentre davantage dans la meme fenetre.

  2. ET PLUS D'OCCASIONS INDEPENDANTES, pas seulement plus de trades. Des
     trades courts se recouvrent moins : deux positions de meme sens
     correlent d'autant moins qu'elles ne partagent pas leur avenir. C'est ce
     que la premiere mesure ne pouvait pas voir.

  3. MAIS LA FRICTION EST UN MONTANT FIXE. Spread et slippage ne dependent
     pas du stop, donc leur poids EN UNITES DE RISQUE double quand le risque
     est divise par deux. C'est ce qui tuait les stops courts.

Ce qui decide n'est aucun des trois, c'est le produit : l'avantage annuel
vaut occasions x esperance, et sa lisibilite vaut esperance x racine des
occasions independantes, divisee par l'ecart-type.

ET UNE QUATRIEME CHOSE, trouvee en modelisant le courtier. Le lot minimum de
0.01 BTC impose un risque plancher : a 12xATR il vaut 1.18 % du capital au
lieu des 0.53 % voulus, a 6xATR seulement 0.59 %. Un stop plus court rapproche
donc le risque REEL du risque VISE — un argument que la mesure du matin ne
pouvait pas produire, puisqu'elle ignorait le courtier.

LE TEST N'EST PAS OUVERT : on s'arrete a `cibles.borne_etude`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import cibles as C
import training as T

STOPS = (3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
PAS = 3                    # une entree candidate toutes les 15 minutes
LOT_MIN = 0.01


class Cfg:
    """Une geometrie : stop variable, trailing a 1.5 R, pas d'objectif."""
    def __init__(self, sl):
        self.atr_sl_mult = sl
        self.use_tp = False
        self.atr_tp_mult = 0.0
        self.use_be_trail = True
        self.atr_be_mult = 1e9
        self.atr_trail_mult = 1.5 * sl
        self.atr_trail_dist = 1.5 * sl


def inflation(cor, duree, pas):
    """De combien la concurrence gonfle l'echantillon sans l'informer.

    ANALYTIQUE, et non par paires. La version en boucle imbriquee etait en
    O(n^2) : elle ne finissait pas sur les stops larges, dont les trades se
    recouvrent par milliers. Or le resultat ne depend que de la structure —
    des entrees espacees de `pas` pendant qu'un trade dure `duree` se
    recouvrent par paires d'ecart pas, 2 x pas, 3 x pas... et la somme de
    leurs correlations suffit.

        facteur = 1 + 2 x somme des correlations aux ecarts inferieurs a la duree

    Un facteur de 1 veut dire que chaque trade concurrent compte pour un ; un
    facteur de 10, qu'il en faut dix pour valoir une occasion.
    """
    f, k = 1.0, 1
    while k * pas < duree and k < 10000:
        ec = k * pas
        proche = min(cor, key=lambda x: abs(x - ec))
        f += 2.0 * max(cor[proche], 0.0) * (1.0 - ec / duree)
        k += 1
    return f


def main() -> int:
    cfg0 = T.PPOConfig()
    df = T.load_mt5_data(cfg0)
    n = len(df)
    va = C.borne_etude(n)
    t = pd.to_datetime(df["time"])
    ans = (t.iloc[va - 1] - t.iloc[0]).days / 365.25
    dv = df.iloc[:va].reset_index(drop=True)
    del df
    prix_fin = float(dv["close"].iloc[-1])
    atr_fin = float(dv["atr_14"].iloc[-1])
    print(f"{t.iloc[0]:%Y-%m-%d} -> {t.iloc[va-1]:%Y-%m-%d} ({ans:.1f} ans) — "
          f"test intouche")
    print(f"friction retenue : {C.SPREAD_BPS + C.SLIP_ENTREE_BPS + C.SLIP_SORTIE_BPS:.2f} "
          f"points de base par aller-retour\n")

    idx = np.arange(100, len(dv) - C.BORNE_DEFAUT - 2, PAS)

    print(f"{'stop':>5} {'risque':>8} {'friction':>9} {'duree':>8} "
          f"{'seul/an':>9} {'E[R] sym':>9} {'+/-':>7} "
          f"{'R/an':>7} {'occ/an conc':>12} {'detect.':>8} {'lot min':>8}")
    print("-" * 104)

    for sl in STOPS:
        cfg = Cfg(sl)
        ra, rv, da, dvv = C.rendements(dv, idx, cfg, durees=True)
        ok = np.isfinite(ra) & np.isfinite(rv) & np.isfinite(da)
        if ok.sum() < 200:
            continue
        sym = (ra[ok] - rv[ok]) / 2.0
        dur = da[ok]
        pos = idx[ok]
        med = float(np.median(dur))

        # ENTREES NON CHEVAUCHANTES, espacees de la duree mediane DE CETTE
        # geometrie : chaque largeur recoit ainsi le nombre d'occasions
        # qu'elle permet reellement, ce qui est tout l'enjeu de la question.
        pas_ind = max(int(med), 1)
        garde, libre = [], -1
        for k in range(len(pos)):
            if pos[k] < libre:
                continue
            garde.append(k)
            libre = pos[k] + dur[k]
        g = np.array(garde, dtype=int)
        y = sym[g]
        se = y.std(ddof=1) / np.sqrt(len(y))
        par_an = len(y) / ans

        # Correlation selon l'ecart, pour le compte effectif sous concurrence.
        cor = {}
        for e in (1, 4, 12, 48, 144, 288, 576, 1152, 2304, 4608):
            d = e // PAS
            if 1 <= d < len(sym) - 10:
                cor[e] = float(np.corrcoef(sym[:-d], sym[d:])[0, 1])
        # Sous concurrence on entre a chaque barre candidate : le nombre de
        # trades explose, mais l'information ne suit que divisee par le
        # facteur de recouvrement.
        infl = inflation(cor, med, PAS) if cor else 1.0
        trades_conc = (len(dv) / PAS) / ans          # entrees possibles par an
        occ_conc = trades_conc / infl

        risque_bps = sl * atr_fin / prix_fin * 1e4
        fric_R = (C.SPREAD_BPS + C.SLIP_ENTREE_BPS + C.SLIP_SORTIE_BPS) / risque_bps
        # Risque impose par le lot minimum, en part d'un capital de 1 000.
        r_lot = LOT_MIN * sl * atr_fin / 1000.0 * 100.0
        detect = y.mean() * np.sqrt(max(occ_conc * ans, 1.0)) / max(y.std(), 1e-9)

        print(f"{sl:>4g}x {risque_bps:>7.0f}b {fric_R:>8.3f}R "
              f"{med*5/60:>7.1f}h {par_an:>10.0f} {y.mean():>+9.4f} "
              f"{se:>7.4f} {par_an*y.mean():>+7.1f} {occ_conc:>9.0f} "
              f"{detect:>8.2f} {r_lot:>7.2f}%")

    print("\nLECTURE. 'occ ind.' est le nombre d'occasions INDEPENDANTES que la")
    print("concurrence laisse esperer, correlations de recouvrement deduites —")
    print("c'est ce que la mesure du matin ne pouvait pas voir. 'detect.' est")
    print("le rapport signal sur bruit total : c'est lui qui decide, pas le")
    print("rendement par trade. 'lot min' est le risque que le courtier IMPOSE")
    print("a 1 000 EUR de capital, contre 0.53 % vise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
