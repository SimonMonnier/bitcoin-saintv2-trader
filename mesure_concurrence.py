"""Tenir plusieurs positions a la fois : combien de trades, et combien d'INFORMATION ?

LA QUESTION SE COUPE EN DEUX, et les deux reponses divergent.

  1. COMBIEN DE TRADES. Mecaniquement plus, puisque c'est l'occupation qui
     les plafonne aujourd'hui : a 5 % de selectivite une position est ouverte
     91 % du temps, donc presque tout signal est ignore. Lever la contrainte
     multiplie le compte.

  2. COMBIEN D'OCCASIONS INDEPENDANTES. C'est autre chose, et c'est la seule
     chose qui decide. Deux positions longues ouvertes a deux heures d'ecart
     dans la meme tendance subissent le MEME mouvement : elles ne comptent pas
     pour deux. La regle que ce depot paie depuis des semaines dit que le
     nombre d'occasions vaut *duree d'historique / duree d'un trade*, jamais
     le nombre de barres — et la concurrence ne change ni l'un ni l'autre.

CE QUE CE FICHIER MESURE. La correlation entre deux trades de MEME SENS
ouverts a un ecart donne. Elle dit directement ce qu'un second trade
concurrent apporte : a correlation 1 il ne compte pas, a correlation 0 il
compte pour un. On en tire la TAILLE D'ECHANTILLON EFFECTIVE,

    N_eff = N / (1 + 2 * somme des correlations aux ecarts < duree)

qui est le nombre de trades reellement independants derriere N trades
chevauchants, et donc le facteur par lequel la detectabilite progresse.

ON MESURE AUSSI LE RISQUE. K positions concurrentes de meme sens, chacune
dimensionnee a `risk_per_trade`, exposent K fois le risque sur un mouvement
qu'elles subissent ensemble. Pour garder le meme risque de portefeuille il
faut diviser la taille par K — ce qui divise aussi le rendement par trade.
Le rendement TOTAL est alors inchange, et seul le gain de N_eff compte.

LE TEST N'EST PAS OUVERT : seule la validation du fold 1 est lue.
"""

from __future__ import annotations

import numpy as np

import cibles as C
import training as T

T_TRAIN, T_VAL = 0.55, 0.15
PAS = 3                                  # une entree candidate toutes les 15 min
ECARTS = (1, 2, 4, 8, 12, 24, 48, 96, 192, 288, 576, 1152, 2304)
CONCURRENCE = (1, 2, 3, 5, 8, 12, 20)


def enchaine_k(pos, dur, k_max):
    """Trades pris quand on tolere jusqu'a `k_max` positions simultanees."""
    pris, fins = [], []
    for i in range(len(pos)):
        if not np.isfinite(dur[i]):
            continue
        fins = [f for f in fins if f > pos[i]]
        if len(fins) >= k_max:
            continue
        pris.append(i)
        fins.append(pos[i] + dur[i])
    return np.array(pris, dtype=int)


def main() -> int:
    cfg = T.PPOConfig()
    df = T.load_mt5_data(cfg)
    n = len(df)
    b = int(n * T_TRAIN)
    dv = df.iloc[b:b + int(n * T_VAL)].reset_index(drop=True)
    del df
    jours = len(dv) * 5 / 60 / 24
    print(f"fold 1, validation — {jours:.0f} jours, "
          f"stop {cfg.atr_sl_mult:g}xATR, trailing "
          f"{cfg.atr_trail_mult / cfg.atr_sl_mult:.2f} R\n")

    idx = np.arange(cfg.lookback, len(dv) - C.BORNE_DEFAUT - 2, PAS)
    ra, rv, da, dvv = C.rendements(dv, idx, cfg, durees=True)
    ok = np.isfinite(ra) & np.isfinite(rv)
    idx, ra, rv, da = idx[ok], ra[ok], rv[ok], da[ok]
    dvv = dvv[ok]
    med = float(np.median(da))
    print(f"{len(idx):,} entrees candidates, duree mediane {med * 5 / 60:.1f} h "
          f"({med:.0f} barres)\n")

    # ------------------------------------------------------------------
    # 1. CE QU'UN SECOND TRADE APPORTE, selon l'ecart qui le separe du premier.
    print("CORRELATION ENTRE DEUX TRADES DE MEME SENS, selon leur ecart")
    print(f"{'ecart':>9} {'en h':>7} {'achat':>8} {'vente':>8} {'moyenne':>9}")
    print("-" * 46)
    cor = {}
    for e in ECARTS:
        d = e // PAS
        if d < 1 or d >= len(ra) - 10:
            continue
        ca = float(np.corrcoef(ra[:-d], ra[d:])[0, 1])
        cv = float(np.corrcoef(rv[:-d], rv[d:])[0, 1])
        cor[e] = (ca + cv) / 2
        marque = "  <- duree mediane" if e <= med < ECARTS[min(
            ECARTS.index(e) + 1, len(ECARTS) - 1)] else ""
        print(f"{e:>9} {e * 5 / 60:>7.1f} {ca:>8.3f} {cv:>8.3f} "
              f"{cor[e]:>9.3f}{marque}")

    # ------------------------------------------------------------------
    # 2. LE COMPTE, ET CE QU'IL VAUT VRAIMENT.
    print("\nTENIR K POSITIONS A LA FOIS")
    print(f"{'K':>4} {'trades':>8} {'x vs 1':>8} {'N effectif':>12} "
          f"{'x vs 1':>8} {'E[R]':>9} {'R total':>9}")
    print("-" * 66)
    base_n = base_eff = None
    for k in CONCURRENCE:
        p = enchaine_k(idx, da, k)
        if len(p) < 10:
            continue
        # Sens tire a pile ou face : on mesure la concurrence, pas une
        # direction. Le rendement d'un trade ne depend pas de K.
        rng = np.random.default_rng(0)
        sg = rng.integers(0, 2, len(p)) * 2 - 1
        y = np.where(sg > 0, ra[p], rv[p])

        # N EFFECTIF. Somme des correlations entre trades qui se recouvrent :
        # chaque paire ouverte en meme temps ne compte pas pour deux.
        inflation = 1.0
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                ec = int(idx[p[j]] - idx[p[i]])
                if ec >= da[p[i]]:
                    break
                proche = min(cor, key=lambda x: abs(x - ec))
                inflation += 2.0 * max(cor[proche], 0.0) / len(p)
        n_eff = len(p) / inflation
        if base_n is None:
            base_n, base_eff = len(p), n_eff
        print(f"{k:>4} {len(p):>8} {len(p)/base_n:>7.2f}x {n_eff:>12.1f} "
              f"{n_eff/base_eff:>7.2f}x {y.mean():>+9.4f} {y.sum():>+9.1f}")

    print("\nLECTURE. 'trades' est ce que la concurrence ajoute au compte ;")
    print("'N effectif' est ce qu'elle ajoute a l'INFORMATION. Si la seconde")
    print("colonne monte beaucoup moins que la premiere, les trades ajoutes")
    print("sont des copies du premier, et la detectabilite — qui va comme la")
    print("racine de N effectif — n'en profite pas.")
    print("\nET LE RISQUE. K positions de meme sens subissent le meme mouvement :")
    print("le risque de portefeuille vaut K fois celui d'une seule. Le garder")
    print("constant impose de diviser la taille par K, ce qui divise le")
    print("rendement par trade d'autant. Le R total ci-dessus est donc a lire")
    print("A TAILLE CONSTANTE, pas a risque constant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
