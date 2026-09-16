"""Le break-even et le trailing valent-ils quelque chose a CETTE geometrie ?

CE QUI A DEJA ETE MESURE, ET POURQUOI ON RECOMMENCE. Le depot avait tranche :
« avec BE/trail, PF 0.98 ; sans, PF 1.85 » — le trailing coupait les gagnants
avant le take-profit, et `update_sl_be_trailing_live()` est commente dans le
live depuis. Mais les seuils de cette mesure declenchaient a 1.0 et 1.5 ATR,
dimensionnes pour un stop de 5xATR. Avec le stop de 8xATR d'aujourd'hui, le
break-even se declenche apres 12 % du chemin vers le stop : le moindre bruit
scratche la position. Ce n'etait pas une mesure du trailing, c'etait une mesure
d'un trailing mal dimensionne.

CE QU'IL NE PEUT PAS FAIRE, ET IL FAUT LE DIRE D'ABORD. La friction se paie une
fois par ALLER-RETOUR — un spread, deux slippages — quelle que soit la regle de
sortie. Un trailing ne la baisse pas. Et sur une martingale, aucune regle
d'arret ne change l'esperance : il ne peut que REDISTRIBUER.

CE QU'IL PEUT ACHETER, ET C'EST MESURABLE :

  moins de VARIANCE par trade. La detectabilite vaut (avantage) x racine(N)
  divise par l'ecart-type des resultats. Resserrer la distribution sans deplacer
  la moyenne ameliore ce qu'on peut prouver, pour le meme avantage.

  des trades plus COURTS sans friction supplementaire, donc plus d'occasions
  sur le meme historique.

Les deux effets tirent dans le bon sens, et l'esperance tire dans l'autre. La
mesure dit lequel gagne.

GESTION A LA MINUTE. Decider a la minute ne cree pas d'occasions — c'est la
duree d'un trade qui les compte, pas la frequence des decisions. Mais SORTIR a
la minute n'est pas decider : un stop suiveur ajuste toutes les minutes colle
mieux au plus haut qu'un stop ajuste toutes les cinq minutes. C'est la seule
chose que le M1 apporte reellement, et elle se mesure ici.

    python mesure_trailing.py
"""

from __future__ import annotations

import numpy as np

RISQUE_BPS = 94.0       # 8xATR(M5) en points de base — la geometrie en place
RR = 2.0
SPREAD_BPS = 1.85       # mesure barre par barre sur 200 000 barres
SLIP_BPS = 3.0          # 1 a l'entree, 2 a la sortie
BORNE_JOURS = 7
N_ENTREES = 3000
PHASES = 8

# (nom, break-even a X du risque, trailing demarre a X, distance, objectif)
# `objectif = None` retire le take-profit : le stop suiveur devient la SEULE
# sortie, et la queue droite n'est plus ecretee.
REGLES = [
    ("temoin (fixe)",         None, None, None, RR),
    ("BE a 0.50 R",           0.50, None, None, RR),
    ("trail 1.0 R, dist 0.5", None, 1.00, 0.50, RR),
    ("BE 0.5 + trail 1.5/1.0", 0.50, 1.50, 1.00, RR),
    ("TP 4 R au lieu de 2",   0.50, 1.50, 1.00, 4.0),
    ("SANS TP, trail 1.0/1.0", None, 1.00, 1.00, None),
    ("SANS TP, trail 1.0/0.5", None, 1.00, 0.50, None),
    ("SANS TP, trail 1.5/1.5", None, 1.50, 1.50, None),
    ("SANS TP, BE+trail 1/1",  0.50, 1.00, 1.00, None),
]


def course(h, l, c, idx, sens, risque, be, tr_start, tr_dist, borne, rr=RR):
    """Une course aux barrieres avec regle de sortie. Rend (R, duree).

    LE STOP SUIVEUR SE CALCULE SUR LE PASSE, decale d'une barre. Le lire sur le
    plus haut de la barre COURANTE permettrait au stop de monter puis d'etre
    touche dans la meme bougie — une fuite de futur qui fabriquerait du
    rendement sans lever d'erreur.
    """
    n = len(c)
    R = np.full(len(idx), np.nan)
    D = np.full(len(idx), np.nan)
    for k, i in enumerate(idx):
        fin = min(i + 1 + borne, n)
        if fin - i < 10:
            continue
        entree = c[i]
        d = risque * entree / 1e4                 # distance du stop, en prix
        hh, ll = h[i + 1:fin], l[i + 1:fin]
        if sens > 0:
            tp = entree + rr * d if rr else np.inf
            # Plus haut atteint AVANT la barre courante.
            pic = np.maximum.accumulate(hh)
            pic = np.concatenate([[entree], pic[:-1]])
            gain = pic - entree
            stop = np.full(len(hh), entree - d)
            if be is not None:
                stop = np.where(gain >= be * d, entree, stop)
            if tr_start is not None:
                suiveur = pic - tr_dist * d
                stop = np.where(gain >= tr_start * d,
                                np.maximum(stop, suiveur), stop)
            a = np.flatnonzero(ll <= stop)
            b = np.flatnonzero(hh >= tp) if rr else np.array([], int)
        else:
            tp = entree - rr * d if rr else -np.inf
            creux = np.minimum.accumulate(ll)
            creux = np.concatenate([[entree], creux[:-1]])
            gain = entree - creux
            stop = np.full(len(hh), entree + d)
            if be is not None:
                stop = np.where(gain >= be * d, entree, stop)
            if tr_start is not None:
                suiveur = creux + tr_dist * d
                stop = np.where(gain >= tr_start * d,
                                np.minimum(stop, suiveur), stop)
            a = np.flatnonzero(hh >= stop)
            b = np.flatnonzero(ll <= tp) if rr else np.array([], int)

        ja = a[0] if len(a) else np.inf
        jb = b[0] if len(b) else np.inf
        if not np.isfinite(ja) and not np.isfinite(jb):
            continue
        # Egalite dans la meme bougie : compte comme une PERTE. On ignore
        # l'ordre reel des extremes, et supposer le contraire fabriquerait du
        # rendement.
        if ja <= jb:
            sortie, j = stop[int(ja)], int(ja)
        else:
            sortie, j = tp, int(jb)
        brut = sens * (sortie - entree) / d
        # Friction : un spread par aller-retour, plus les slippages.
        R[k] = brut - (SPREAD_BPS + SLIP_BPS) / RISQUE_BPS
        D[k] = j + 1
    ok = np.isfinite(R)
    return R[ok], D[ok]


def main() -> int:
    import MetaTrader5 as mt5
    if not mt5.initialize():
        print("MT5 injoignable")
        return 1
    res = {}
    for nom_ut, tf, par_jour in (("M5", mt5.TIMEFRAME_M5, 288),
                                 ("M1", mt5.TIMEFRAME_M1, 1440)):
        mt5.symbol_select("BTCUSD", True)
        r = mt5.copy_rates_from_pos("BTCUSD", tf, 0, 700_000)
        if r is None or len(r) < 50_000:
            print(f"{nom_ut} : historique insuffisant")
            continue
        h = r["high"].astype(np.float64)
        l = r["low"].astype(np.float64)
        c = r["close"].astype(np.float64)
        borne = BORNE_JOURS * par_jour
        ans = (r["time"][-1] - r["time"][0]) / (365.25 * 24 * 3600)
        print(f"\n{'='*74}\nGESTION EN {nom_ut} — {len(r):,} barres, {ans:.1f} ans, "
              f"borne {BORNE_JOURS} jours")
        print(f"{'regle':<24} {'trades':>7} {'E[R]':>8} {'ecart-type':>11} "
              f"{'duree med':>10} {'occasions':>10} {'detect.':>9}")
        print("-" * 84)
        rng = np.random.default_rng(0)
        for nom, be, ts, td, rr in REGLES:
            ers, sds, dus = [], [], []
            for ph in range(PHASES):
                idx = np.sort(rng.choice(np.arange(50, len(c) - borne - 2),
                                         size=N_ENTREES // PHASES,
                                         replace=False))
                for sens in (1, -1):
                    R, D = course(h, l, c, idx, sens, RISQUE_BPS, be, ts, td,
                                  borne, rr)
                    if len(R) < 50:
                        continue
                    ers.append(R.mean()); sds.append(R.std()); dus.append(np.median(D))
            if not ers:
                print(f"{nom:<24} trop peu de courses")
                continue
            er, sd, du = np.mean(ers), np.mean(sds), np.mean(dus)
            occ = len(c) / max(du, 1)
            # Detectabilite : ce qu'on peut prouver = |E[R]| / (sd / racine(N)).
            det = np.sqrt(occ) / max(sd, 1e-9)
            res.setdefault(nom_ut, {})[nom] = det
            print(f"{nom:<24} {len(ers)*N_ENTREES//PHASES:>7,} {er:>+8.4f} "
                  f"{sd:>11.3f} {du*(1 if nom_ut=='M1' else 5)/60:>9.1f}h "
                  f"{occ:>10,.0f} {det:>9.1f}")
        print("\n  detect. = racine(occasions) / ecart-type : le facteur par lequel")
        print("  un avantage donne devient visible. Plus haut est mieux.")
    mt5.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
