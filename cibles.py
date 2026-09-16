"""Ce que rend un trade ouvert a cette barre — une seule reponse, lue de la config.

POURQUOI CE FICHIER EXISTE. La tete auxiliaire regressait `cibles_brutes`, qui
calcule un trade a take-profit FIXE de 2 R avec plafond de detention. Le jour
ou l'environnement est passe au stop suiveur SANS objectif, l'etiquette est
restee la meme : la tete apprenait a predire le resultat d'un trade que
personne ne fait. Rien ne l'a signale — les deux fonctions tournaient, les deux
rendaient des nombres.

C'est la faute que ce depot passe son temps a payer : deux descriptions du meme
trade, qui doivent s'accorder par convention. Ici la regle de sortie est LUE
dans `PPOConfig`, donc changer l'environnement change l'etiquette, sans rien a
synchroniser.

CE QUI EST MODELISE, ET DANS QUEL ORDRE. A chaque barre postérieure a l'entree :

  1. le stop suiveur se calcule sur le plus haut ATTEINT AVANT la barre
     courante. Le lire sur la barre en cours laisserait le stop monter puis
     etre touche dans la meme bougie — une fuite de futur qui fabrique du
     rendement sans lever d'erreur ;
  2. si le stop et l'objectif sont touches dans la meme bougie, on compte une
     PERTE. On ignore l'ordre reel des extremes, et supposer le contraire
     fabriquerait du rendement ;
  3. la friction se retranche une fois : un spread par aller-retour, plus les
     slippages d'entree et de sortie.

CE QUI N'EST PAS MODELISE. L'ordre des extremes intra-barre, le glissement reel
au stop, et la file d'attente du carnet. L'environnement non plus ne les
modelise pas — c'est voulu : les deux doivent decrire le MEME trade, pas le
trade le plus realiste possible.
"""

from __future__ import annotations

import numpy as np

SPREAD_BPS = 1.85       # mesure barre par barre sur 200 000 barres
SLIP_ENTREE_BPS = 1.0
SLIP_SORTIE_BPS = 2.0
BORNE_DEFAUT = 30 * 288  # 30 jours de M5 : borne de securite, pas une sortie


def _regle_cible(cfg) -> dict:
    """La geometrie que le modele apprend a CLASSER — pas celle qui est jouee.

    POURQUOI LES DEUX DIFFERENT, mesure du 2026-09-16. Deux proprietes etaient
    recherchees et paraissaient inconciliables :

      un objectif ETROIT (2 R) rend la cible bornee et bimodale, donc
      classable — la courbe de selectivite y descend de +0.56 au sommet a
      -0.12 au tout-venant ;
      un objectif LARGE, ou pas d'objectif, rend la geometrie NEUTRE — -0.0042
      contre -0.0732 a 2 R sur neuf ans d'entrees neutres.

    Elargir l'objectif supprimait la perte ET le classement en meme temps :
    exec35 l'a mesure a 6 R, gain d'apprentissage exactement nul.

    Elles se reunissent des qu'on cesse de les confondre. L'objectif a 2 R
    devient un INDICATEUR a atteindre, pas une sortie : le modele apprend a
    predire "ce trade touchera-t-il 2 R avant -1 R", et la position, elle,
    court jusqu'au stop suiveur.

    Ce n'est legitime que parce que les deux se suivent. Mesure sur 3 288
    occasions : correlation de RANG +0.892 entre la cible et le rendement
    reellement encaisse, et les occasions qui atteignent 2 R rendent +2.04 R
    en sortie reelle contre -0.91 pour les autres — 2.94 R d'ecart.
    """
    return {"sl": float(cfg.atr_sl_mult),
            "tp": float(getattr(cfg, "aux_tp_mult", 16.0)),
            "be": None, "ts": None, "td": 0.0}


def _regle(cfg) -> dict:
    """Extrait la regle de SORTIE de la configuration d'entrainement."""
    return {
        "sl": float(cfg.atr_sl_mult),
        "tp": float(cfg.atr_tp_mult) if getattr(cfg, "use_tp", True) else None,
        "be": (float(cfg.atr_be_mult)
               if getattr(cfg, "use_be_trail", False) else None),
        "ts": (float(cfg.atr_trail_mult)
               if getattr(cfg, "use_be_trail", False) else None),
        "td": float(cfg.atr_trail_dist),
    }


def rendements(df, idx, cfg, borne: int = BORNE_DEFAUT, indicateur: bool = False):
    """Rendement net (achat, vente) en unites de risque, aux indices donnes.

    Rend deux tableaux alignes sur `idx`, avec NaN quand la course ne se
    resout pas dans la borne. Ces NaN doivent etre MASQUES et non remplis :
    un zero serait une prediction, pas une absence.
    """
    # `indicateur=True` rend ce que le modele apprend a CLASSER (l'objectif a
    # 2 R), et non ce que la position encaissera. Voir `_regle_cible`.
    r = _regle_cible(cfg) if indicateur else _regle(cfg)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    atr = np.maximum(df["atr_14"].to_numpy(np.float64), 1e-9)
    n = len(c)
    fric = (SPREAD_BPS + SLIP_ENTREE_BPS + SLIP_SORTIE_BPS) / 1e4

    sorties = []
    for sens in (1, -1):
        out = np.full(len(idx), np.nan)
        for k, i in enumerate(idx):
            i = int(i)
            fin = min(i + 1 + borne, n)
            if fin - i < 10:
                continue
            entree = c[i]
            d = r["sl"] * atr[i]
            if d <= 0:
                continue
            hh, ll = h[i + 1:fin], l[i + 1:fin]

            if sens > 0:
                tp = entree + r["tp"] * atr[i] if r["tp"] else np.inf
                pic = np.maximum.accumulate(hh)
                pic = np.concatenate([[entree], pic[:-1]])   # decale d'une barre
                gain = pic - entree
                stop = np.full(len(hh), entree - d)
                if r["be"] is not None:
                    stop = np.where(gain >= r["be"] * atr[i], entree, stop)
                if r["ts"] is not None:
                    suiveur = pic - r["td"] * atr[i]
                    stop = np.where(gain >= r["ts"] * atr[i],
                                    np.maximum(stop, suiveur), stop)
                a = np.flatnonzero(ll <= stop)
                b = np.flatnonzero(hh >= tp) if r["tp"] else np.array([], int)
            else:
                tp = entree - r["tp"] * atr[i] if r["tp"] else -np.inf
                creux = np.minimum.accumulate(ll)
                creux = np.concatenate([[entree], creux[:-1]])
                gain = entree - creux
                stop = np.full(len(hh), entree + d)
                if r["be"] is not None:
                    stop = np.where(gain >= r["be"] * atr[i], entree, stop)
                if r["ts"] is not None:
                    suiveur = creux + r["td"] * atr[i]
                    stop = np.where(gain >= r["ts"] * atr[i],
                                    np.minimum(stop, suiveur), stop)
                a = np.flatnonzero(hh >= stop)
                b = np.flatnonzero(ll <= tp) if r["tp"] else np.array([], int)

            ja = a[0] if len(a) else np.inf
            jb = b[0] if len(b) else np.inf
            if not np.isfinite(ja) and not np.isfinite(jb):
                continue
            sortie = stop[int(ja)] if ja <= jb else tp
            out[k] = sens * (sortie - entree) / d - fric * entree / d
        sorties.append(out)
    return sorties[0], sorties[1]


def main() -> int:
    import numpy as np
    import training as T
    cfg = T.PPOConfig()
    r = _regle(cfg)
    trail = (f"depuis {r['ts']}xATR, distance {r['td']}xATR"
             if r["ts"] is not None else "inactif")
    print(f"regle lue dans PPOConfig : stop {r['sl']}xATR, "
          f"objectif {r['tp'] or 'AUCUN'}, trailing {trail}")
    df = T.load_mt5_data(cfg)
    tr = int(len(df) * 0.55)
    idx = np.arange(1000, tr - BORNE_DEFAUT - 2, 997)
    ra, rv = rendements(df, idx, cfg)
    for nom, y in (("achat", ra), ("vente", rv)):
        ok = np.isfinite(y)
        print(f"  {nom} : {ok.sum():,} resolus sur {len(y):,} "
              f"({100*(1-ok.mean()):.1f} % non resolus)")
        print(f"     E[R] {y[ok].mean():+.4f}   ecart-type {y[ok].std():.3f}   "
              f"mediane {np.median(y[ok]):+.3f}   max {y[ok].max():+.2f}")
    tous = np.concatenate([ra[np.isfinite(ra)], rv[np.isfinite(rv)]])
    print(f"  ensemble : E[R] {tous.mean():+.4f}, "
          f"variance {tous.var():.3f} — c'est elle que la tete doit expliquer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
