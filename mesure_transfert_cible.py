# -*- coding: utf-8 -*-
"""LA CIBLE DE LA TETE TRANSFERE-T-ELLE AU RENDEMENT REELLEMENT ENCAISSE ?

TOUT LE SYSTEME REPOSE SUR CETTE AFFIRMATION. `cibles._regle_cible` justifie
d'entrainer la tete sur une geometrie DIFFERENTE de celle qui est jouee :

    "Ce n'est legitime que parce que les deux se suivent. Mesure sur 3 288
     occasions : correlation de RANG +0.892 entre la cible et le rendement
     reellement encaisse, et les occasions qui atteignent 2 R rendent +2.04 R
     en sortie reelle contre -0.91 pour les autres."

Si ce +0.892 ne tient plus, la tete apprend a classer une chose et on s'en
sert pour en trier une autre — et tout ce qui est construit dessus (le seuil,
la selectivite, `sommet`, `rhoAux`) mesure alors du vide.

POURQUOI LE DOUTE. La GEOMETRIE A CHANGE depuis cette mesure. La cible reste
"toucher 2 R avant -1 R", mais la sortie jouee n'est plus la meme : stop
10xATR, trailing arme a 2 R et distance 2 R. Or le +0.892 a ete mesure sous
une autre regle de sortie, et rien dans le code ne re-verifie qu'il tient.

CE QUE LA MESURE FAIT. Sur la fenetre de VALIDATION, aux memes occasions :

    rho(indicateur 2 R, rendement reel)   le transfert lui-meme
    E[R reel | indicateur = 1] et | = 0   l'ecart que l'ouvrage annonce a 2.94

LA FENETRE DE TEST N'EST PAS LUE.

    python mesure_transfert_cible.py
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
    n = len(df)
    so, ci = C._regle(cfg), C._regle_cible(cfg)
    print(f"SORTIE JOUEE   stop {so['sl']:g}xATR | objectif "
          f"{(str(so['tp']/so['sl']) + ' R') if so['tp'] else 'AUCUN'} | "
          f"trailing " + (f"arme {so['ts']/so['sl']:.1f} R dist "
                          f"{so['td']/so['sl']:.1f} R"
                          if so["ts"] is not None else "inactif"))
    print(f"CIBLE APPRISE  stop {ci['sl']:g}xATR | objectif "
          f"{ci['tp']/ci['sl']:.1f} R | trailing "
          f"{'inactif' if ci['ts'] is None else 'actif'}")
    print()

    for nom, (a, b) in (("ENTRAINEMENT", (0, int(n * 0.55))),
                        ("VALIDATION", (int(n * 0.55), int(n * 0.70)))):
        idx = np.arange(a + cfg.lookback + 1, b - C.BORNE_DEFAUT - 2, PAS)
        ind, _ = C.rendements(df, idx, cfg, indicateur=True)
        reel, _ = C.rendements(df, idx, cfg, indicateur=False)
        i = np.asarray(ind, float)
        r = np.asarray(reel, float)
        ok = np.isfinite(i) & np.isfinite(r)
        i, r = i[ok], r[ok]
        rho = T._correlation_rang(i, r)
        # occasions INDEPENDANTES, pour ne pas surestimer la certitude
        n_ind = (b - a) / 372.0
        se = 1.0 / np.sqrt(max(n_ind, 1.0))
        m1 = i > 0.5
        print(f"{nom}  ({len(i):,} occasions, ~{n_ind:,.0f} independantes)")
        print(f"  rho(indicateur, rendement reel)  {rho:>+8.4f}   "
              f"(+/- {se:.3f})   [le code annonce +0.892]")
        if m1.sum() > 30 and (~m1).sum() > 30:
            print(f"  E[R reel | indicateur = 1]       {r[m1].mean():>+8.3f} R"
                  f"   sur {int(m1.sum()):,} occasions")
            print(f"  E[R reel | indicateur = 0]       {r[~m1].mean():>+8.3f} R"
                  f"   sur {int((~m1).sum()):,} occasions")
            print(f"  ECART                            "
                  f"{r[m1].mean()-r[~m1].mean():>+8.3f} R   "
                  f"[le code annonce 2.94]")
        print(f"  part d'indicateurs a 1 : {100*m1.mean():.1f} %")
        print()
    print("LECTURE. Si le rho mesure est tres en dessous de +0.892, la tete")
    print("apprend a classer une chose et on s'en sert pour en trier une")
    print("autre. Tout ce qui est bati dessus — seuil, selectivite, `sommet`,")
    print("`rhoAux` — mesure alors autre chose que ce qu'on croit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
