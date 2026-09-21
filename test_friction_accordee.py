# -*- coding: utf-8 -*-
"""L'etiquette et le simulateur facturent-ils LE MEME trade au meme prix ?

CE QUI A RENDU CE TEST NECESSAIRE, le 2026-09-21. Deux endroits comptaient
la friction et ne trouvaient pas pareil :

    l'etiquette (`cibles_m1.cout_aller_retour`)   4.36 bps
    le simulateur (`PPOConfig` + l'environnement) 2.27 bps     1.92x

L'etiquette prenait le spread a sa valeur CENTRALE alors que le simulateur
le tire d'une loi bimodale, et facturait 3.0 bps de glissement a l'aller ET
au retour quand le simulateur en tire un dans [0, 1.0] a l'ENTREE seulement.

L'etiquette est ce que les tetes apprennent a predire. Sur un mouvement
brut de +3.0 bps elle annoncait -1.36 — « perdant, ne pas prendre » — quand
le compte encaissait +0.73. On apprenait au modele a s'abstenir deux fois
trop.

CE TEST RECALCULE LES DEUX SEPAREMENT et exige qu'ils coincident. Il ne
lit pas une constante : il refait le calcul du simulateur a partir de la
config, et celui de l'etiquette a partir du registre.

    python test_friction_accordee.py
"""
import sys

import numpy as np

import instruments as I
import cibles_m1 as C
import training as T

_ok = _ko = 0


def verifie(nom, cond, detail=""):
    global _ok, _ko
    if cond:
        _ok += 1
        print("  ok   %-50s %s" % (nom, detail))
    else:
        _ko += 1
        print("  ECHEC %-49s %s" % (nom, detail))


for sym in ("XAUUSD", "BTCUSD"):
    print("\n%s" % sym)
    c = T.PPOConfig(symbol=sym)

    # CE QUE LE SIMULATEUR PRELEVE, recalcule depuis la CONFIG — pas lu
    # dans le registre, sans quoi le test serait une tautologie.
    p_large = float(c.spread_wide_prob)
    base = float(c.spread_bps)
    haut = float(c.spread_bps_wide_factor)
    spread_sim = (1 - p_large) * base + p_large * base * (1.2 + haut) / 2
    cout_sim = 2 * spread_sim + float(c.entry_slippage_bps) / 2

    cout_cib = C.cout_aller_retour(sym)

    verifie("l'etiquette et le simulateur s'accordent",
            abs(cout_cib - cout_sim) < 1e-9,
            "etiquette %.4f bps   simulateur %.4f bps" % (cout_cib, cout_sim))

    # LE SPREAD COMPTE DEUX FOIS, LE GLISSEMENT UNE. C'est la geometrie du
    # simulateur : il tire le spread une fois par trade et le facture a
    # l'ouverture comme a la fermeture, le glissement ne frappe qu'a
    # l'entree. Une symetrie mal posee ici vaut ~1 bps d'erreur.
    verifie("  le spread est compte DEUX fois",
            abs(cout_cib - (2 * I.spread_espere(sym)
                            + float(c.entry_slippage_bps) / 2)) < 1e-9,
            "2 x %.4f de spread" % I.spread_espere(sym))
    verifie("  le glissement est compte UNE fois, a l'entree",
            abs((cout_cib - 2 * I.spread_espere(sym))
                - float(c.entry_slippage_bps) / 2) < 1e-9,
            "%.4f bps, moitie de [0, %.1f] tire uniformement"
            % (float(c.entry_slippage_bps) / 2, c.entry_slippage_bps))

    # LA VALEUR CENTRALE N'EST PAS L'ESPERANCE, et c'etait la premiere des
    # deux fautes. Si la loi s'aplatissait un jour, ce test le dirait.
    verifie("  l'esperance du spread depasse sa valeur centrale",
            I.spread_espere(sym) > base,
            "%.4f contre %.4f au centre, %+.1f %%"
            % (I.spread_espere(sym), base,
               100 * (I.spread_espere(sym) / base - 1)))

# =====================================================================
print("\nLA CONFIG NE PORTE PLUS DE SECOND EXEMPLAIRE")
# =====================================================================
for sym in ("XAUUSD", "BTCUSD"):
    c = T.PPOConfig(symbol=sym)
    r = I.INSTRUMENTS[sym]
    for champ in ("spread_bps", "spread_wide_prob", "spread_bps_wide_factor",
                  "entry_slippage_bps"):
        verifie("%s : %s vient du registre" % (sym, champ),
                abs(float(getattr(c, champ)) - float(r[champ])) < 1e-12,
                "%.4f" % float(getattr(c, champ)))

# =====================================================================
print("\nCE QUE LA CORRECTION CHANGE POUR LES ETIQUETTES")
# =====================================================================
_avant = 2.0 * 0.68 + 3.0        # l'ancien calcul, en dur
_apres = C.cout_aller_retour("XAUUSD")
print("  aller-retour facture : %.2f -> %.2f bps  (%.2fx moins cher)"
      % (_avant, _apres, _avant / _apres))
print("  un mouvement brut de +3.0 bps :")
print("     avant  3.00 - %.2f = %+.2f bps  -> refuse" % (_avant, 3.0 - _avant))
print("     apres  3.00 - %.2f = %+.2f bps  -> pris" % (_apres, 3.0 - _apres))
verifie("le seuil de rentabilite a baisse", _apres < _avant,
        "il faut %.2f bps de mouvement brut au lieu de %.2f" % (_apres, _avant))

print("\n%d/%d OK" % (_ok, _ok + _ko))
if _ko:
    print("\nDEUX COMPTABLES POUR LE MEME TRADE. L'etiquette est ce que les")
    print("tetes apprennent ; si elle ne decrit pas ce que le compte paie,")
    print("le modele optimise une strategie que personne ne joue.")
sys.exit(1 if _ko else 0)
