"""Ce qui differe d'un instrument a l'autre — et ce qui n'en differe pas.

CE QUI EST PARTAGE. Les 256 colonnes de features, le reseau, le compte. Les
colonnes se calculent a partir d'un OHLCV pur depuis que les quatre colonnes
de carnet ont ete retirees, donc n'importe quel instrument les produit ; le
retrait a ete mesure gratuit (rho +0.0432 -> +0.0426).

CE QUI NE L'EST PAS, et qu'aucune valeur ne doit etre copiee d'un instrument
a l'autre :

  LA LARGEUR DU STOP. L'ATR relatif vaut 15.9 points de base sur le BTC et
  6.6 sur l'or. Un stop de 6xATR fait donc 95 bps sur l'un et 40 sur l'autre —
  des trades qui n'ont rien a voir. Chaque largeur vient de son propre
  balayage, hors test.

  LA FRICTION. Spread releve chez le courtier : 2.23 points de base sur le
  BTC, 0.68 sur l'or. Appliquer celle du BTC a l'or surestimait sa friction de
  moitie, et la sienne au BTC la sous-estimerait d'autant.

  LA TAILLE DU CONTRAT, et c'est le piege le moins visible. Le contrat de l'or
  vaut 100 ONCES : son lot minimum represente 4 265 $ de notionnel contre 762
  pour le BTC, soit 5.6 fois plus. A 1 000 EUR de capital, une position
  minimale sur l'or risque donc 2.8 % contre 0.72 % pour le BTC — et le budget
  de risque partage n'autorise qu'UNE position d'or. Les deux instruments ne
  jouent pas du tout de la meme facon sur le meme compte : le BTC en essaim,
  l'or a l'unite.

LA MARGE, ELLE, EST LA MEME : 0.174 % du notionnel des deux cotes, mesure par
`order_calc_margin`. Ce n'est jamais elle qui borne.
"""

from __future__ import annotations

INSTRUMENTS = {
    "BTCUSD": {
        "cache": "data_cache_BTCUSD_M5.pkl",
        # Balayage du 2026-09-16, rendement PAR AN hors test : 6x donne
        # +12.5 R/an contre +8.4 a 12x, et reste positif jusqu'a trois fois
        # la friction supposee.
        "atr_sl_mult": 6.0,
        "trail_R": 2.0,          # plateau mesure, declenchement et distance
        "spread_bps": 2.23,
        # LA LOI DU SPREAD, ET PAS SEULEMENT SA VALEUR CENTRALE.
        #
        # Le spread n'est pas constant : il s'elargit la nuit et sur les
        # annonces. Le simulateur le TIRE de cette loi bimodale a chaque
        # trade, une fois, et le facture a l'aller comme au retour.
        #
        # ELLE VIT ICI DEPUIS LE 2026-09-21, ET C'EST UNE CORRECTION.
        # Elle etait dans `PPOConfig` seulement, et `cibles_m1` prenait la
        # valeur centrale comme si le spread etait fixe : les etiquettes
        # facturaient 1.36 bps de spread quand le compte en payait 1.77.
        "spread_wide_prob": 0.30,
        "spread_bps_wide_factor": 2.83,
        # LE GLISSEMENT, A L'ENTREE SEULEMENT, tire uniformement dans
        # [0, entry_slippage_bps]. Son esperance vaut donc la moitie.
        # `cibles_m1` facturait 3.0 bps en dur, six fois trop.
        "entry_slippage_bps": 1.0,
        "lot_min": 0.01,
        "lot_pas": 0.01,
        "contrat": 1.0,
        "marge_frac": 0.001745,
        "atr_rel_bps": 15.9,
    },
    "XAUUSD": {
        "cache": "data_cache_XAUUSD_M5.pkl",
        # Balayage CONJOINT stop x trailing sur GPU, 35 combinaisons, hors
        # test : 4xATR avec un trailing de 6 R donne +42.5 R/an sous friction
        # majoree contre +16.2 pour 10x/2R. Le gain vient du NOMBRE de trades
        # — 63 par an contre 31 — la duree tombant de 27.6 h a 6.8 h.
        #
        # Sans trailing la part symetrique vaut EXACTEMENT zero a toutes les
        # largeurs : chaque trade perd alors 1 R. Le trailing n'est pas un
        # reglage d'appoint, c'est le seul mecanisme qui gagne.
        #
        # 4.6 % des reouvertures franchissent un stop de 26 points de base :
        # un cout que  ne modelise pas.
        "atr_sl_mult": 10.0,
        # LE TRAILING PASSE DE 2.0R A 0.5R — POUR LA MESURE, PAS POUR LE GAIN.
        #
        # LE PROBLEME QU'IL RESOUT, et il prime sur tous les autres. Une
        # position tenait 332 barres en mediane. Sur une fenetre de
        # validation de 47 908 barres, le sommet a 5 % retenait 163 occasions
        # dont ~22 DISJOINTES seulement. La barre de bruit d'un `sommet`
        # valait alors 0.19 a 0.63 R selon le groupement du score — mesuree
        # par rotation circulaire de la cible sous un score reel, donc en
        # conservant son autocorrelation.
        #
        # TOUT exec69 TENAIT SOUS UN ECART-TYPE DE BRUIT : exces +0.299
        # (0.5 sigma), meilleur quart +0.594 (0.95 sigma), rho +0.044
        # (0.74 sigma), rhoAux +0.075 (1.27 sigma). On ne pouvait donc ni
        # retenir un checkpoint, ni comparer deux architectures, ni choisir
        # des colonnes : aucune de ces decisions n'avait de resolution.
        #
        # RACCOURCIR PAR LE TRAILING, JAMAIS PAR LE STOP. Resserrer le stop
        # de 10x a 2.5xATR raccourcit autant, mais QUADRUPLE la friction —
        # 0.223 R par aller-retour, 22 % de chaque R devore par le spread,
        # qui lui ne bouge pas. Le trailing laisse le R defini par le meme
        # stop, donc la friction reste a 0.056.
        #
        # LE BALAYAGE, sur la fenetre d'entrainement du fold 1. `plafond` est
        # ce qu'encaisserait une tete qui apprendrait la cible PARFAITEMENT ;
        # `sigma` est son exces divise par la barre de bruit, et c'est la
        # seule grandeur qui decide :
        #
        #   trailing  duree  hasard  plafond  exces  bruit  sigma
        #     2.00R     332  +0.154  +2.025  +1.872  0.191    9.8
        #     1.50R     280  +0.106  +1.983  +1.876  0.149   12.6
        #     1.00R     202  +0.025  +1.460  +1.435  0.093   15.4
        #     0.50R     100  -0.037  +0.674  +0.711  0.042   16.8  <--
        #     0.25R      43  -0.048  +0.288  +0.336  0.021   15.7
        #
        # CE QU'ON PERD, ET IL FAUT LE DIRE. Le lien entre la cible apprise
        # — « touche-t-il 2 R avant -1 R », qui ne depend pas du trailing —
        # et le rendement encaisse se degrade : correlation de rang +0.955 a
        # 2.0R, +0.556 a 0.5R. Le plafond atteignable tombe de 1.87 a 0.71 R.
        # Mais le bruit tombe plus vite, et c'est le rapport qui commande.
        #
        # CE QU'ON GAGNE, ET C'EST LE POINT. Le hasard passe de +0.154 a
        # -0.037 : la geometrie devient NEUTRE. Sous 2.0R le modele devait
        # battre une cible mouvante faite de derive ; desormais tout exces
        # mesure est de la SELECTION PURE.
        #
        # L'ARITHMETIQUE QUI JUSTIFIE LE CHANGEMENT. Le modele a atteint 16 %
        # du plafond au fold 1 (+0.299 sur +1.872). A competence EGALE sous
        # 0.5R il rendrait +0.114 pour un bruit de 0.042, soit 2.7 sigma —
        # visible. La meme competence est aujourd'hui invisible.
        #
        # LE BTC GARDE 2.0R : cette mesure porte sur l'or, et recopier un
        # reglage sur un instrument qu'on n'a pas mesure est exactement ce
        # que ce depot se reproche ailleurs.
        # 0.5R -> 1.0R, LE MEME JOUR. LE SIGNAL ETAIT TROP PETIT A APPRENDRE.
        #
        # 0.5R avait ete choisi sur le meilleur rapport signal/bruit du
        # balayage — 16.8 sigma contre 15.4 pour 1.0R — et il a bien rendu
        # la mesure nette. Mais deux mesures ensuite ont montre le cout :
        #
        #   LE POINT MORT. Le trailing coupe les gains a 0.525 R pendant que
        #   les pertes restent a 0.841 R : ratio 0.62, donc il faut 65.7 %
        #   de trades gagnants pour ne rien perdre. Le modele tourne a
        #   64.4 % — pile sous le seuil, PnL a zero, PF 1.05.
        #
        #   LE PLAFOND. Une tete PARFAITE n'obtient que +0.65 R par occasion
        #   sous 0.5R, contre +1.40 sous 1.0R et +2.03 sous 2.0R. Le signal
        #   a extraire est trois fois plus petit, et la tete de rang cesse
        #   d'apprendre : exces plat a +0.037 sur douze epochs, `net` plat a
        #   -0.475 depuis l'epoch 2, etendue des scores divisee par huit.
        #
        # LE BALAYAGE COMPLET, pour situer 1.0R :
        #
        #   trailing  duree  hasard  plafond  exces  bruit  sigma  pt mort
        #     2.00R     332  +0.154  +2.025  +1.872  0.191    9.8   30.3 %
        #     1.00R     202  +0.025  +1.460  +1.435  0.093   15.4   50.3 %  <--
        #     0.50R     100  -0.037  +0.674  +0.711  0.042   16.8   65.7 %
        #
        # 1.0R GARDE L'ESSENTIEL DE CE QU'ON CHERCHAIT : le hasard tombe a
        # +0.025 — donc la geometrie reste NEUTRE et tout exces mesure est
        # de la selection, ce qui etait le but — et le bruit reste deux fois
        # plus bas qu'a 2.0R. On perd 1.4 sigma de rapport pour recuperer un
        # signal deux fois plus gros et un point mort quinze points plus bas.
        "trail_R": 1.0,
        "spread_bps": 0.68,
        # LA LOI DU SPREAD, ET PAS SEULEMENT SA VALEUR CENTRALE.
        #
        # Le spread n'est pas constant : il s'elargit la nuit et sur les
        # annonces. Le simulateur le TIRE de cette loi bimodale a chaque
        # trade, une fois, et le facture a l'aller comme au retour.
        #
        # ELLE VIT ICI DEPUIS LE 2026-09-21, ET C'EST UNE CORRECTION.
        # Elle etait dans `PPOConfig` seulement, et `cibles_m1` prenait la
        # valeur centrale comme si le spread etait fixe : les etiquettes
        # facturaient 1.36 bps de spread quand le compte en payait 1.77.
        "spread_wide_prob": 0.30,
        "spread_bps_wide_factor": 2.83,
        # LE GLISSEMENT, A L'ENTREE SEULEMENT, tire uniformement dans
        # [0, entry_slippage_bps]. Son esperance vaut donc la moitie.
        # `cibles_m1` facturait 3.0 bps en dur, six fois trop.
        "entry_slippage_bps": 1.0,
        "lot_min": 0.01,
        "lot_pas": 0.01,
        "contrat": 100.0,        # 1 lot = 100 onces — le piege
        "marge_frac": 0.001744,
        "atr_rel_bps": 6.6,
    },
}


def config_instrument(cfg_base, symbole: str):
    """Une copie de la configuration, reglee pour CET instrument.

    Les deux instruments partagent le reseau et le compte, jamais la
    geometrie. On copie donc la configuration plutot que de la muter : deux
    environnements qui liraient le meme objet se marcheraient dessus, et
    l'erreur serait silencieuse — chacun poserait les stops de l'autre.
    """
    import copy
    if symbole not in INSTRUMENTS:
        raise KeyError(f"instrument inconnu : {symbole} "
                       f"(connus : {sorted(INSTRUMENTS)})")
    p = INSTRUMENTS[symbole]
    c = copy.copy(cfg_base)
    c.atr_sl_mult = float(p["atr_sl_mult"])
    c.atr_trail_mult = float(p["trail_R"]) * c.atr_sl_mult
    c.atr_trail_dist = float(p["trail_R"]) * c.atr_sl_mult
    c.atr_tp_mult = 6.0 * c.atr_sl_mult          # inutilise, use_tp = False
    c.aux_tp_mult = 2.0 * c.atr_sl_mult          # la cible du classement
    c.spread_bps = float(p["spread_bps"])
    c.lot_min = float(p["lot_min"])
    c.lot_pas = float(p["lot_pas"])
    c.marge_frac = float(p["marge_frac"])
    c.contrat = float(p["contrat"])
    c.symbole = symbole
    return c


def risque_lot_min(symbole: str, prix: float, capital: float) -> float:
    """Part du capital qu'un lot MINIMUM met en jeu, en pourcentage.

    C'est le chiffre qui decide du nombre de positions tenables, et il ne se
    devine pas : il depend de la taille du contrat autant que du stop.
    """
    p = INSTRUMENTS[symbole]
    notionnel = p["lot_min"] * p["contrat"] * prix
    stop_frac = p["atr_sl_mult"] * p["atr_rel_bps"] / 1e4
    return 100.0 * notionnel * stop_frac / max(capital, 1e-9)


def main() -> int:
    print(f"{'instrument':<10} {'stop':>6} {'risque':>8} {'friction':>9} "
          f"{'lot min':>10} {'risque/lot min':>15}")
    print("-" * 64)
    for sym, p in INSTRUMENTS.items():
        prix = {"BTCUSD": 76189.0, "XAUUSD": 4265.0}[sym]
        risque_bps = p["atr_sl_mult"] * p["atr_rel_bps"]
        fric = (p["spread_bps"] + 3.0) / risque_bps
        notionnel = p["lot_min"] * p["contrat"] * prix
        print(f"{sym:<10} {p['atr_sl_mult']:>5g}x {risque_bps:>7.0f}b "
              f"{fric:>8.3f}R {notionnel:>9,.0f}$ "
              f"{risque_lot_min(sym, prix, 1000.0):>14.2f}%")
    print("\nA 1 000 EUR et 3 % de budget de risque PARTAGE, l'or ne tient")
    print("qu'une position quand le BTC en tient quatre. Ce n'est pas un")
    print("reglage : c'est la taille du contrat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def spread_espere(symbole: str) -> float:
    """Le spread MOYEN d'un franchissement, sous la loi bimodale. En bps.

    Le simulateur tire `base` avec la probabilite `1 - wide_prob`, et
    uniformement dans `[1.2 x base, wide_factor x base]` sinon. L'esperance
    de la seconde branche est donc `base x (1.2 + wide_factor) / 2`.
    """
    p = INSTRUMENTS[symbole]
    base = float(p["spread_bps"])
    q = float(p["spread_wide_prob"])
    f = float(p["spread_bps_wide_factor"])
    return (1.0 - q) * base + q * base * (1.2 + f) / 2.0


def cout_aller_retour_espere(symbole: str) -> float:
    """CE QUE COUTE UN TRADE COMPLET, en points de base. Source unique.

    POURQUOI CETTE FONCTION EXISTE — un desaccord mesure le 2026-09-21.
    Deux endroits comptaient la friction et ne trouvaient pas pareil :

        l'etiquette (`cibles_m1`)   4.36 bps
        le simulateur              2.27 bps        soit 1.92x

    Deux fautes dans l'etiquette. Elle prenait le spread a sa valeur
    CENTRALE — 0.68 — alors que le simulateur le tire d'une loi bimodale
    dont l'esperance vaut 0.887. Et elle facturait 3.0 bps de glissement,
    a l'aller ET au retour, quand le simulateur en tire un uniformement
    dans [0, 1.0] et SEULEMENT a l'entree.

    CE QUE L'ECART COUTAIT. L'etiquette est ce que les tetes apprennent a
    predire. Sur un mouvement brut de +3.0 bps, elle annonce -1.36 —
    « perdant, ne pas prendre » — quand le compte, lui, encaisse +0.73.
    Mesure sur la fenetre de validation, 7 146 occasions d'achat : 415
    d'entre elles (5.8 %) sont declarees perdantes alors qu'elles paient,
    pour +1.01 bps en moyenne. La part d'occasions gagnantes passe de
    41.0 % annoncees a 46.8 % reelles.

    ON APPRENAIT DONC AU MODELE A S'ABSTENIR DEUX FOIS TROP.

    LE SPREAD COMPTE DEUX FOIS, le glissement une. C'est la geometrie du
    simulateur : il tire le spread UNE fois par trade et le facture a
    l'ouverture comme a la fermeture, tandis que le glissement ne frappe
    qu'a l'entree.

    RESERVE A GARDER EN TETE : la loi bimodale a ete calibree sur les logs
    MT5 Vantage du BTCUSD et appliquee telle quelle a l'or. Les deux
    instruments portent donc la meme forme de loi pour des spreads de base
    differents. Ce n'est pas mesure sur l'or.
    """
    return 2.0 * spread_espere(symbole) + float(
        INSTRUMENTS[symbole]["entry_slippage_bps"]) / 2.0
