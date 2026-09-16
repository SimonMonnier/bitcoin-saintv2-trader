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
        "lot_min": 0.01,
        "lot_pas": 0.01,
        "contrat": 1.0,
        "marge_frac": 0.001745,
        "atr_rel_bps": 15.9,
    },
    "XAUUSD": {
        "cache": "data_cache_XAUUSD_M5.pkl",
        # Balayage propre a l'or : +16.5 R/an a 10x contre +20.1 a 6x, mais
        # deux fois moins de sauts de reouverture franchissant le stop
        # (1.6 % contre 3.1 %) et une friction de 0.054 R contre 0.090.
        "atr_sl_mult": 10.0,
        "trail_R": 2.0,
        "spread_bps": 0.68,
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
