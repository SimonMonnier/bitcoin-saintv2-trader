# -*- coding: utf-8 -*-
"""Quel instrument ajouter a l'or ? Trois criteres, dans l'ordre qui compte.

POURQUOI CETTE MESURE. Le mur du projet n'est ni l'architecture ni
l'algorithme : c'est le nombre d'OCCASIONS INDEPENDANTES. Une position dure
~4 jours, l'historique fait 7 ans, donc ~930 occasions — et rien d'autre.
L'erreur-type sur le R moyen vaut alors 0.065, ce qui rend INVISIBLE tout
avantage sous +0.15 R. Un vrai avantage de +0.10 R se lirait « entre -0.03 et
+0.23 » et serait jete.

Un second instrument DECORRELE n'ajoute pas un pari : il ajoute une
EXPERIENCE. Ses occasions ne recouvrent pas celles de l'or, donc elles
comptent pour de nouvelles preuves.

    or seul      930 occasions   erreur-type 0.065   visible au-dela de +0.15 R
    + 3 autres  ~3 700          erreur-type 0.033   visible au-dela de +0.077 R

LES TROIS CRITERES, DANS CET ORDRE.

  1. LA FRICTION PAR R. C'est le filtre qui elimine, pas la correlation. Un
     instrument qui mange 0.05 R par trade exige +0.05 R d'avantage
     supplementaire AVANT de rapporter quoi que ce soit — et on cherche des
     avantages de l'ordre de 0.10 R. La friction n'est pas un cout annexe,
     c'est la moitie de la cible.

         friction = spread_bps / (stop_mult x atr_relatif_bps)

  2. LA CORRELATION DES RENDEMENTS DE STRATEGIE, pas des prix. Deux marches
     dont les prix bougent differemment peuvent avoir des strategies qui
     gagnent et perdent ensemble. C'est la mesure qui a valide le BTC a 0.088
     et ecarte ETH a 0.81 — ce dernier n'apportait que 10 % d'occasions
     nouvelles contre 84 % pour l'or.

  3. LES OCCASIONS DISPONIBLES = profondeur d'historique / duree de tenue.
     Un instrument decorrele et bon marche mais avec deux ans d'historique
     n'ajoute presque rien.

CE QUE LE SCRIPT NE FAIT PAS. Il ne juge pas si une strategie est rentable
sur le candidat — seulement si l'ajouter AMELIORE LA MESURE. Un instrument
peut etre parfait ici et sans avantage exploitable.

    python mesure_instruments.py            # tour d'horizon, spreads MT5
    python mesure_instruments.py complet    # + geometrie et correlations
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

# Les familles a couvrir. On veut de la DIVERSITE DE MOTEUR : ce qui fait
# bouger le prix doit differer de ce qui fait bouger l'or (taux reels,
# dollar, peur). Deux instruments du meme moteur correlent, quoi qu'en dise
# leur secteur.
FAMILLES = {
    "metal":  ["XAUUSD"],
    # LE SEUL CANDIDAT RETENU, et les alias possibles du meme sous-jacent
    # selon le courtier. On ne mesure qu'un instrument a la fois : trois
    # candidats decorreles DE L'OR peuvent etre correles ENTRE EUX, et les
    # juger ensemble masquerait ce recouvrement.
    "indice": ["NAS100", "USTEC", "NDX100", "US100", "NAS100.cash"],
    # LE BTC EST REMESURE ICI, et pas repris de l'archive. Son 0.088 vient
    # d'une mesure anterieure dont la methode n'est pas connue : le comparer
    # a un chiffre produit autrement serait exactement le genre de
    # rapprochement que ce depot paie cher. Meme code, meme agregation, meme
    # profondeur pour les trois.
    "crypto": ["BTCUSD"],
}

# La geometrie de reference. On evalue chaque candidat SOUS LA MEME REGLE que
# l'or, sinon on compare des strategies differentes et non des instruments.
STOP_MULT = 10.0
BARRES_PAR_AN = 70_752      # M5, week-ends et feries deduits (mesure du cache)


def _mt5():
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 indisponible : {mt5.last_error()}")
    return mt5


def tour_horizon():
    """Ce que le courtier propose vraiment, et a quel spread."""
    mt5 = _mt5()
    try:
        dispo = {s.name for s in mt5.symbols_get()}
        lignes = []
        for famille, noms in FAMILLES.items():
            for nom in noms:
                if nom not in dispo:
                    continue
                mt5.symbol_select(nom, True)
                info = mt5.symbol_info(nom)
                tick = mt5.symbol_info_tick(nom)
                if info is None:
                    continue
                # LE SPREAD EN POINTS DE BASE DU PRIX, la seule unite
                # comparable entre un or a 3 300 $ et un indice a 20 000.
                #
                # ON N'EXIGE PAS UN TICK VIVANT. La premiere version sautait
                # tout symbole sans cotation courante — et elle a donc ecarte
                # NAS100 en silence, simplement parce que les indices ferment
                # quand l'or est encore ouvert. Le terminal conserve
                # `info.spread` en POINTS meme marche ferme : on s'en sert en
                # repli, et on dit lequel des deux a servi.
                if tick is not None and tick.ask:
                    spread = (tick.ask - tick.bid) / tick.ask * 1e4
                    vif = "vif"
                else:
                    prix = float(info.bid or info.ask or info.last or 0.0)
                    if prix <= 0:
                        continue
                    spread = info.spread * info.point / prix * 1e4
                    vif = "decl"
                del vif
                # LA PROFONDEUR SE SONDE, ELLE NE SE TELECHARGE PAS.
                #
                # Premiere version : tirer 500 000 barres par symbole pour les
                # COMPTER. Trente symboles x 500 000 barres, et le tour
                # d'horizon devenait plus long que la mesure qu'il prepare. On
                # demande UNE barre a des profondeurs decroissantes : le
                # terminal repond vide au-dela de ce qu'il a, donc sept
                # requetes d'une barre remplacent quinze millions de lignes.
                n = 0
                for prof in (500_000, 300_000, 200_000, 120_000,
                             70_000, 35_000, 10_000):
                    rr = mt5.copy_rates_from_pos(nom, mt5.TIMEFRAME_M5,
                                                 prof - 1, 1)
                    if rr is not None and len(rr):
                        n = prof
                        break
                lignes.append((famille, nom, spread, n,
                               info.trade_contract_size, info.volume_min))
                # AU FIL DE L'EAU : un tour d'horizon muet pendant cinq
                # minutes ne dit pas s'il avance ou s'il est bloque.
                print(f"  {famille:<9} {nom:<9} {spread:>11.3f} "
                      f"{n:>11,} {n/BARRES_PAR_AN:>7.1f}  "
                      f"{info.trade_contract_size:g} "
                      f"(min {info.volume_min:g})", flush=True)
        return lignes
    finally:
        mt5.shutdown()


def geometrie(nom, n_barres=500_000):
    """ATR relatif, friction par R, occasions par an — sous la regle de l'or.

    500 000 BARRES, PAS 200 000. La premiere passe n'en tirait que 200 000,
    soit 2.8 ans sur les 7.1 disponibles — et elle ecartait donc justement
    les regimes de crise, qui sont le cas ou la correlation indice/or est
    censee basculer. Mesurer la decorrelation en excluant les periodes ou
    elle est douteuse n'aurait aucun sens.
    """
    mt5 = _mt5()
    try:
        mt5.symbol_select(nom, True)
        r = mt5.copy_rates_from_pos(nom, mt5.TIMEFRAME_M5, 0, n_barres)
        tick = mt5.symbol_info_tick(nom)
        info = mt5.symbol_info(nom)
        # MEME REPLI QUE DANS LE TOUR D'HORIZON : marche ferme, spread
        # declare. Voir la-bas pour ce que son absence a coute.
        if tick is not None and tick.ask:
            spread_bps = (tick.ask - tick.bid) / tick.ask * 1e4
        elif info is not None:
            prix = float(info.bid or info.ask or info.last or 0.0)
            spread_bps = (info.spread * info.point / prix * 1e4
                          if prix > 0 else float("nan"))
        else:
            spread_bps = float("nan")
    finally:
        mt5.shutdown()
    if r is None or len(r) < 5_000:
        return None
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    df["atr_14"] = atr
    atr_rel = float(np.nanmedian(atr / c) * 1e4)
    # La friction rapportee au RISQUE, pas au prix : c'est ce qui se compare.
    friction = spread_bps / max(STOP_MULT * atr_rel, 1e-9)
    ans = len(df) / BARRES_PAR_AN
    return {"nom": nom, "df": df, "barres": len(df), "ans": ans,
            "spread_bps": spread_bps, "atr_rel_bps": atr_rel,
            "friction_R": friction}


def serie_rendements(g, pas=12):
    """Le R d'un achat ouvert a chaque occasion, sous la regle de l'or.

    MEME REGLE POUR TOUS : stop a 10 x ATR, sortie sur barriere. Comparer un
    candidat sous une autre geometrie reviendrait a comparer deux strategies
    et non deux instruments.
    """
    import cibles as CIB

    # LES NOMS SONT CEUX QUE `cibles._regle` LIT, pas ceux de `kairos_live`.
    # La premiere version portait `trailing_start_atr_mult` — le nom du live —
    # et la regle attend `atr_trail_mult`. Elle a donc plante au premier
    # appel. Les valeurs sont celles de `PPOConfig` : meme geometrie pour les
    # deux instruments, sinon on compare deux strategies et non deux marches.
    class _Cfg:
        atr_sl_mult = STOP_MULT
        use_tp = False
        atr_tp_mult = 60.0
        atr_be_mult = 0.0         # pas de break-even : mesure a 0.4 sigma
        atr_trail_mult = 20.0     # declenchement a 2.0 R
        atr_trail_dist = 20.0     # distance 2.0 R
        spread_bps = 0.0          # la friction est jugee a part, en critere 1
        slippage_bps = 0.0

    df = g["df"]
    idx = np.arange(14, len(df) - CIB.BORNE_DEFAUT - 2, pas)
    if len(idx) < 500:
        return None, None
    ra, _ = CIB.rendements(df, idx, _Cfg())
    return df["time"].to_numpy()[idx], ra


def main() -> int:
    complet = len(sys.argv) > 1 and sys.argv[1].startswith("c")

    print("\nCE QUE LE COURTIER PROPOSE")
    print("=" * 84)
    print(f"  {'famille':<9} {'symbole':<9} {'spread bps':>11} "
          f"{'barres M5':>11} {'annees':>7}  contrat", flush=True)
    lignes = tour_horizon()
    if not lignes:
        print("  aucun symbole lisible — MT5 est-il connecte ?")
        return 1
    # (chaque ligne s'est affichee au fil du sondage)

    if not complet:
        print("\n`python mesure_instruments.py complet` pour la geometrie et")
        print("les correlations — plus long, il rejoue la regle de sortie.")
        return 0

    print("\n\nGEOMETRIE SOUS LA REGLE DE L'OR (stop 10 x ATR)")
    print("=" * 84)
    print("  La friction est le filtre qui elimine : on cherche des avantages")
    print("  de l'ordre de 0.10 R, donc 0.05 R de friction en mange la moitie.")
    print()
    geos = {}
    for _, nom, _, n, _, _ in sorted(lignes):
        if n < 50_000:
            continue
        g = geometrie(nom)
        if g is None:
            continue
        geos[nom] = g
        print(f"  {nom:<9} spread {g['spread_bps']:>7.3f} bps   "
              f"ATR {g['atr_rel_bps']:>6.2f} bps   "
              f"friction {g['friction_R']:>7.4f} R   "
              f"{g['ans']:>4.1f} ans")

    if "XAUUSD" not in geos:
        print("\n  XAUUSD absent : impossible de correler a la reference.")
        return 1

    print("\n\nCORRELATION DES RENDEMENTS DE STRATEGIE AVEC L'OR")
    print("=" * 84)
    print("  Ce qui compte n'est pas la correlation des PRIX mais celle des")
    print("  rendements de la STRATEGIE : deux marches qui bougent")
    print("  differemment peuvent gagner et perdre ensemble.")
    print()
    t_or, r_or = serie_rendements(geos["XAUUSD"])
    if r_or is None:
        print("  serie de reference trop courte.")
        return 1
    s_or = pd.Series(r_or, index=pd.DatetimeIndex(t_or)).dropna()

    res = []
    for nom, g in geos.items():
        if nom == "XAUUSD":
            continue
        t, r = serie_rendements(g)
        if r is None:
            continue
        s = pd.Series(r, index=pd.DatetimeIndex(t)).dropna()
        # Alignement a l'heure : les seances different d'un marche a l'autre,
        # et seules les occasions SIMULTANEES disent si les deux flux bougent
        # ensemble.
        a = s_or.resample("1h").mean()
        b = s.resample("1h").mean()
        j = pd.concat([a, b], axis=1, join="inner").dropna()
        if len(j) < 500:
            continue
        # CE RHO EST UN MAJORANT, et il faut le savoir pour le lire.
        # Moyenner par heure lisse le bruit propre a chaque marche et laisse
        # les facteurs communs : la correlation mesuree est donc plus haute
        # que celle des occasions prises une a une. C'est acceptable tant que
        # TOUS les candidats subissent le meme traitement — ce qui est le cas
        # ici, et ce qui ne l'etait pas quand on comparait a un 0.088
        # d'archive.
        rho = float(j.corr().iloc[0, 1])
        # La correlation BRUTE, sans agregation, sur les occasions communes
        # a la minute pres. Plus bruitee, mais sans le biais de lissage.
        jb = pd.concat([s_or, s], axis=1, join="inner").dropna()
        rho_brut = (float(jb.corr().iloc[0, 1]) if len(jb) > 200
                    else float("nan"))
        occ = g["ans"] * 365.25 / 4.0        # tenue mediane ~4 jours
        res.append((nom, rho, g["friction_R"], occ, len(j), rho_brut))

    print(f"  {'symbole':<9} {'rho horaire':>12} {'rho brut':>10} "
          f"{'friction R':>12} {'occ/an':>8} {'heures':>9}  "
          f"{'obs. effectives':>16}")
    for nom, rho, fr, occ, n, rb in sorted(res, key=lambda x: abs(x[1])):
        # CE QUE LE CANDIDAT AJOUTE VRAIMENT. Deux flux correles a rho
        # donnent n_eff = 2n / (1 + rho) observations independantes, pas 2n.
        # C'est ce nombre qui decide, pas la friction.
        gain = 2.0 / (1.0 + max(rho, -0.99)) - 1.0
        print(f"  {nom:<9} {rho:>+12.3f} {rb:>+10.3f} {fr:>12.4f} "
              f"{occ:>8.0f} {n:>9,}  {100*gain:>+14.0f} %")

    print("\n\nCLASSEMENT — decorrelation ET friction, pas l'une sans l'autre")
    print("=" * 84)
    print("  Score = |rho| + friction/0.02 . Les deux comptent a parts egales :")
    print("  0.02 R de friction est le repere de l'or (0.68 bps / 66 bps).")
    print()
    for nom, rho, fr, occ, n, rb in sorted(res,
                                           key=lambda x: abs(x[1]) + x[2]/0.02):
        print(f"  {abs(rho) + fr/0.02:>6.2f}  {nom:<9} "
              f"rho {rho:>+6.3f}   friction {fr:.4f} R   {occ:>4.0f} occ/an")
    print("\n  Les trois du haut sont les candidats. Verifier qu'ils ne")
    print("  partagent pas le meme MOTEUR entre eux — trois paires en dollar")
    # NOTE : trois candidats decorreles DE L'OR peuvent etre correles ENTRE
    # EUX, et l'etendue effective serait alors bien plus faible que 4x.
    print("  correleraient entre elles meme si chacune l'est peu avec l'or.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
