"""Regles de taille et replay chronologique du compte KAIROS.

Les indices ``t`` sont des decisions a la cloture : entree a l'ouverture
``t + 1``, sortie sur ``t + du``. Un timeout sort a l'ouverture, un TP/SL
pendant la barre. Les drawdowns equity sont mesures aux clotures de bougie,
pas aux ticks ni aux extremes intrabar. Le compte du jeu beneficie d'une
protection de solde negatif : equity et solde ne passent jamais sous zero.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _tailles(equite, dist, prix, cfg, marge_libre=None, limite_risque=True):
    eq, distance, prix = np.broadcast_arrays(
        np.asarray(equite, dtype=float), np.asarray(dist, dtype=float),
        np.asarray(prix, dtype=float))
    contrat, pas, minimum, levier = (float(getattr(cfg, k)) for k in
                                   ("contrat", "pas_lot", "lot_min", "levier"))
    maximum = float(getattr(cfg, "lot_max", np.inf))
    if min(contrat, pas, minimum, levier) <= 0 or not np.isfinite(
            [contrat, pas, minimum, levier, cfg.risque_pct]).all():
        raise ValueError("Contrat, pas, lot minimum et levier doivent etre positifs et finis.")
    marge = eq if marge_libre is None else np.broadcast_to(
        np.asarray(marge_libre, dtype=float), eq.shape)
    valide = (np.isfinite(eq) & np.isfinite(distance) & np.isfinite(prix)
              & (eq > 0) & (distance > 0) & (prix > 0) & (marge > 0))
    budget = np.maximum(eq, 0) * max(float(cfg.risque_pct), 0) / 100.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        plafond_risque = budget / (distance * contrat) if limite_risque else np.full(eq.shape, np.inf)
        max_lots = np.minimum(np.minimum(plafond_risque,
                                         marge * levier / (prix * contrat)), maximum)
        lots = np.floor(max_lots / pas + 1e-10) * pas
        valide &= ((lots + 1e-12 >= minimum)
                   & ((not limite_risque) | (lots * distance * contrat <= budget + 1e-10))
                   & (lots * prix * contrat / levier <= marge + 1e-10))
    return np.where(valide, lots, 0.0)


def taille_position(equite, dist, prix, cfg, marge_libre=None, limite_risque=True) -> float:
    """Lots autorises, arrondis vers le bas ; jamais de remontage au lot minimum.

    Zero signifie que le risque ou la marge ne permettent pas le lot minimum.
    La marge peut reduire la taille. La fonction scalaire et le masque utilisent
    exactement le meme calcul.
    """
    return float(_tailles(equite, dist, prix, cfg, marge_libre, limite_risque))


def masque_stops(dist, prix, equite, cfg, marge_libre=None) -> np.ndarray:
    """Masque des stops finançables (broadcasting NumPy ; equity B x 1 pour B x K)."""
    return _tailles(equite, dist, prix, cfg, marge_libre) > 0


class _Drawdown:
    def __init__(self, capital):
        self.pic = float(capital)
        self.dollars = self.pct = 0.0
        self.minimum = float(capital)

    def ajoute(self, valeurs):
        # Meme protection que le rollout : une liquidation ne peut pas
        # transformer un capital positif en solde negatif, ni produire un
        # drawdown inferieur a -100 % dans les statistiques.
        v = np.maximum(np.asarray(valeurs, dtype=float).reshape(-1), 0.0)
        if not len(v):
            return
        pics = np.maximum.accumulate(np.maximum(v, self.pic))
        self.dollars = min(self.dollars, float(np.min(v - pics)))
        self.pct = min(self.pct, float(np.min((v - pics) / np.maximum(pics, 1e-12))))
        self.minimum = min(self.minimum, float(v.min()))
        self.pic = float(pics[-1])


def compte_live(t, j, r, du, close, atr, cfg, *, sens=None, open_=None,
                spread=None, sorties=None, temps=None):
    """Rejoue les trades sur un compte, avec risque et capacite stricts.

    ``sens`` utilise 0 = achat, 1 = vente. ``spread`` est en bps, comme
    ``table_coups``. ``sorties`` utilise 2 = timeout, 0/1 = TP/SL ; sans
    cette information, toutes les sorties sont traitees comme intrabar.
    Les resultats ``r`` incluent deja les couts et swaps : ils ne sont jamais
    factures deux fois. Les entrees utilisent open_[t+1] quand disponible.

    Fournir sens, open_ et spread ensemble active l'equity aux clotures, avec
    une valeur de liquidation bid/ask apres glissement et swap. ``temps`` est
    accepte pour les consommateurs, sans changer la convention de swap par
    nombre de bougies de table_coups. Les anciennes cles live_dd_* sont des
    alias explicites du drawdown du SOLDE realise, pas de l'equity.
    """
    t, j, du = (np.asarray(v, dtype=np.int64) for v in (t, j, du))
    r, close, atr = (np.asarray(v, dtype=float) for v in (r, close, atr))
    n = len(t)
    if any(len(v) != n for v in (j, r, du)) or len(atr) != len(close):
        raise ValueError("Longueurs de trades ou de prix incompatibles.")
    if np.any(t < 0) or np.any(du < 1) or np.any(t + du >= len(close)):
        raise ValueError("Un trade depasse les donnees ou a une duree invalide.")
    if not np.isfinite(r).all() or not np.isfinite(close).all():
        raise ValueError("Prix et rendements doivent etre finis.")
    sortie = np.zeros(n, dtype=int) if sorties is None else np.asarray(sorties, dtype=int)
    if len(sortie) != n or np.any((sortie < 0) | (sortie > 2)):
        raise ValueError("Sorties attendues : 0 (TP), 1 (SL), 2 (temps).")
    ksl = np.asarray(cfg.sl_atr, dtype=float)
    if np.any(j < 0) or np.any(j >= len(ksl)):
        raise ValueError("Indice de stop invalide.")
    mtm = sens is not None and open_ is not None and spread is not None
    sens_a = None if sens is None else np.asarray(sens, dtype=int)
    oa = None if open_ is None else np.asarray(open_, dtype=float)
    spa = None if spread is None else np.asarray(spread, dtype=float)
    if oa is not None and (len(oa) != len(close) or not np.isfinite(oa).all()):
        raise ValueError("Ouvertures incompatibles ou non finies.")
    if spa is not None and (len(spa) != len(close) or not np.isfinite(spa).all()):
        raise ValueError("Spreads incompatibles ou non finis.")
    if sens_a is not None and (len(sens_a) != n or np.any((sens_a < 0) | (sens_a > 1))):
        raise ValueError("Sens attendu : 0 achat, 1 vente.")
    if temps is not None and len(temps) != len(close):
        raise ValueError("Dates incompatibles avec les prix.")
    E = float(cfg.capital)
    if E <= 0 or not np.isfinite(E):
        raise ValueError("Capital initial positif et fini requis.")
    capacite = int(getattr(cfg, "positions_max", 1))
    if capacite < 1:
        raise ValueError("positions_max doit etre au moins 1.")
    realise, equity = _Drawdown(E), _Drawdown(E)
    entrees, fermetures = defaultdict(list), defaultdict(list)
    for k in np.argsort(t, kind="stable"):
        entrees[int(t[k] + 1)].append(int(k))
    evenements = sorted(set(entrees) | set((t + du).tolist()))
    ouverts, risques = {}, []
    pris = refus_marge = refus_risque = refus_places = points_equity = 0
    contrat = float(cfg.contrat)
    se = float(getattr(cfg, "glissement_entree_bps", 0.0)) / 1e4
    sx = float(getattr(cfg, "glissement_sortie_bps", 0.0)) / 1e4

    def cloture(barre, phase):
        nonlocal E
        pnl = 0.0
        faits = False
        for k in fermetures.get((barre, phase), ()):
            position = ouverts.pop(k)
            pnl += position["pnl"]
            faits = True
        if faits:
            E = max(0.0, E + pnl)
            realise.ajoute([E])

    def marque(a, b):
        nonlocal points_equity
        if not mtm or a > b:
            return
        points_equity += b - a + 1
        if not ouverts:
            equity.ajoute([E])
            return
        idx = np.arange(a, b + 1)
        eq = np.full(len(idx), E, dtype=float)
        for k, p in ouverts.items():
            achat = sens_a[k] == 0
            px = close[idx] * (1 - sx) if achat else close[idx] * (1 + spa[idx] / 1e4) * (1 + sx)
            swap = float(getattr(cfg, "swap_achat_bps_jour" if achat else
                                 "swap_vente_bps_jour", 0.0))
            frais_swap = swap / 1e4 * p["prix_entree"] * (idx - t[k]) * float(
                getattr(cfg, "minutes_par_barre", 1)) / 1440.0
            latent = ((1 if achat else -1) * (px - p["prix_entree"]) - frais_swap)
            eq += latent * p["lots"] * contrat
        equity.ajoute(eq)

    curseur = evenements[0] if evenements else 0
    for barre in evenements:
        marque(curseur, barre - 1)
        # Timeout a l'ouverture : argent et place disponibles a cet open.
        cloture(barre, 0)
        for k in entrees.get(barre, ()):
            if len(ouverts) >= capacite:
                refus_places += 1
                continue
            dist = ksl[j[k]] * atr[t[k]]
            prix = float(oa[barre] if oa is not None else close[t[k]])
            libre = E - sum(p["marge"] for p in ouverts.values())
            # Distinguer lot minimum trop risque et manque de marge.
            nominal = taille_position(E, dist, prix, cfg, marge_libre=np.inf)
            if nominal == 0:
                refus_risque += 1
                continue
            lots = taille_position(E, dist, prix, cfg, marge_libre=libre)
            if lots == 0:
                refus_marge += 1
                continue
            risque = lots * contrat * dist
            p0 = prix
            if mtm:
                p0 *= (1 + spa[t[k]] / 1e4 + se) if sens_a[k] == 0 else (1 - se)
            ouverts[k] = {"pnl": float(r[k]) * risque, "lots": lots,
                          "marge": lots * contrat * prix / float(cfg.levier),
                          "prix_entree": p0}
            fermetures[(int(t[k] + du[k]), 0 if sortie[k] == 2 else 1)].append(k)
            risques.append(risque / E)
            pris += 1
        # TP/SL : aucune place liberee retroactivement a l'ouverture.
        cloture(barre, 1)
        curseur = barre
    if evenements:
        marque(curseur, curseur)
    resultat = {"live_total": float(E - cfg.capital),
                "live_dd_dollars": realise.dollars, "live_dd_pct": realise.pct,
                "live_dd_realise_dollars": realise.dollars, "live_dd_realise_pct": realise.pct,
                "live_dd_type": "solde_realise", "live_pris": pris,
                "live_marge": refus_marge, "live_positions": refus_places,
                "live_risque_refuses": refus_risque,
                "live_risque_moy": float(np.mean(risques)) if risques else float("nan"),
                "live_risque_max": float(np.max(risques)) if risques else float("nan"),
                "live_equity_disponible": bool(mtm)}
    if mtm:
        resultat.update(live_equity_dd_dollars=equity.dollars,
                        live_equity_dd_pct=equity.pct,
                        live_equity_min=equity.minimum,
                        live_equity_points=points_equity,
                        live_equity_mode="liquidation_aux_clotures_bougies")
    return resultat
