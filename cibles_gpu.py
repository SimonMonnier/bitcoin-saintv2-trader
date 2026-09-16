"""La course aux barrieres sur GPU — meme trade que `cibles`, en cent fois moins.

POURQUOI. `cibles.rendements` boucle en Python sur les entrees : chaque
balayage de largeur de stop prend des minutes, et une surface a deux
parametres en prend des dizaines. Or la course est massivement parallele —
chaque entree est independante des autres — donc elle se met en tableau.

CE FICHIER EST UNE SECONDE IMPLEMENTATION DU MEME TRADE, et c'est exactement
ce que ce depot passe son temps a payer : deux descriptions qui doivent
s'accorder par convention finissent toujours par diverger sans lever
d'erreur. La parade est `verifie()` : elle rejoue les deux sur le meme
echantillon et exige l'egalite au 1e-9. Elle tourne a chaque import en mode
script, et aucun chiffre produit ici ne vaut sans elle.

LA SEMANTIQUE EST CELLE DE `cibles`, terme pour terme :
  - le stop suiveur se calcule sur le plus haut ATTEINT AVANT la barre
    courante, jamais sur celle en cours ;
  - stop et objectif touches dans la meme bougie comptent une PERTE ;
  - la friction se retranche une fois, spread plus slippage.
"""

from __future__ import annotations

import numpy as np
import torch

import cibles as C


def rendements(df, idx, cfg, borne: int = C.BORNE_DEFAUT,
               indicateur: bool = False, lot: int = 512,
               device=None, durees: bool = False):
    """(achat, vente) en unites de risque — meme resultat que `cibles`.

    `durees=True` rend en plus la duree de chaque trade, en barres. Elle n'est
    pas un supplement d'agrement : espacer des entrees non chevauchantes avec
    une duree MEDIANE commune au lieu de celle de chaque trade laisse passer
    trop d'entrees et gonfle le rendement annuel — mesure sur l'or, le meme
    reglage passait de +17 a +63 R/an selon la methode.
    """
    r = C._regle_cible(cfg) if indicateur else C._regle(cfg)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    h = torch.as_tensor(df["high"].to_numpy(np.float64), device=dev)
    l = torch.as_tensor(df["low"].to_numpy(np.float64), device=dev)
    c = torch.as_tensor(df["close"].to_numpy(np.float64), device=dev)
    atr = torch.clamp(torch.as_tensor(df["atr_14"].to_numpy(np.float64),
                                      device=dev), min=1e-9)
    n = c.numel()
    fric = (float(getattr(cfg, "spread_bps", C.SPREAD_BPS))
            + C.SLIP_ENTREE_BPS + C.SLIP_SORTIE_BPS) / 1e4
    idx = np.asarray(idx, dtype=np.int64)

    sorties, temps = [], []
    for sens in (1, -1):
        out = torch.full((len(idx),), float("nan"), dtype=torch.float64,
                         device=dev)
        dur = torch.full((len(idx),), float("nan"), dtype=torch.float64,
                         device=dev)
        for d0 in range(0, len(idx), lot):
            bloc = torch.as_tensor(idx[d0:d0 + lot], device=dev)
            m = bloc.numel()
            fin = torch.clamp(bloc + 1 + borne, max=n)
            longueur = fin - bloc - 1
            valide = longueur >= 9
            if not bool(valide.any()):
                continue
            # Fenetres (m, borne) : les positions au-dela de la fin sont
            # neutralisees par un masque, jamais lues.
            off = torch.arange(borne, device=dev).unsqueeze(0)
            pos = bloc.unsqueeze(1) + 1 + off
            dedans = (off < longueur.unsqueeze(1)) & valide.unsqueeze(1)
            pos_c = torch.clamp(pos, max=n - 1)
            hh, ll = h[pos_c], l[pos_c]

            entree = c[bloc].unsqueeze(1)
            a_e = atr[bloc].unsqueeze(1)
            dist = r["sl"] * a_e

            if sens > 0:
                tp = entree + r["tp"] * a_e if r["tp"] else None
                # Le pic vu AVANT la barre : cummax puis decalage d'un cran,
                # la premiere colonne valant le prix d'entree.
                pic = torch.cummax(torch.where(dedans, hh, entree.expand_as(hh)),
                                   dim=1).values
                pic = torch.cat([entree, pic[:, :-1]], dim=1)
                gain = pic - entree
                stop = (entree - dist).expand_as(hh).clone()
                if r["be"] is not None:
                    stop = torch.where(gain >= r["be"] * a_e,
                                       entree.expand_as(stop), stop)
                if r["ts"] is not None:
                    stop = torch.where(gain >= r["ts"] * a_e,
                                       torch.maximum(stop, pic - r["td"] * a_e),
                                       stop)
                touche_sl = (ll <= stop) & dedans
                touche_tp = ((hh >= tp) & dedans) if tp is not None else None
            else:
                tp = entree - r["tp"] * a_e if r["tp"] else None
                creux = torch.cummin(torch.where(dedans, ll, entree.expand_as(ll)),
                                     dim=1).values
                creux = torch.cat([entree, creux[:, :-1]], dim=1)
                gain = entree - creux
                stop = (entree + dist).expand_as(hh).clone()
                if r["be"] is not None:
                    stop = torch.where(gain >= r["be"] * a_e,
                                       entree.expand_as(stop), stop)
                if r["ts"] is not None:
                    stop = torch.where(gain >= r["ts"] * a_e,
                                       torch.minimum(stop, creux + r["td"] * a_e),
                                       stop)
                touche_sl = (hh >= stop) & dedans
                touche_tp = ((ll <= tp) & dedans) if tp is not None else None

            grand = borne + 1

            def premier(masque):
                """Indice du premier True, ou `grand` s'il n'y en a pas."""
                k = torch.where(masque, off.expand_as(masque),
                                torch.full_like(masque, grand, dtype=off.dtype))
                return k.min(dim=1).values

            ja = premier(touche_sl)
            jb = (premier(touche_tp) if touche_tp is not None
                  else torch.full_like(ja, grand))
            resolu = (ja < grand) | (jb < grand)
            # Egalite = PERTE : le stop l'emporte quand les deux tombent dans
            # la meme bougie, comme dans `cibles`.
            par_stop = ja <= jb
            j = torch.clamp(torch.where(par_stop, ja, jb), max=borne - 1)
            prix_stop = torch.gather(stop, 1, j.unsqueeze(1)).squeeze(1)
            prix = (prix_stop if tp is None
                    else torch.where(par_stop, prix_stop, tp.squeeze(1)))
            val = sens * (prix - entree.squeeze(1)) / dist.squeeze(1) \
                - fric * entree.squeeze(1) / dist.squeeze(1)
            out[d0:d0 + m] = torch.where(resolu & valide, val,
                                         torch.full_like(val, float("nan")))
            dj = (j + 1).to(torch.float64)
            dur[d0:d0 + m] = torch.where(resolu & valide, dj,
                                         torch.full_like(dj, float("nan")))
        sorties.append(out.cpu().numpy())
        temps.append(dur.cpu().numpy())
    if durees:
        return sorties[0], sorties[1], temps[0], temps[1]
    return sorties[0], sorties[1]


def verifie(df, cfg, n=1500, pas=97, borne=C.BORNE_DEFAUT, tol=1e-9) -> bool:
    """Les deux implementations decrivent-elles le MEME trade ?

    Sans cette verification les chiffres du GPU ne valent rien : deux
    descriptions du meme trade qui doivent s'accorder par convention finissent
    toujours par diverger, et c'est silencieux.
    """
    idx = np.arange(100, min(len(df) - borne - 3, 100 + n * pas), pas)
    a1, v1, da1, dv1 = C.rendements(df, idx, cfg, borne=borne, durees=True)
    a2, v2, da2, dv2 = rendements(df, idx, cfg, borne=borne, durees=True)
    ok = True
    for nom, x, y in (("achat", a1, a2), ("vente", v1, v2),
                      ("duree achat", da1, da2), ("duree vente", dv1, dv2)):
        fx, fy = np.isfinite(x), np.isfinite(y)
        if not np.array_equal(fx, fy):
            print(f"  {nom} : resolutions differentes "
                  f"({int(fx.sum())} contre {int(fy.sum())})")
            ok = False
            continue
        if fx.any():
            e = float(np.abs(x[fx] - y[fx]).max())
            print(f"  {nom} : {int(fx.sum())} resolus, ecart maximal {e:.2e}")
            ok = ok and e <= tol
    return ok


def main() -> int:
    import time
    import pandas as pd
    import training as T
    cfg = T.PPOConfig()
    df = pd.read_pickle("data_cache_XAUUSD_M5.pkl")
    df = df.iloc[:C.borne_etude(len(df))].reset_index(drop=True)
    print(f"verification sur {len(df):,} barres, "
          f"stop {cfg.atr_sl_mult:g}xATR, trailing "
          f"{cfg.atr_trail_mult/cfg.atr_sl_mult:.1f} R")
    ok = verifie(df, cfg)
    print("  ->", "IDENTIQUE" if ok else "DIVERGENT — ne rien lire du GPU")
    if not ok:
        return 1
    idx = np.arange(100, len(df) - C.BORNE_DEFAUT - 2, 6)
    t0 = time.perf_counter(); rendements(df, idx, cfg); tg = time.perf_counter() - t0
    t0 = time.perf_counter(); C.rendements(df, idx[:2000], cfg)
    tc = (time.perf_counter() - t0) * len(idx) / 2000
    print(f"\n{len(idx):,} entrees : GPU {tg:.1f} s, "
          f"numpy {tc:.0f} s estimees — {tc/max(tg,1e-9):.0f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
