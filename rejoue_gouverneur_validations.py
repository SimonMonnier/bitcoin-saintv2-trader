# -*- coding: utf-8 -*-
"""Mesure le gouverneur sur les validations deja effectuees, sans entrainement.

Les CSV des meilleurs checkpoints fournissent les memes entrees, sorties et
PnL. Ce script ne modifie que l'exposition appliquee a chaque coup et retire
les entrees bloquees apres une degradation de drawdown pendant la journee.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import jeu_kairos as J


def charge_validations(cfg: J.JeuConfig) -> pd.DataFrame:
    morceaux = []
    for bloc in range(1, int(cfg.n_blocs) + 1):
        ck_path = Path(f"best_{cfg.prefixe}_bloc{bloc}.pth")
        csv_path = Path(f"trades_{cfg.prefixe}_bloc{bloc}.csv")
        if not ck_path.is_file() or not csv_path.is_file():
            raise FileNotFoundError(f"Validation du fold {bloc} introuvable")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        epoch = int(ck.get("epoch", -999))
        d = pd.read_csv(csv_path)
        d = d[(d["phase"] == "validation") & (d["epoch"] == epoch)].copy()
        if d.empty:
            raise ValueError(f"Aucun trade de validation pour fold {bloc}, epoch {epoch}")
        d["fold"], d["checkpoint_epoch"] = bloc, epoch
        morceaux.append(d)
    trades = pd.concat(morceaux, ignore_index=True)
    trades["entree_time"] = pd.to_datetime(trades["entree_time"], utc=True, format="mixed")
    trades["sortie_time"] = pd.to_datetime(trades["sortie_time"], utc=True, format="mixed")
    return trades.sort_values(["entree_time", "sortie_time"], kind="stable").reset_index(drop=True)


def replay(trades: pd.DataFrame, cfg: J.JeuConfig) -> dict:
    """Rejoue strictement les trades archives avec le seul lot re-echelle."""
    e = float(cfg.capital)
    pic = e
    dd_debut_jour = 0.0
    jour = None
    bloques = set()
    ouverts = []  # (sortie, pnl gouverne)
    courbe = [e]
    pris = refuses = 0
    echelles = []

    def realise(jusqua):
        nonlocal e, pic
        restants = []
        for sortie, pnl in ouverts:
            if sortie <= jusqua:
                e = max(0.0, e + pnl)
                pic = max(pic, e)
                courbe.append(e)
            else:
                restants.append((sortie, pnl))
        ouverts[:] = restants

    for x in trades.itertuples(index=False):
        entree, sortie = x.entree_time, x.sortie_time
        realise(entree)
        jour_x = entree.date()
        if jour_x != jour:
            jour = jour_x
            dd_debut_jour = (pic - e) / max(pic, 1e-12)
        echelle, bloque, dd = J.gouverneur_exposition(e, pic, dd_debut_jour, cfg)
        if bloque:
            bloques.add(jour)
        if jour in bloques:
            refuses += 1
            continue
        # Le CSV conserve le PnL et l'equite au moment du trade. On en tire
        # le rendement du coup, puis on le rejoue sur l'equite courante :
        # c'est le comportement compose du jeu lorsque le plafond change.
        rendement = float(x.pnl_reel_usd) / max(float(x.equite_avant_usd), 1e-12)
        pnl = max(e * rendement * echelle, -e)
        ouverts.append((sortie, pnl))
        echelles.append(echelle)
        pris += 1
    realise(pd.Timestamp.max.tz_localize("UTC"))
    courbe = np.asarray(courbe, dtype=float)
    pics = np.maximum.accumulate(courbe)
    dd = (courbe - pics) / np.maximum(pics, 1e-12)
    return {
        "trades": int(pris), "refuses_gouverneur": int(refuses),
        "total_dollars": float(e - cfg.capital),
        "drawdown_dollars": float((courbe - pics).min()),
        "drawdown_pct": float(np.clip(dd.min(), -1.0, 0.0)),
        "exposition_moyenne": float(np.mean(echelles)) if echelles else 0.0,
        "exposition_min": float(np.min(echelles)) if echelles else 0.0,
    }


def main() -> int:
    base = J.JeuConfig()
    trades = charge_validations(base)
    grille = [
        ("sans_gouverneur", False, 0.05, 0.20, 1.00),
        ("frein_5_stop_20_plancher_25", True, 0.05, 0.20, 0.25),
        ("frein_5_stop_15_plancher_25", True, 0.05, 0.15, 0.25),
        ("frein_7_stop_20_plancher_35", True, 0.07, 0.20, 0.35),
        ("frein_10_stop_25_plancher_50", True, 0.10, 0.25, 0.50),
    ]
    resultats = []
    for nom, actif, frein, stop, plancher in grille:
        cfg = replace(base, gouverneur_exposition=actif, gouverneur_dd_frein=frein,
                      gouverneur_dd_stop=stop, gouverneur_exposition_min=plancher)
        par_fold = [replay(g, cfg) | {"fold": int(bloc)}
                    for bloc, g in trades.groupby("fold", sort=True)]
        r = {
            "trades": int(sum(x["trades"] for x in par_fold)),
            "refuses_gouverneur": int(sum(x["refuses_gouverneur"] for x in par_fold)),
            "total_dollars": float(sum(x["total_dollars"] for x in par_fold)),
            "drawdown_pct": float(min(x["drawdown_pct"] for x in par_fold)),
            "drawdown_moyen_pct": float(np.mean([x["drawdown_pct"] for x in par_fold])),
            "exposition_moyenne": float(np.mean([x["exposition_moyenne"] for x in par_fold])),
            "exposition_min": float(min(x["exposition_min"] for x in par_fold)),
            "par_fold": par_fold,
        } | {"scenario": nom, "frein_pct": 100 * frein,
             "stop_pct": 100 * stop, "plancher_pct": 100 * plancher}
        resultats.append(r)
        print(f"{nom:31s}  total {r['total_dollars']:+9.2f} $  "
              f"DD pire/moyen {100 * r['drawdown_pct']:+6.1f}/{100 * r['drawdown_moyen_pct']:+5.1f} %  "
              f"{r['trades']:5d} trades  "
              f"refuses {r['refuses_gouverneur']:4d}  expo moy {100 * r['exposition_moyenne']:5.1f} %",
              flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(f"replay_gouverneur_validations_{stamp}.json")
    with out.open("x", encoding="utf-8") as f:
        json.dump({"source": "validations_des_meilleurs_checkpoints", "trades_source": len(trades),
                   "resultats": resultats}, f, indent=2)
    pd.DataFrame([{k: v for k, v in x.items() if k != "par_fold"} for x in resultats]).to_csv(
        out.with_suffix(".csv"), index=False)
    print(f"\nReplay ecrit : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
