# -*- coding: utf-8 -*-
"""Rejoue un bloc depuis son pipeline fige, sans ecraser le test original.

Les anciens runs sans pipeline exigent --ancien-reapprentissage : cette
voie garde leur cible historique mais ne promet pas une reproduction exacte.
"""
import argparse
import dataclasses
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

import jeu_kairos as J
from jeu_artifacts import (charge_pipeline_bloc, ecrit_resultat_rejeu, empreinte_fichier,
                           normalise_config_historique, predit_pipeline_bloc)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def config_depuis_dict(brut) -> J.JeuConfig:
    brut = normalise_config_historique(brut)
    champs = {f.name for f in dataclasses.fields(J.JeuConfig)}
    return J.JeuConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                          for k, v in brut.items() if k in champs})


def config(prefixe: str) -> J.JeuConfig:
    with open(f"run_{prefixe}.json", encoding="utf-8") as fh:
        return config_depuis_dict(json.load(fh))


def verifie_lien_pipeline(etat, pipeline_path, bloc, version):
    if "bloc" in etat and int(etat["bloc"]) != bloc:
        raise ValueError("Le checkpoint appartient a un autre bloc")
    reference = etat.get("pipeline")
    if reference is None:
        if version >= 2:
            raise ValueError("Checkpoint recent sans reference de pipeline verifiable")
        return
    if (reference.get("fichier") != pipeline_path.name
            or reference.get("sha256") != empreinte_fichier(pipeline_path)):
        raise ValueError("Le pipeline ne correspond pas a celui du checkpoint")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefixe")
    parser.add_argument("bloc", type=int)
    parser.add_argument("--ancien-reapprentissage", action="store_true")
    args = parser.parse_args()
    prefixe, k = args.prefixe, args.bloc - 1
    src = f"best_{prefixe}_bloc{k + 1}.pth"
    etat = torch.load(src, map_location="cpu", weights_only=False)
    cfg = config_depuis_dict(etat["config"]) if "config" in etat else config(prefixe)
    if not 0 <= k < cfg.n_blocs:
        raise ValueError(f"Le bloc doit etre compris entre 1 et {cfg.n_blocs}")
    pipeline_path = Path(f"pipeline_{prefixe}_bloc{k + 1}.json")
    if not pipeline_path.exists() and (getattr(cfg, "jeu_version", 1) >= 2
                                      or not args.ancien_reapprentissage):
        raise FileNotFoundError(
            f"Pipeline manquant : {pipeline_path}. Un ancien run peut etre reconstruit "
            "explicitement avec --ancien-reapprentissage; un nouveau run exige son pipeline.")
    verifie_lien_pipeline(etat, pipeline_path, k + 1, getattr(cfg, "jeu_version", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = pd.read_pickle(cfg.cache)
    if getattr(cfg, "marche_seul", ""):
        d, cfg = J.donnees_marche_seul(d, cfg)
    cols = list(dict.fromkeys(["time", "open", "high", "low", "close", "atr_14",
                               "spread_bar"] + J.colonnes_jeu(cfg)))
    d = d[cols].reset_index(drop=True)
    pipeline = None
    if pipeline_path.exists():
        pipeline = charge_pipeline_bloc(pipeline_path, donnees=d, colonnes=J.colonnes_jeu(cfg))
        pcfg = config_depuis_dict(pipeline["config"])
        if dataclasses.asdict(pcfg) != dataclasses.asdict(cfg):
            raise ValueError("La configuration du pipeline differe du checkpoint")
    N = len(d)
    o, h, l, c = (d[x].to_numpy(np.float64) for x in ("open", "high", "low", "close"))
    atr = J.atr_effectif(d["atr_14"].to_numpy(np.float64), c, cfg)
    sp = d["spread_bar"].to_numpy(np.float64)
    if getattr(cfg, "marche_seul", ""):
        cout_med = float(np.median(sp)) + cfg.glissement_entree_bps + cfg.glissement_sortie_bps
        atr = np.maximum(atr, cfg.plancher_couts * cout_med * 1e-4 * c)
    t_ns = d["time"].values.astype("int64")
    X = d[J.colonnes_jeu(cfg)].to_numpy(np.float32)
    R1, D1, S1 = J.table_coups(o, h, l, sp, atr, cfg, 1.0)
    toutes = J.parties(d["time"], 0, N, cfg)
    purge = J.purge_barres(cfg)
    permis, (te0, te1), _ = J.masque_blocs(N, cfg.n_blocs, k, purge)
    fin_tr = J.prochain_exclu(permis)
    j_te = np.array([w for w in toutes if w[0] >= te0 and w[1] <= te1], np.int64).reshape(-1, 2)
    if pipeline is not None:
        Xk, rangs, _ = predit_pipeline_bloc(pipeline, d)
        mode_rejeu = "pipeline fige, aucun reapprentissage"
    else:
        print("ANCIEN RUN : expert reconstruit avec sa cible historique ; "
              "la reproduction exacte depend aussi des versions et du cache.", flush=True)
        y_ex = J.cibles_expert(R1, cfg)
        st_m = X[permis].astype(np.float64).mean(0)
        st_s = X[permis].astype(np.float64).std(0)
        Xn = J.safe_normalize(X, {"mean": st_m.astype(np.float32),
                                  "std": st_s.astype(np.float32)}).astype(np.float32)
        pred, _ = J.expert_multi(Xn, y_ex, t_ns, permis, fin_tr, cfg)
        FE = J.features_expert(pred, fenetre=10_000 // int(cfg.minutes_par_barre))
        rangs = FE[:, 2:4].copy()
        m_e = FE[permis].astype(np.float64).mean(0)
        s_e = FE[permis].astype(np.float64).std(0)
        FEn = np.clip((FE - m_e) / (s_e + 1e-8), -5.0, 5.0).astype(np.float32)
        Xk = np.concatenate([Xn, FEn[:, :cfg.n_expert]], axis=1)
        mode_rejeu = "ancien expert reconstruit"
    policy = J.PolitiqueJeu(cfg).to(device)
    policy.load_state_dict(etat["modele"])
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.graine)
    s_t, c_t, _ = J.joue(policy, j_te, Xk, R1, D1, S1, te1, cfg, device, explore=False,
                         gen=gen, rangs=rangs, marge=J.fraction_marge(c, atr, cfg),
                         prix=c, atr=atr)
    bt = J.bilan(s_t, c_t, c, atr, sp, cfg, open_=o, temps=d["time"])
    fmt = lambda i: pd.Timestamp(d["time"].iloc[min(i, N - 1)]).strftime("%Y-%m-%d")
    print(f"{prefixe} bloc {k + 1} : test {fmt(te0)} -> {fmt(te1 - 1)} ({len(j_te)} parties), "
          f"modele {src} (epoch {etat.get('epoch')})")
    print(f"TEST fold {k + 1} (rejoue)  {J.ligne_bilan(bt, cfg)}")
    print(f"TEST fold {k + 1} bilan  {J.ligne_detail(bt, cfg)}")
    sortie = ecrit_resultat_rejeu(prefixe, k + 1, bt,
                                  meta={"checkpoint": src, "epoch": etat.get("epoch"),
                                        "mode": mode_rejeu, "pipeline": str(pipeline_path)})
    print(f"Resultat de rejeu : {sortie} ; test original conserve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
