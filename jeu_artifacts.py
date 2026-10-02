# -*- coding: utf-8 -*-
"""Pipeline fige d'un bloc : donnees, scalers, experts et predictions OOF.

Les predictions OOF sont necessaires aux rangs glissants au bord du test.
Reappliquer le seul expert final aux bougies du train ne les reproduit pas.
"""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd


SCHEMA_PIPELINE = 1


def normalise_config_historique(brut):
    """Les anciens manifestes gardent les regles avec lesquelles ils ont appris."""
    cfg = dict(brut)
    cfg.setdefault("expert_ratio_min", 0.0)
    cfg.setdefault("expert_mode", "moyenne")
    cfg.setdefault("jeu_version", 1)
    # Un checkpoint anterieur a M5_10_lot ne possede pas les poids de cette
    # tete. Le conserver tel quel est indispensable pour le live M5_09.
    cfg.setdefault("apprendre_lot", False)
    # Idem pour les checkpoints anterieurs a M5_11_risque : ils ne possedent
    # pas la sixieme tete et doivent rester chargeables sans ambiguite.
    cfg.setdefault("apprendre_risque", False)
    cfg.setdefault("observe_compte", False)
    return cfg


def empreinte_donnees(donnees):
    """Empreinte des colonnes, types, ordre et valeurs, independante de l'index."""
    entete = {"lignes": len(donnees), "colonnes": list(donnees.columns),
              "types": [str(t) for t in donnees.dtypes]}
    h = hashlib.sha256(json.dumps(entete, sort_keys=True).encode("utf-8"))
    # Une colonne a la fois pour ne pas dupliquer le cache complet en memoire.
    for col in donnees.columns:
        valeurs = pd.util.hash_pandas_object(donnees[col], index=False).to_numpy()
        h.update(valeurs.astype("<u8", copy=False).tobytes())
    return {**entete, "sha256": h.hexdigest()}


def empreinte_fichier(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloc in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(bloc)
    return h.hexdigest()


def _stats_json(stats, largeur):
    out = {}
    for nom in ("mean", "std"):
        v = np.asarray(stats[nom])
        if v.shape != (largeur,) or not np.isfinite(v).all():
            raise ValueError(f"Statistique {nom} incompatible avec {largeur} colonnes")
        if nom == "std" and (v < 0).any():
            raise ValueError("Ecart type negatif")
        out[nom] = v.tolist()
        out[nom + "_dtype"] = str(v.dtype)
    return out


def _stats_numpy(stats):
    return {nom: np.asarray(stats[nom], dtype=stats[nom + "_dtype"])
            for nom in ("mean", "std")}


def sauvegarde_pipeline_bloc(path, *, cfg, donnees, colonnes,
                              normalisation_marche, normalisation_expert,
                              modeles_expert, predictions_expert,
                              colonnes_expert=None, meta=None):
    """Ecrit un JSON et un NPZ immuables; refuse d'ecraser un bloc existant."""
    path = Path(path)
    sidecar = path.with_name(path.stem + "_predictions.npz")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"Pipeline deja present : {path}")
    config = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    colonnes = list(colonnes)
    if len(set(colonnes)) != len(colonnes) or not set(colonnes) <= set(donnees.columns):
        raise ValueError("Colonnes du modele absentes ou dupliquees")
    pred = np.asarray(predictions_expert)
    if pred.dtype != np.float32 or pred.shape != (len(donnees), 2):
        raise ValueError("Les predictions figees doivent etre float32 de forme (N, 2)")
    noms_expert = list(colonnes_expert or
                      ["expert_achat", "expert_vente", "expert_rang_achat", "expert_rang_vente"])
    if len(modeles_expert) != 2 or len(noms_expert) != 4:
        raise ValueError("Deux experts et quatre colonnes expert sont attendus")
    boosters = [getattr(m, "booster_", m).model_to_string() for m in modeles_expert]
    ratio = float(config.get("expert_ratio_min", 0.0))
    info = {
        "schema_version": SCHEMA_PIPELINE,
        "cible_version": "moyenne_toutes_barrieres_v1" if ratio <= 0 else "moyenne_ratio_min_v1",
        "config": config, "colonnes": colonnes, "colonnes_expert": noms_expert,
        "normalisation_marche": _stats_json(normalisation_marche, len(colonnes)),
        "normalisation_expert": _stats_json(normalisation_expert, len(noms_expert)),
        "normalisation_clip": 5.0, "normalisation_epsilon": 1e-8,
        "fenetre_expert": 10_000 // int(config["minutes_par_barre"]),
        "dataset": empreinte_donnees(donnees), "boosters": boosters,
        "predictions": {"fichier": sidecar.name, "forme": list(pred.shape)},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "meta": dict(meta or {}),
    }
    with sidecar.open("xb") as fh:
        np.savez_compressed(fh, predictions=pred)
    info["predictions"]["sha256"] = empreinte_fichier(sidecar)
    with path.open("x", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1, ensure_ascii=False, allow_nan=False)
    return info


def charge_pipeline_bloc(path, *, donnees, colonnes=None):
    """Valide le cache et le sidecar avant de charger les predictions figees."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        info = json.load(fh)
    if info.get("schema_version") != SCHEMA_PIPELINE:
        raise ValueError("Version de pipeline inconnue")
    if colonnes is not None and list(colonnes) != info["colonnes"]:
        raise ValueError("Ordre ou noms des colonnes du modele modifies")
    if empreinte_donnees(donnees) != info["dataset"]:
        raise ValueError("Le cache a change : empreinte du dataset differente du run")
    fichier = info["predictions"]["fichier"]
    if Path(fichier).name != fichier:
        raise ValueError("Chemin du sidecar invalide")
    sidecar = path.parent / fichier
    if empreinte_fichier(sidecar) != info["predictions"]["sha256"]:
        raise ValueError("Le fichier des predictions a ete modifie ou corrompu")
    with np.load(sidecar, allow_pickle=False) as z:
        pred = z["predictions"]
    if list(pred.shape) != info["predictions"]["forme"] or pred.dtype != np.float32:
        raise ValueError("Format des predictions incompatible avec le pipeline")
    info["_predictions"] = pred
    info["_normalisation_marche"] = _stats_numpy(info["normalisation_marche"])
    info["_normalisation_expert"] = _stats_numpy(info["normalisation_expert"])
    return info


def predit_pipeline_bloc(pipeline, donnees):
    """Reconstruit les observations exactes du dataset valide, sans fit."""
    # Import tardif : jeu_kairos utilise ce module lors de ses sauvegardes.
    import jeu_kairos as J

    X = donnees[pipeline["colonnes"]].to_numpy(np.float32)
    st = pipeline["_normalisation_marche"]
    eps, clip = pipeline["normalisation_epsilon"], pipeline["normalisation_clip"]
    Xn = np.clip((X - st["mean"]) / (st["std"] + eps), -clip, clip).astype(np.float32)
    pred = pipeline["_predictions"]
    FE = J.features_expert(pred, fenetre=pipeline["fenetre_expert"])
    se = pipeline["_normalisation_expert"]
    FEn = np.clip((FE - se["mean"]) / (se["std"] + eps), -clip, clip).astype(np.float32)
    largeur = int(pipeline["config"]["n_expert"])
    if not 0 <= largeur <= FEn.shape[1]:
        raise ValueError("Nombre de colonnes expert incompatible avec le pipeline")
    return np.concatenate([Xn, FEn[:, :largeur]], axis=1), FE[:, 2:4].copy(), pred


def charge_modeles_expert(pipeline):
    """Experts finaux exportables pour l'inference, sans reapprentissage."""
    import lightgbm as lgb
    return [lgb.Booster(model_str=s) for s in pipeline["boosters"]]


def ecrit_resultat_rejeu(prefixe, bloc, resultat, *, dossier=".", meta=None):
    """N'ecrit jamais dans test_<prefixe>_bloc<k>.json, meme au second replay."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(dossier) / f"rejeu_{Path(prefixe).name}_bloc{bloc}_{stamp}_{uuid.uuid4().hex[:8]}.json"
    with path.open("x", encoding="utf-8") as fh:
        json.dump({"bilan": resultat, "rejeu": dict(meta or {})}, fh, indent=1, default=str)
    return path
