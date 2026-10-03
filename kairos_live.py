# -*- coding: utf-8 -*-
"""KAIROS M5 en direct : la configuration, l'agent de l'interface, la commande.

    python kairos_live.py            interface graphique + agent (kairos_gui.py --start)
    python kairos_live.py --check    verifie que le modele final est pret, sans MT5
    python kairos_live.py --console  agent seul, sans interface

LE MOTEUR est `kairos_m5_live.py` : il lit le modele final du run (le
« deploy », eleve des dix meilleurs folds) et son pipeline fige (experts
LightGBM et normalisations), reconstruit les memes features que le jeu a
partir de Binance, Coinbase et du financement, et passe ses ordres dans MT5.
IL REFUSE TOUT COMPTE QUI N'EST PAS UN COMPTE DEMO.

L'ancien moteur M1 (XAUUSD, multi-agents wf1/wf2/wf3) a ete retire le
2026-10-02 avec le nettoyage du depot : il reste dans l'historique git.
"""
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch


@dataclass
class LiveConfig:
    engine: str = "kairos_m5"
    # Le run dont on trade le modele final : deploy_<prefixe>_ensemble_validation_ddsafe.pth
    # et pipeline_<prefixe>_deploy_ensemble_validation_ddsafe.json, ecrits par
    # `jeu_kairos.py` a la fin des dix blocs (ou `--rebuild-deploy-ensemble`).
    kairos_prefixe: str = "kairos_jeu_m5_24_paires"
    kairos_bloc: int = 1
    # Fuseau de l'horloge des graphiques Vantage/MT5 (EEST l'ete, EET l'hiver).
    kairos_tz_mt5: str = "Europe/Helsinki"
    symbol: str = "BTCUSD"
    side: str = "both"


class TradingAgent:
    """L'agent que pilote l'interface : un fil qui fait tourner le moteur M5."""

    def __init__(self, cfg: LiveConfig):
        self.cfg = cfg
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _should_continue(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            print("[AGENT] Deja en cours d'execution.")
            return
        print("[AGENT] Demarrage du bot...")
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            from kairos_m5_live import live_loop as kairos_m5_loop
            print("[AGENT] KAIROS M5 : moteur aligne au jeu")
            kairos_m5_loop(self.cfg, self._should_continue)
        except Exception as e:
            print(f"[AGENT] Erreur dans live_loop : {e}")
        finally:
            self._running = False
            print("[AGENT] live_loop termine.")

    def stop(self):
        if not self._running:
            print("[AGENT] Bot deja arrete.")
            return
        print("[AGENT] Arret demande...")
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        print("[AGENT] Bot arrete.")


def verifie_deploy_kairos(prefixe: str) -> tuple:
    """Refuse tout demarrage sans le modele final complet et son pipeline."""
    checkpoint = f"deploy_{prefixe}_ensemble_validation_ddsafe.pth"
    pipeline = f"pipeline_{prefixe}_deploy_ensemble_validation_ddsafe.json"
    manquants = [p for p in (checkpoint, pipeline) if not os.path.isfile(p)]
    if manquants:
        raise FileNotFoundError(
            "Modele deploy KAIROS pas encore pret : " + ", ".join(manquants) +
            ". Il est ecrit a la fin des dix blocs du run, ou par "
            "`python jeu_kairos.py --rebuild-deploy-ensemble`.")
    # Lecture minimale avant MT5 : un fichier incomplet ou une mauvaise paire
    # est refuse avant toute connexion au compte demo.
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "modele" not in ck or "config" not in ck:
        raise ValueError(f"Checkpoint deploy invalide : {checkpoint}")
    if ck.get("type") != "deploiement_ensemble_10_folds" or len(ck.get("folds", [])) != 10:
        raise ValueError(f"Checkpoint deploy incomplet ou non-ensemble : {checkpoint}")
    with open(pipeline, encoding="utf-8") as fh:
        pipe = json.load(fh)
    if pipe.get("meta", {}).get("type") != "deploiement_ensemble_10_folds":
        raise ValueError(f"Pipeline deploy invalide ou non final : {pipeline}")
    return checkpoint, pipeline


def main_kairos_m5() -> int:
    """`python kairos_live.py` : interface + moteur M5 BTCUSD."""
    cfg = LiveConfig()
    try:
        checkpoint, pipeline = verifie_deploy_kairos(cfg.kairos_prefixe)
    except Exception as e:
        print(f"[KAIROS M5] Demarrage refuse : {e}", file=sys.stderr)
        return 2
    if "--check" in sys.argv:
        print(f"[KAIROS M5] DEPLOY pret : {checkpoint} | {pipeline}")
        return 0
    if "--console" not in sys.argv:
        gui = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kairos_gui.py")
        print("[KAIROS M5] Ouverture du live avec interface graphique.")
        return subprocess.call([sys.executable, gui, "--start"])
    print("[KAIROS M5] Demarrage console du moteur aligne au jeu.")
    print(f"[KAIROS M5] modele : {checkpoint}")
    print("[KAIROS M5] Ctrl+C pour arreter proprement.")
    from kairos_m5_live import live_loop as kairos_m5_loop
    try:
        kairos_m5_loop(cfg, lambda: True)
    except KeyboardInterrupt:
        print("\n[KAIROS M5] Arret demande par l'utilisateur.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_kairos_m5())
