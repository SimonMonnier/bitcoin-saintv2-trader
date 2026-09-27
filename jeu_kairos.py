# -*- coding: utf-8 -*-
"""KAIROS EN JEU — le scalping BTC pose comme une partie, plus comme un flux de trades.

POURQUOI CE FICHIER EXISTE — 2026-09-26, demande du proprietaire : « faire
le PPO comme pour gagner a un jeu », revoir les actions possibles, avec un
expert a imiter d'abord.

LE BON JEU N'EST PAS TAXI, C'EST LE BLACKJACK. Dans Taxi chaque action a
une consequence certaine ; ici chaque coup est en grande partie du hasard,
et l'on perd un peu a jouer toujours — l'avantage de la maison est le
spread. On ne gagne qu'en misant quand la situation est favorable. C'est
ce que la mesure a montre (`mesure_sources_btc.py`) : l'avantage n'existe
que sur les signaux les plus forts, a 15 minutes et plus.

LES REGLES
  partie       une journee de BTC (heure du serveur)
  coup         UN geste : un sens, un objectif, un stop. Le marche le
               resout seul — objectif touche, stop touche, ou temps ecoule
               (`horizon_max`). Plus de decision « tenir ou fermer » a
               chaque minute : c'est elle qui fermait tout a la premiere
               minute.
  jetons       `jetons` coups par partie. Les garder pour les meilleures
               occasions fait partie du jeu.
  vie          la partie s'arrete si le score tombe a -`vie_R` R.
  score        la somme des R nets de la partie. Un R = la perte au stop,
               soit `risque_pct` du capital : chaque coup risque la meme
               somme.

LES QUATRE TETES, CHACUNE AVEC SON OPTIMISEUR — choix du proprietaire :
  achat   (`tete_achat`)      logit d'acheter contre attendre
  vente   (`tete_vente`)      logit de vendre contre attendre
  gain    (`tete_objectif`)   coupure des gains : ou poser l'objectif
  perte   (`tete_stop`)       coupure des pertes : ou poser le stop
Le tronc SAINT, partage, et le critique ont chacun le leur.

L'EXPERT D'ABORD. Avant de jouer, le modele imite un expert qui connait
l'avenir : chaque jour, les meilleurs coups a posteriori, sans
chevauchement, au plus `expert_k`, chacun rapportant au moins
`expert_R_min` R net. C'est ce qu'a fait AlphaGo avec des parties humaines
avant de jouer contre lui-meme. Puis PPO joue, et le jeu seul juge.

L'EXPERT EST REALISTE DEPUIS LE RUN kairos_jeu_btc03. Celui qui lisait
l'avenir enseignait des coups qu'aucune observation ne permet de
reconnaitre. Il est remplace par un LightGBM appris sur le PASSE, qui
predit le R net moyen d'un achat et d'une vente a chaque minute ; sur le
train, chaque bloc est predit par un modele qui ne l'a pas appris
(validation croisee par blocs purges). `mesure_sources_btc.py` a mesure
qu'un tel modele capte +3.3 bps par trade sur ses 2 % de signaux les plus
forts, hors echantillon. L'expert joue d'abord SEUL la validation : c'est
la reference que le PPO doit battre.

LE SCORE DE L'EXPERT EST UNE ENTREE DU MODELE DEPUIS LE RUN
kairos_jeu_btc04. `mesure_expert_val.py` : joue seul sur ses signaux les
plus forts (rang glissant, sommet 0.5 a 2 %), l'expert gagne en
validation sur 11 a 12 des 16 barrieres, jusqu'a +4.63 $/jour (PF 1.58).
Le PPO, lui, glissait vers « ne jamais trader » : il devait reapprendre
seul, depuis un signal a +0.04 de correlation, ce que l'expert avait deja
trouve. Il recoit donc ses deux predictions et leur rang glissant
(`features_expert`) et garde toutes les decisions : quand, quel sens,
quel objectif, quel stop. AUCUNE FUITE : l'expert n'apprend que sur le
train du fold 1 ; chaque bloc du train est note par un modele qui ne l'a
pas appris ; tout le reste par le modele du train entier, comme en direct.
Les deux modeles de l'expert sont sauvegardes pour le live.

L'EXPERT N'ENSEIGNE QUE L'ENTREE — quand, et dans quel sens. Premier
lancement, 2026-09-26 : imites, l'objectif et le stop etaient TOUJOURS
8 ATR et 2 ATR — le coup qui rapporte le plus de R quand on connait
l'avenir. Joue sans le connaitre, ce stop etait touche 80 % du temps, et
les deux tetes etaient figees (entropie 0.010 sur 1.099) : PPO ne pouvait
plus rien leur apprendre. Les barrieres partent donc d'un choix uniforme,
et c'est le jeu seul qui les regle.

CE QUI NE CHANGE PAS : le modele SAINT, les 71 features du BTC, la rampe
de cout (niveau 1 sans spread), l'action la plus probable en validation et
au test, la fenetre de test jamais lue pour choisir quoi que ce soit.

CE QUE LE COURTIER FERA EN LIVE, et c'est pour ca que le jeu se deploie :
un ordre au marche avec son stop et son objectif attaches. C'est le
serveur qui ferme, meme si le bot s'arrete.

    python jeu_kairos.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from saint_core import (ACHETER, ATTENDRE, FEATURE_COLS, N_ACTIONS,
                        OBS_N_FEATURES, VENDRE, SAINTPolicySingleHead,
                        safe_normalize)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

N_ETAT = 5          # le bloc de position de l'observation porte l'etat du jeu


@dataclass
class JeuConfig:
    prefixe: str = "kairos_multi_m15_02"
    # LE JEU EN BOUGIES DE 15 MINUTES — 2026-09-27, demande du proprietaire :
    # « recommence le jeu avec des bougies de 15 minutes, et pas M1 pour
    # entrer ». Le modele voit des bougies M15 (contextes H1 et H4, voir
    # `prepare_btc_m15`) et ne decide qu'a leur cloture. Une partie reste
    # une journee : 96 decisions au lieu de 1440.
    #
    # CE QUE CELA CHANGE AU COUT : l'ATR M15 median vaut 26 bps, l'ATR M1
    # environ 6, pour le meme cout de ~3.3 bps par coup. Rapporte au
    # mouvement d'une bougie, il pese quatre fois moins.
    cache: str = "data_cache_MULTI_M15.pkl"
    minutes_par_barre: int = 15
    # LE JEU MULTI-MARCHES — 2026-09-27, demande du proprietaire : un seul
    # jeu qui trade le BTC, l'ETH, l'or et les indices a petit spread et
    # bonne volatilite, pour maximiser le nombre de trades et le profit en
    # gardant un drawdown bas. Voir `prepare_multi_m15` et `main_multi`.
    #
    # CHAQUE MARCHE JOUE SA JOURNEE, avec ses jetons, sa vie et la porte de
    # l'expert ; le resultat du JOUR est la somme des marches, et le
    # drawdown se mesure sur le compte commun, coups dans l'ordre du temps.
    # Vide = le jeu d'un seul marche (`cache`).
    marches: Tuple[str, ...] = ("BTCUSD", "ETHUSD", "XAUUSD", "NAS100", "SP500",
                                "DJ30", "GER40")
    # LE PLANCHER DES BARRIERES, PAR MARCHE : cinq fois son cout median
    # (spread + glissements), jamais sous 10 bps. Le cout ne depasse donc
    # jamais ~0.2 R, sur le BTC (17 bps) comme sur l'ETH (58) ou le Dow (10).
    plancher_couts: float = 5.0
    plancher_min_bps: float = 10.0
    # --- le modele : celui du run PPO, a l'identique ---
    lookback: int = 4
    d_model: int = 8
    num_blocks: int = 2
    heads: int = 1
    n_freq: int = 2
    mlp_dim: int = 32
    # --- les regles ---
    # 6 -> 3 LE 2026-09-26 (run kairos_jeu_btc03). La politique jouait tous
    # ses jetons, ~5.5 coups par jour, donc aussi des signaux moyens. La
    # mesure dit que l'avantage ne vit que dans les plus forts : +1.2 bps
    # sur les 10 % du sommet, +3.3 sur les 2 %.
    # 3 -> 10 LE 2026-09-27, demande du proprietaire : davantage de trades
    # par jour. En M15, le run m15_01 jouait ~2 coups par jour avec 3
    # jetons, rentable en validation. SEUL CE REGLAGE CHANGE dans m15_03 :
    # m15_02 avait aussi elargi la porte et l'expert, sans demande.
    jetons: int = 10
    vie_R: float = 3.0
    # En ATR de la barre de decision. L'ATR M1 du BTC vaut ~7 bps, un
    # mouvement de 15 minutes ~2.5 ATR : les coups vont de la demi-heure a
    # quelques heures.
    # 16 ATR ET 240 MINUTES AJOUTES LE 2026-09-26 (run kairos_jeu_btc03) :
    # le cout se paie une fois par coup, le mouvement capte grandit avec la
    # duree, et plusieurs colonnes portent davantage a 60 minutes qu'a 15.
    #
    # EN M15 LE MENU EST EN ATR M15, quatre fois plus grand : 1 a 8 ATR,
    # soit ~26 a ~210 bps en mediane — le meme ordre de grandeur que 4 a 32
    # ATR M1.
    tp_atr: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    sl_atr: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    # LE PLANCHER DE VOLATILITE DES BARRIERES — 2026-09-26, run
    # kairos_jeu_btc01. L'ATR M1 du BTC tombe a 2 bps dans les 10 % de
    # minutes les plus calmes, pour 3.5 bps de cout par coup. Un stop de
    # 2 ATR y faisait 4 bps : le cout mangeait 0.81 R de chaque coup AVANT
    # que le marche bouge (0.29 R en mediane). Chaque coup risquant la meme
    # somme, ces coups-la faisaient perdre les journees alors que le net
    # par coup, en bps, etait deja proche de zero.
    #
    # L'ATR qui place l'objectif et le stop ne descend donc jamais sous
    # `atr_min_bps` : a 8 bps, le stop fait au moins 16 bps et le cout ne
    # depasse jamais ~0.2 R. En marche agite, rien ne change ; en marche
    # calme, le coup est plus petit au lieu d'etre mange par le spread.
    # En M15 : l'ATR ne descend pas sous 20 bps (p10 : 13), le plus petit
    # stop fait 20 bps et le cout ne depasse jamais ~0.17 R.
    atr_min_bps: float = 20.0
    # EN BOUGIES : 32 bougies M15, huit heures.
    horizon_max: int = 32
    # Glissements ESPERES (la moitie des bornes de `training.PPOConfig`) :
    # entree toujours, sortie au stop et au temps, jamais a l'objectif.
    glissement_entree_bps: float = 0.5
    glissement_sortie_bps: float = 1.0
    capital: float = 1000.0
    # 1 % -> 0.5 % LE 2026-09-27 (run m15_05), demande du proprietaire :
    # reduire le drawdown. Le jeu apprend en R, donc les decisions ne
    # changent pas ; les dollars et le drawdown sont divises par deux, le
    # rapport gain / drawdown est inchange. C'est la mise du live.
    risque_pct: float = 0.5
    # --- le walk-forward : les memes proportions que le run PPO ---
    part_train: float = 0.55
    part_val: float = 0.15
    part_test: float = 0.10
    n_folds: int = 3
    epochs: int = 40
    rampe_cout: int = 10
    parties_par_epoch: int = 256
    # --- l'expert ---
    expert_k: int = 4
    expert_R_min: float = 1.0          # l'ancien expert, qui lisait l'avenir
    # EN M15, une bougie sur deux : il n'y en a que 96 par jour.
    expert_pas_neg: int = 2
    expert_poids_pos: float = 5.0
    expert_epochs: int = 3
    # L'EXPERT REALISTE : LightGBM, validation croisee par blocs purges.
    expert_blocs: int = 4
    # EN M15, toutes les bougies : 34 000 lignes d'apprentissage.
    expert_pas_app: int = 1
    expert_arbres: int = 300
    # Les colonnes de l'expert ajoutees a l'observation. Voir `features_expert`.
    n_expert: int = 4
    # LA PORTE D'ENTREE — 2026-09-26, run kairos_jeu_btc05.
    #
    # `analyse_jeu_val.py`, validation du fold 1, trade par trade, sur les
    # deux modeles du run btc04 (epochs 39 et 40) :
    #
    #     rang de l'expert (sens joue)   epoch 39             epoch 40
    #     les 2 % du sommet              +0.092 R  PF 1.34    +0.116 R  PF 1.45
    #     de 2 % a 10 %                  +0.012 R  PF 1.04    +0.015 R  PF 1.05
    #     au-dela de 10 %                -0.034 R  PF 0.88    -0.167 R  PF 0.54
    #
    # Pres de la moitie des coups se prenaient au-dela des 10 %, et c'etaient
    # eux qui perdaient. Un achat n'est donc PERMIS que si le rang de
    # l'expert a l'achat est au moins `porte_rang_expert`, une vente de meme.
    # A l'interieur de la porte, le PPO et les quatre tetes decident de tout :
    # entrer ou non, le sens, l'objectif, le stop. 0 = pas de porte.
    #
    # LE RUN m15_02 L'AVAIT ELARGIE A 0.80 SANS QUE LE PROPRIETAIRE L'AIT
    # DEMANDE — seul le nombre de jetons devait changer. Remise a 0.90 le
    # 2026-09-27 (run m15_03).
    porte_rang_expert: float = 0.90
    # --- PPO ---
    lr: float = 5e-4
    gamma: float = 0.9995          # par minute
    lam: float = 0.95
    clip: float = 0.2
    ppo_epochs: int = 4
    minibatch: int = 2048
    max_transitions: int = 65_536
    entropie_entree: float = 0.01
    entropie_barrieres: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.6
    # --- la sauvegarde ---
    min_coups_val: int = 30
    # --- LE DRAWDOWN — 2026-09-27, run kairos_jeu_m15_04, demande du
    # proprietaire : « tres bons resultats, a part le drawdown ». Le run m15_03
    # (10 jetons) gagnait +0.3 a +3.6 $/jour en validation, pour un drawdown
    # de -7 a -15 % : a 1 % de risque par coup et un win rate de 31 a 45 %,
    # dix pertes d'affilee ne sont pas rares.
    #
    # 1. LES PERTES PESENT PLUS DANS LA RECOMPENSE DU PPO, a l'entrainement
    #    seulement : un coup perdant compte `poids_pertes` fois. La politique
    #    prefere alors les coups qui enchainent moins de pertes. Le score
    #    affiche, la vie et la sauvegarde restent en vrais R.
    #
    # REMIS A 1.0 LE 2026-09-27 (run m15_05). Mesure du run m15_04 contre
    # m15_03, memes epochs 1 a 12 de validation : le drawdown etait divise
    # par deux (-4 a -8 % contre -6 a -15 %), mais le gain par trois
    # (~+120 $ contre ~+350 $ sur 97 jours) ; le rapport gain / drawdown
    # tombait de ~3.3 a ~2.3. Le modele devenait prudent — objectifs plus
    # proches, win rate jusqu'a 64 % — et chaque coup rapportait moins.
    # Reduire la mise (`risque_pct`) divise le drawdown sans toucher ce
    # rapport : c'est ce levier qui est garde.
    poids_pertes: float = 1.0
    # 2. LA MISE REDUITE EN BAISSE — la regle des gerants : quand le compte
    #    est a plus de `seuil_baisse` sous son plus haut, chaque coup ne
    #    risque plus que `mise_en_baisse` fois la mise ; elle revient
    #    entiere au nouveau plus haut. C'est une regle de GESTION, pas de
    #    decision : elle ne change pas quels coups sont bons, seulement
    #    combien on y met. Elle se joue jour apres jour dans l'ordre, en
    #    validation, au test et en live ; le bilan la montre a cote de la
    #    mise fixe, avec le rapport gain / drawdown des deux.
    seuil_baisse: float = 0.05
    mise_en_baisse: float = 0.5
    graine: int = 7

    @property
    def risque_dollars(self) -> float:
        return self.capital * self.risque_pct / 100.0

    @property
    def barres_par_jour(self) -> int:
        return 1440 // int(self.minutes_par_barre)


def colonnes_jeu(cfg: "JeuConfig") -> list:
    """Les features du cache que joue ce jeu : multi-marches, M15 ou M1."""
    if tuple(getattr(cfg, "marches", ())):
        from prepare_multi_m15 import FEATURE_COLS_MULTI
        return list(FEATURE_COLS_MULTI)
    if int(cfg.minutes_par_barre) == 15:
        from prepare_btc_m15 import FEATURE_COLS_M15
        return list(FEATURE_COLS_M15)
    if int(cfg.minutes_par_barre) == 1:
        return list(FEATURE_COLS)
    raise ValueError(f"pas de jeu de features pour {cfg.minutes_par_barre} min")


# ======================================================================
# LE MODELE — le SAINT du run, deux tetes de barrieres en plus
# ======================================================================
class PolitiqueJeu(SAINTPolicySingleHead):
    """Le tronc et les tetes d'achat et de vente du run PPO ; les tetes de
    coupure des gains et des pertes choisissent desormais l'objectif et le
    stop, AU MOMENT DU COUP, en lisant le tronc et le sens."""

    def __init__(self, cfg: JeuConfig):
        super().__init__(
            n_features=len(colonnes_jeu(cfg)) + cfg.n_expert + N_ETAT,
            d_model=cfg.d_model,
            num_blocks=cfg.num_blocks, heads=cfg.heads, n_freq=cfg.n_freq,
            mlp_dim=cfg.mlp_dim, lecture="colonnes", dropout=0.05, ff_mult=2,
            max_len=cfg.lookback, n_actions=N_ACTIONS, n_ref=0)
        dl = self.dim_lecture

        def _lecteur():
            return nn.Sequential(nn.Linear(dl + 1, cfg.mlp_dim), nn.GELU(),
                                 nn.Linear(cfg.mlp_dim, 64), nn.GELU())
        self.lecteur_objectif = _lecteur()
        self.tete_objectif = nn.Linear(64, len(cfg.tp_atr))
        self.lecteur_stop = _lecteur()
        self.tete_stop = nn.Linear(64, len(cfg.sl_atr))
        for m in list(self.lecteur_objectif) + list(self.lecteur_stop):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        for m in (self.tete_objectif, self.tete_stop):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)

    def jeu(self, x: torch.Tensor):
        """(logits d'entree (B,3), valeur (B,), objectif (B,2,K), stop (B,2,K)).

        L'axe 1 des barrieres est le sens : 0 achat, 1 vente. Les deux sont
        calcules — les tetes sont petites — pour que le sens tire ensuite
        choisisse les siennes sans second passage dans le tronc.
        """
        zn = self._lecture_tronc(x)
        la = self.tete_achat(self.mlp_achat(zn))
        lv = self.tete_vente(self.mlp_vente(zn))
        le = torch.cat([la, lv, torch.zeros_like(la)], dim=-1)
        v = self.critic(self.mlp(zn)).squeeze(-1)
        un = torch.ones_like(la)
        zl, zs = torch.cat([zn, un], -1), torch.cat([zn, -un], -1)
        ltp = torch.stack([self.tete_objectif(self.lecteur_objectif(zl)),
                           self.tete_objectif(self.lecteur_objectif(zs))], 1)
        lsl = torch.stack([self.tete_stop(self.lecteur_stop(zl)),
                           self.tete_stop(self.lecteur_stop(zs))], 1)
        return le, v, ltp, lsl

    def groupes_jeu(self) -> Dict[str, list]:
        """Six groupes DISJOINTS, un optimiseur chacun."""
        def _p(*mods):
            return [q for m in mods for q in m.parameters()]
        g = {
            "achat": _p(self.mlp_achat, self.tete_achat),
            "vente": _p(self.mlp_vente, self.tete_vente),
            "gain": _p(self.lecteur_objectif, self.tete_objectif),
            "perte": _p(self.lecteur_stop, self.tete_stop),
            "valeur": _p(self.mlp, self.critic),
            "tronc": _p(self.embed, self.col_emb, *self.blocks, self.norm)
                     + [self.cls],
        }
        if self.memoire is not None:
            g["tronc"] += list(self.memoire.parameters())
        return g


def optimiseurs(policy: PolitiqueJeu, cfg: JeuConfig):
    return {k: torch.optim.Adam(v, lr=cfg.lr, eps=1e-8)
            for k, v in policy.groupes_jeu().items()}


def _pas(policy, optims, perte, cfg):
    for o in optims.values():
        o.zero_grad(set_to_none=True)
    perte.backward()
    for nom, ps in policy.groupes_jeu().items():
        torch.nn.utils.clip_grad_norm_(ps, cfg.max_grad_norm)
        optims[nom].step()


# ======================================================================
# LES COUPS — chaque coup possible, a chaque minute, resolu d'avance
# ======================================================================
def atr_effectif(atr, prix, cfg: JeuConfig) -> np.ndarray:
    """L'ATR qui place les barrieres : jamais sous `atr_min_bps`."""
    return np.maximum(np.asarray(atr, np.float64),
                      float(cfg.atr_min_bps) * 1e-4 * np.asarray(prix, np.float64))


def table_coups(o, h, l, sp, atr, cfg: JeuConfig, frac: float):
    """Le resultat de CHAQUE coup possible a CHAQUE minute.

    Rend (R, D, S), de forme (N, 2, K_tp, K_sl) :
      R  le resultat net en R (NaN si le coup lirait au-dela des donnees)
      D  la duree : la decision suivante tombe a t + D
      S  la sortie : 0 objectif, 1 stop, 2 temps

    LA DECISION EST PRISE A LA CLOTURE DE t, l'entree se fait a
    l'OUVERTURE de t+1 — la convention de l'environnement PPO. Les barres
    sont des BID :
      achat   entre a l'ask (bid + spread de la barre t), sort au bid ;
              objectif et stop se lisent sur le bid.
      vente   entre au bid, sort a l'ask ; objectif et stop se lisent sur
              l'ask (bid + spread de chaque barre) — c'est ainsi que MT5
              declenche les ordres d'une position courte.
    UN STOP ET UN OBJECTIF TOUCHES DANS LA MEME BARRE : le stop d'abord.
    On ne sait pas lequel est venu en premier ; on suppose le pire.
    UN STOP SAUTE PAR UN TROU : sortie a l'ouverture, pas au niveau.

    `frac` est la part du cout — spread et glissements — de la rampe.
    """
    N = len(o)
    H = int(cfg.horizon_max)
    ktp = np.asarray(cfg.tp_atr, np.float64)
    ksl = np.asarray(cfg.sl_atr, np.float64)
    se = frac * cfg.glissement_entree_bps / 1e4
    sx = frac * cfg.glissement_sortie_bps / 1e4
    spf = frac * np.asarray(sp, np.float64) / 1e4
    shape = (N, 2, len(ktp), len(ksl))
    R = np.full(shape, np.nan, np.float32)
    D = np.full(shape, H + 1, np.int16)
    S = np.full(shape, 2, np.int8)
    t = np.arange(max(N - 1 - H, 0))
    if len(t) == 0:
        return R, D, S
    e = t + 1
    a = np.asarray(atr, np.float64)[t]
    oa, ha, la_ = o * (1 + spf), h * (1 + spf), l * (1 + spf)
    for si, sens in enumerate((1.0, -1.0)):
        if sens > 0:
            p0 = o[e] * (1 + spf[t] + se)
            tp = p0[:, None] + ktp[None, :] * a[:, None]
            sl = p0[:, None] - ksl[None, :] * a[:, None]
            hi, lo = h, l
        else:
            p0 = o[e] * (1 - se)
            tp = p0[:, None] - ktp[None, :] * a[:, None]
            sl = p0[:, None] + ksl[None, :] * a[:, None]
            hi, lo = ha, la_
        t_tp = np.full((len(t), len(ktp)), H, np.int16)
        t_sl = np.full((len(t), len(ksl)), H, np.int16)
        for k in range(H):
            hk = hi[e + k][:, None]
            lk = lo[e + k][:, None]
            if sens > 0:
                m_tp = (t_tp == H) & (hk >= tp)
                m_sl = (t_sl == H) & (lk <= sl)
            else:
                m_tp = (t_tp == H) & (lk <= tp)
                m_sl = (t_sl == H) & (hk >= sl)
            t_tp[m_tp] = k
            t_sl[m_sl] = k
        for i in range(len(ktp)):
            for j in range(len(ksl)):
                a_tp, a_sl = t_tp[:, i], t_sl[:, j]
                stop = (a_sl <= a_tp) & (a_sl < H)
                obj = (a_tp < a_sl) & (a_tp < H)
                barre = e + np.minimum(a_sl, H - 1)
                if sens > 0:
                    x_sl = np.minimum(sl[:, j], o[barre]) * (1 - sx)
                    x_to = o[e + H] * (1 - sx)
                else:
                    x_sl = np.maximum(sl[:, j], oa[barre]) * (1 + sx)
                    x_to = oa[e + H] * (1 + sx)
                x = np.where(stop, x_sl, np.where(obj, tp[:, i], x_to))
                R[t, si, i, j] = (sens * (x - p0) / (ksl[j] * a)).astype(np.float32)
                D[t, si, i, j] = np.where(stop, a_sl + 1,
                                          np.where(obj, a_tp + 1, H + 1))
                S[t, si, i, j] = np.where(obj, 0, np.where(stop, 1, 2))
    return R, D, S


# ======================================================================
# LES PARTIES
# ======================================================================
def journees(temps: pd.Series, debut: int, fin: int,
             min_barres: int = 600) -> np.ndarray:
    """Les journees ENTIERES de [debut, fin) : (n, 2) indices [a, b)."""
    jour = temps.dt.normalize().to_numpy()
    idx = np.arange(len(jour))
    df = pd.DataFrame({"j": jour, "i": idx})
    g = df.groupby("j")["i"].agg(["min", "max"])
    g = g[(g["min"] >= debut) & (g["max"] < fin)]
    # UNE JOURNEE TROP COURTE (trou de donnees) n'est pas une partie.
    g = g[(g["max"] - g["min"]) >= min_barres]
    return np.stack([g["min"].to_numpy(), g["max"].to_numpy() + 1], 1)


def etat_jeu(jetons, score, reste_min, cfg: JeuConfig) -> np.ndarray:
    """Les cinq colonnes du bloc de position, qui portent l'etat de la partie."""
    n = len(jetons)
    return np.stack([
        jetons / float(cfg.jetons),
        np.clip(score / cfg.vie_R, -1.0, 3.0),
        np.clip(reste_min / float(cfg.barres_par_jour), 0.0, 1.0),
        np.clip((score + cfg.vie_R) / cfg.vie_R, 0.0, 3.0),
        np.zeros(n),
    ], 1).astype(np.float32)


def observations(Xn: np.ndarray, t: np.ndarray, etat: np.ndarray, L: int):
    """(B, L, 71 + 5) : les L barres CLOSES jusqu'a t inclus, et l'etat."""
    idx = t[:, None] + np.arange(-L + 1, 1)[None, :]
    base = Xn[idx]
    blk = np.repeat(etat[:, None, :], L, axis=1)
    return np.concatenate([base, blk], axis=-1)


_NEG = -1e9


def _masque_logits(le: torch.Tensor, peut_achat: torch.Tensor,
                   peut_vente: Optional[torch.Tensor] = None) -> torch.Tensor:
    """ATTENDRE est toujours permis ; acheter et vendre chacun selon sa porte."""
    if peut_vente is None:
        peut_vente = peut_achat
    m = torch.zeros_like(le)
    m[:, ACHETER] = torch.where(peut_achat, 0.0, _NEG)
    m[:, VENDRE] = torch.where(peut_vente, 0.0, _NEG)
    return le + m


def portes(peut: np.ndarray, t: np.ndarray, rangs, cfg: JeuConfig):
    """(achat permis, vente permise) : les regles du jeu, puis la porte de
    l'expert. `rangs` (N, 2) sont les rangs glissants bruts de
    `features_expert` ; None = pas de porte."""
    g = float(getattr(cfg, "porte_rang_expert", 0.0))
    if rangs is None or g <= 0.0:
        return peut, peut
    return peut & (rangs[t, 0] >= g), peut & (rangs[t, 1] >= g)


def _choix(logits: torch.Tensor, explore: bool, gen=None) -> torch.Tensor:
    if not explore:
        return logits.argmax(-1)
    p = torch.softmax(logits, -1)
    return torch.multinomial(p, 1, generator=gen).squeeze(-1)


def departs_tires(jours: np.ndarray, rng, marge: int = 240) -> np.ndarray:
    """Les memes journees, commencees a une minute TIREE AU HASARD.

    POURQUOI, premier lancement du 2026-09-26 : la politique d'entrainement
    ouvrait sur 37 % des minutes. Ses six jetons partaient dans la premiere
    heure, et le reste de la journee n'etait jamais joue — le modele
    n'apprenait que l'ouverture de la journee. La validation, elle, joue
    toujours les journees entieres depuis minuit.
    """
    j = jours.copy()
    long_ = np.maximum(j[:, 1] - j[:, 0] - marge, 1)
    j[:, 0] = j[:, 0] + (rng.random(len(j)) * long_).astype(np.int64)
    return j


def _lim(fin_valide, t):
    """La borne d'un coup : un entier, ou une borne PAR LIGNE (multi-marches :
    la fin du segment dans le bloc du marche)."""
    return fin_valide[t] if isinstance(fin_valide, np.ndarray) else fin_valide


def joue(policy, jours: np.ndarray, Xn, R, D, S, fin_valide: int,
         cfg: JeuConfig, device, explore: bool, gen=None,
         collecte: bool = False, rangs=None):
    """Joue toutes les parties de `jours` EN PARALLELE.

    Rend (scores (G,), coups (liste de tuples), transitions par partie ou
    None). Un coup n'est permis que s'il se resout AVANT `fin_valide` :
    aucune partie de validation ne lit une barre du test.
    """
    G = len(jours)
    L = int(cfg.lookback)
    H = int(cfg.horizon_max)
    t = np.maximum(jours[:, 0], L - 1).astype(np.int64)
    fin = jours[:, 1].astype(np.int64)
    jet = np.full(G, cfg.jetons, np.int64)
    score = np.zeros(G, np.float64)
    actif = t < fin
    trans = [[] for _ in range(G)] if collecte else None
    coups = []
    policy.eval()
    while actif.any():
        g = np.flatnonzero(actif)
        tt = t[g]
        et = etat_jeu(jet[g], score[g], fin[g] - tt, cfg)
        ob = observations(Xn, tt, et, L)
        peut_np = (jet[g] > 0) & (tt + 1 + H < _lim(fin_valide, tt))
        # LA PORTE DE L'EXPERT, par sens. Voir `porte_rang_expert`.
        pa_np, pv_np = portes(peut_np, tt, rangs, cfg)
        with torch.no_grad():
            x = torch.from_numpy(ob).to(device)
            pa_t = torch.from_numpy(pa_np).to(device)
            pv_t = torch.from_numpy(pv_np).to(device)
            le, v, ltp, lsl = policy.jeu(x)
            le = _masque_logits(le, pa_t, pv_t)
            a = _choix(le, explore, gen)
            sidx = (a == VENDRE).long()
            ar = torch.arange(len(g), device=device)
            ltp_s, lsl_s = ltp[ar, sidx], lsl[ar, sidx]
            i = _choix(ltp_s, explore, gen)
            j = _choix(lsl_s, explore, gen)
            lp_e = torch.log_softmax(le, -1).gather(1, a[:, None]).squeeze(1)
            lp_i = torch.log_softmax(ltp_s, -1).gather(1, i[:, None]).squeeze(1)
            lp_j = torch.log_softmax(lsl_s, -1).gather(1, j[:, None]).squeeze(1)
        a, sidx = a.cpu().numpy(), sidx.cpu().numpy()
        i, j = i.cpu().numpy(), j.cpu().numpy()
        v = v.cpu().numpy()
        lp_e, lp_i, lp_j = lp_e.cpu().numpy(), lp_i.cpu().numpy(), lp_j.cpu().numpy()
        coup = a != ATTENDRE
        r = np.zeros(len(g))
        dur = np.ones(len(g), np.int64)
        if coup.any():
            c = np.flatnonzero(coup)
            tc = tt[c]
            r[c] = R[tc, sidx[c], i[c], j[c]]
            dur[c] = D[tc, sidx[c], i[c], j[c]]
            for q in c:
                coups.append((int(g[q]), int(tt[q]), int(sidx[q]), int(i[q]),
                              int(j[q]), float(r[q]), int(dur[q]),
                              int(S[tt[q], sidx[q], i[q], j[q]])))
        t[g] = tt + dur
        score[g] += r
        jet[g] -= coup.astype(np.int64)
        fini = (t[g] >= fin[g]) | (jet[g] <= 0) | (score[g] <= -cfg.vie_R)
        actif[g[fini]] = False
        if collecte:
            for q in range(len(g)):
                # LES DEUX PORTES, codees : bit 0 achat, bit 1 vente.
                trans[g[q]].append((int(tt[q]), et[q],
                                    int(pa_np[q]) + 2 * int(pv_np[q]),
                                    int(a[q]), int(i[q]), int(j[q]),
                                    float(lp_e[q]), float(lp_i[q]),
                                    float(lp_j[q]), float(v[q]), float(r[q]),
                                    int(dur[q]), bool(fini[q])))
    return score, coups, trans


# ======================================================================
# PPO
# ======================================================================
def avantages(trans, cfg: JeuConfig):
    """GAE semi-markovien : l'actualisation suit la DUREE de chaque decision."""
    lignes = []
    for tr in trans:
        n = len(tr)
        if n == 0:
            continue
        adv = np.zeros(n)
        prochain_v, prochain_a = 0.0, 0.0
        for k in range(n - 1, -1, -1):
            (_, _, _, _, _, _, _, _, _, vk, rk, dk, fk) = tr[k]
            # LES PERTES PESENT PLUS. Voir `poids_pertes`.
            if rk < 0.0:
                rk = rk * float(getattr(cfg, "poids_pertes", 1.0))
            g = cfg.gamma ** dk
            suite = 0.0 if fk else 1.0
            delta = rk + g * prochain_v * suite - vk
            prochain_a = delta + g * cfg.lam * prochain_a * suite
            adv[k] = prochain_a
            prochain_v = vk
        for k in range(n):
            lignes.append(tr[k] + (adv[k], adv[k] + tr[k][9]))
    return lignes


def maj_ppo(policy, optims, lignes, Xn, cfg: JeuConfig, device, rng):
    """Une mise a jour PPO, chaque groupe par son optimiseur.

    LES ATTENTES SONT SOUS-ECHANTILLONNEES, pas les coups. Une partie compte
    des centaines d'attentes pour six coups ; on garde tous les coups et un
    tirage des attentes, repondere pour que l'esperance ne change pas.
    """
    n = len(lignes)
    a_all = np.array([x[3] for x in lignes])
    coup = a_all != ATTENDRE
    n_c = int(coup.sum())
    garde = np.ones(n, bool)
    poids = np.ones(n, np.float32)
    place = max(cfg.max_transitions - n_c, 1)
    n_att = n - n_c
    if n_att > place:
        f = place / n_att
        att = np.flatnonzero(~coup)
        jet = rng.random(len(att)) >= f
        garde[att[jet]] = False
        poids[~coup] = 1.0 / f
    sel = np.flatnonzero(garde)
    L = int(cfg.lookback)
    tt = np.array([lignes[k][0] for k in sel])
    et = np.stack([lignes[k][1] for k in sel])
    peut = np.array([lignes[k][2] for k in sel])
    a = a_all[sel]
    i = np.array([lignes[k][4] for k in sel])
    j = np.array([lignes[k][5] for k in sel])
    lpe = np.array([lignes[k][6] for k in sel], np.float32)
    lpi = np.array([lignes[k][7] for k in sel], np.float32)
    lpj = np.array([lignes[k][8] for k in sel], np.float32)
    adv = np.array([lignes[k][13] for k in sel], np.float32)
    ret = np.array([lignes[k][14] for k in sel], np.float32)
    w = poids[sel]
    mu = float(np.average(adv, weights=w))
    sd = float(np.sqrt(np.average((adv - mu) ** 2, weights=w))) + 1e-8
    advn = (adv - mu) / sd
    st = {"kl": [], "clip": [], "H": [], "Hb": [], "v": []}
    policy.eval()   # le dropout rendrait le rapport de PPO inexact
    T = lambda z, dt=torch.float32: torch.as_tensor(z, dtype=dt, device=device)
    for _ in range(cfg.ppo_epochs):
        perm = rng.permutation(len(sel))
        for d0 in range(0, len(perm), cfg.minibatch):
            b = perm[d0:d0 + cfg.minibatch]
            x = T(observations(Xn, tt[b], et[b], L))
            le, v, ltp, lsl = policy.jeu(x)
            le = _masque_logits(le, T((peut[b] & 1) > 0, torch.bool),
                                T((peut[b] & 2) > 0, torch.bool))
            ab = T(a[b], torch.long)
            wb, Ab = T(w[b]), T(advn[b])
            logp = torch.log_softmax(le, -1)
            lp = logp.gather(1, ab[:, None]).squeeze(1)
            ratio = torch.exp(lp - T(lpe[b]))
            pe = -torch.min(ratio * Ab, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * Ab)
            perte = (pe * wb).sum() / wb.sum()
            ent = -(logp.exp() * logp).sum(-1)
            perte = perte - cfg.entropie_entree * (ent * wb).sum() / wb.sum()
            pv = (v - T(ret[b])) ** 2
            perte = perte + cfg.vf_coef * (pv * wb).sum() / wb.sum()
            cb = ab != ATTENDRE
            hb = torch.zeros((), device=device)
            if bool(cb.any()):
                sidx = (ab[cb] == VENDRE).long()
                ar = torch.arange(int(cb.sum()), device=device)
                for lg, choisi, lp_vieux in ((ltp, i, lpi), (lsl, j, lpj)):
                    lgc = torch.log_softmax(lg[cb][ar, sidx], -1)
                    ch = T(choisi[b], torch.long)[cb]
                    lpn = lgc.gather(1, ch[:, None]).squeeze(1)
                    rb = torch.exp(lpn - T(lp_vieux[b])[cb])
                    Ac = Ab[cb]
                    pb = -torch.min(rb * Ac, rb.clamp(1 - cfg.clip, 1 + cfg.clip) * Ac)
                    eb = -(lgc.exp() * lgc).sum(-1)
                    perte = perte + pb.mean() - cfg.entropie_barrieres * eb.mean()
                    hb = hb + eb.mean().detach() / 2
            _pas(policy, optims, perte, cfg)
            with torch.no_grad():
                st["kl"].append(float(((ratio - 1) - torch.log(ratio)).mean()))
                st["clip"].append(float(((ratio - 1).abs() > cfg.clip).float().mean()))
                st["H"].append(float(ent.mean()))
                st["Hb"].append(float(hb))
                st["v"].append(float(pv.mean()))
    return {k: float(np.mean(x)) if x else float("nan") for k, x in st.items()} | {
        "n": len(sel), "n_coups": n_c, "n_total": n}


# ======================================================================
# L'EXPERT
# ======================================================================
def coups_expert(jours, R, D, fin_valide: int, cfg: JeuConfig):
    """Chaque jour, les meilleurs coups A POSTERIORI, sans chevauchement."""
    H, L = int(cfg.horizon_max), int(cfg.lookback)
    K = R.shape[2] * R.shape[3]
    out = []
    for a0, b0 in jours:
        ts = np.arange(max(a0, L - 1), b0)
        ts = ts[ts + 1 + H < _lim(fin_valide, ts)]
        if len(ts) == 0:
            continue
        Rt = R[ts].reshape(len(ts), -1)
        Rt = np.where(np.isfinite(Rt), Rt, -np.inf)
        best = Rt.argmax(1)
        rb = Rt[np.arange(len(ts)), best]
        occupe = np.zeros(b0 - a0 + H + 2, bool)
        pris = 0
        for q in np.argsort(-rb, kind="stable"):
            if rb[q] < cfg.expert_R_min or pris >= cfg.expert_k:
                break
            t0 = int(ts[q])
            s, r2 = divmod(int(best[q]), K)
            i, j = divmod(r2, R.shape[3])
            d0 = int(D[t0, s, i, j])
            if occupe[t0 - a0: t0 - a0 + d0 + 1].any():
                continue
            occupe[t0 - a0: t0 - a0 + d0 + 1] = True
            out.append((t0, s, i, j, float(rb[q]), int(b0)))
            pris += 1
    return out


def blocs_marches(marche: np.ndarray):
    """Les blocs contigus d'un meme marche : [(nom, debut, fin), ...]."""
    out, a = [], 0
    for i in range(1, len(marche) + 1):
        if i == len(marche) or marche[i] != marche[a]:
            out.append((str(marche[a]), a, i))
            a = i
    return out


def fin_segment(t_ns: np.ndarray, blocs, t_fin_ns) -> np.ndarray:
    """Pour chaque ligne : le premier indice de SON bloc a partir de `t_fin`.
    Un coup n'est permis que s'il se resout avant — ni le segment suivant,
    ni le marche suivant ne sont lus."""
    out = np.empty(len(t_ns), np.int64)
    for _, a, b in blocs:
        out[a:b] = a + int(np.searchsorted(t_ns[a:b], t_fin_ns, side="left"))
    return out


def journees_multi(temps: pd.Series, blocs, t0_ns, t1_ns, min_barres: int) -> np.ndarray:
    """Les journees entieres de [t0, t1), marche par marche."""
    t_ns = temps.values.astype("int64")
    out = []
    for _, a, b in blocs:
        i0 = a + int(np.searchsorted(t_ns[a:b], t0_ns, side="left"))
        i1 = a + int(np.searchsorted(t_ns[a:b], t1_ns, side="left"))
        if i1 - i0 < min_barres:
            continue
        j = journees(temps.iloc[a:b].reset_index(drop=True), i0 - a, i1 - a, min_barres)
        if len(j):
            out.append(j + a)
    return np.concatenate(out) if out else np.zeros((0, 2), np.int64)


def expert_multi(Xn, y, t_ns, dans_train: np.ndarray, fin_tr: np.ndarray,
                 cfg: JeuConfig):
    """L'expert multi-marches : (predictions (N, 2), modeles du train entier).

    Les memes regles que `expert_realiste`, en DATES : les blocs croises du
    train sont des tranches de temps communes a tous les marches, purgees
    de `horizon_max` bougies de part et d'autre ; tout ce qui n'est pas le
    train est note par le modele du train entier."""
    import lightgbm as lgb
    H = int(cfg.horizon_max)
    purge = np.int64((H + 1) * int(cfg.minutes_par_barre) * 60 * 10**9)
    idx_tr = np.flatnonzero(dans_train)
    ok = idx_tr[idx_tr + 1 + H < fin_tr[idx_tr]]
    pred = np.full((len(Xn), 2), np.nan, np.float32)

    def _modeles(idx):
        ms = []
        for s_ in range(2):
            u = idx[np.isfinite(y[idx, s_])]
            m = lgb.LGBMRegressor(
                n_estimators=cfg.expert_arbres, learning_rate=0.03, num_leaves=31,
                min_child_samples=400, subsample=0.7, subsample_freq=1,
                colsample_bytree=0.7, reg_lambda=5.0, n_jobs=4, verbose=-1)
            m.fit(Xn[u], y[u, s_])
            ms.append(m)
        return ms

    bornes = np.quantile(t_ns[idx_tr], np.linspace(0, 1, cfg.expert_blocs + 1))
    for b in range(cfg.expert_blocs):
        e0, e1 = np.int64(bornes[b]), np.int64(bornes[b + 1])
        der = b == cfg.expert_blocs - 1
        dans_b = (t_ns[idx_tr] >= e0) & ((t_ns[idx_tr] <= e1) if der else (t_ns[idx_tr] < e1))
        app = ok[(t_ns[ok] < e0 - purge) | (t_ns[ok] > e1 + purge)][::cfg.expert_pas_app]
        ms = _modeles(app)
        r = idx_tr[dans_b]
        for s_ in range(2):
            pred[r, s_] = ms[s_].predict(Xn[r])
    ms = _modeles(ok[::cfg.expert_pas_app])
    hors = np.flatnonzero(~dans_train)
    for s_ in range(2):
        pred[hors, s_] = ms[s_].predict(Xn[hors])
    return pred, ms


def extras_btc(temps_bloc: pd.Series):
    """Les features Binance et Coinbase du BTC M15, alignees sur ses bougies.

    Elles ne concernent que le BTC, et le cache M15 qui les porte ne commence
    qu'en novembre 2024 : avant, elles valent NaN, ce que LightGBM sait
    traiter. Elles ne vont qu'a l'EXPERT du BTC, pas au modele commun.
    Rend None si le cache est absent.
    """
    import features_scalping as FS
    f = "data_cache_BTCUSD_M15.pkl"
    if not os.path.exists(f):
        return None
    cols = FS.colonnes_flux(15) + FS.COLONNES_COINBASE + ["flux_rang", "creux_x_flux"]
    b = pd.read_pickle(f)[["time"] + cols]
    m = pd.DataFrame({"time": temps_bloc.values}).merge(b, on="time", how="left")
    return m[cols].to_numpy(np.float32)


def par_jour(scores: np.ndarray, jours: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Le score de chaque JOUR : la somme des parties de tous les marches."""
    if len(scores) == 0:
        return np.zeros(0)
    return pd.Series(scores).groupby(dates[jours[:, 0]]).sum().to_numpy()


def ligne_marches(coups, marche: np.ndarray, cfg: JeuConfig) -> str:
    """Trades, dollars et profit factor de chaque marche."""
    if not coups:
        return "par marche : aucun trade"
    c = np.array(coups)
    t, r = c[:, 1].astype(np.int64), c[:, 5]
    m = marche[t]
    parts = []
    for nom in dict.fromkeys(m):
        rr = r[m == nom]
        g, p = rr[rr > 0].sum(), -rr[rr < 0].sum()
        parts.append(f"{nom} {len(rr)} trades {rr.sum() * cfg.risque_dollars:+.1f} $ "
                     f"PF {g / p if p > 0 else float('inf'):.2f}")
    return "par marche : " + "  |  ".join(parts)


def cibles_expert(R: np.ndarray) -> np.ndarray:
    """(N, 2) : le R net MOYEN de tous les coups d'un sens, a chaque minute.

    C'est la valeur d'acheter (ou de vendre) maintenant, toutes barrieres
    confondues — plus lisse que le meilleur coup, et sans choisir a la
    place du PPO les barrieres qu'il doit apprendre.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y = np.nanmean(R.reshape(R.shape[0], 2, -1), axis=2)
    return np.clip(y, -3.0, 5.0).astype(np.float32)


def expert_realiste(Xn, y, n_tr: int, fin_pred: int, cfg: JeuConfig):
    """Les predictions de l'expert : (N, 2), NaN hors de [0, fin_pred).

    SUR LE TRAIN, VALIDATION CROISEE PAR BLOCS PURGES : chaque bloc est
    predit par un modele appris sur les AUTRES, sans les `horizon_max`
    minutes qui le bordent — un coup y lirait le bloc predit. Les coups que
    l'expert enseigne sont donc choisis sur des minutes qu'il n'a pas
    apprises, comme ils le seraient en direct.

    AU-DELA DU TRAIN (la validation), un modele appris sur tout le train.
    Rien n'est predit au-dela de `fin_pred` : le test n'est pas touche.
    """
    import lightgbm as lgb
    H = int(cfg.horizon_max)
    N = len(Xn)
    pred = np.full((N, 2), np.nan, np.float32)
    app_tout = np.arange(0, max(n_tr - H - 1, 0), cfg.expert_pas_app)

    def _modeles(idx):
        ms = []
        for s_ in range(2):
            ok = idx[np.isfinite(y[idx, s_])]
            m = lgb.LGBMRegressor(
                n_estimators=cfg.expert_arbres, learning_rate=0.03,
                num_leaves=31, min_child_samples=400, subsample=0.7,
                subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                n_jobs=4, verbose=-1)
            m.fit(Xn[ok], y[ok, s_])
            ms.append(m)
        return ms

    bornes = np.linspace(0, n_tr, cfg.expert_blocs + 1).astype(np.int64)
    for b in range(cfg.expert_blocs):
        a0, b0 = int(bornes[b]), int(bornes[b + 1])
        app = app_tout[(app_tout < a0 - H - 1) | (app_tout >= b0 + H + 1)]
        ms = _modeles(app)
        for s_ in range(2):
            pred[a0:b0, s_] = ms[s_].predict(Xn[a0:b0])
    ms = _modeles(app_tout)
    for s_ in range(2):
        pred[n_tr:fin_pred, s_] = ms[s_].predict(Xn[n_tr:fin_pred])
    return pred, ms


NOMS_EXPERT = ["expert_achat", "expert_vente",
               "expert_rang_achat", "expert_rang_vente"]


def features_expert(pred: np.ndarray, fenetre: int = 10_000) -> np.ndarray:
    """(N, 4) : les deux predictions de l'expert et leur RANG GLISSANT.

    Le rang compare la minute aux `fenetre` minutes PRECEDENTES (et a
    elle-meme) : il ne lit que le passe, et il ne depend pas de l'echelle
    des predictions — celle des blocs croises du train n'est pas celle du
    modele appris sur tout le train. C'est sur ce rang que l'expert, joue
    seul, gagnait en validation.
    """
    out = np.zeros((len(pred), 4), np.float32)
    for s_ in range(2):
        q = pd.Series(pred[:, s_].astype(np.float64))
        out[:, s_] = q.fillna(0.0).to_numpy()
        out[:, 2 + s_] = (q.rolling(fenetre, min_periods=min(2_000, fenetre))
                          .rank(pct=True).fillna(0.5).to_numpy())
    return out


def _ref(cfg: JeuConfig):
    """Le coup de reference de l'expert : objectif et stop de 4 ATR, les
    deuxiemes du menu."""
    return min(1, len(cfg.tp_atr) - 1), min(1, len(cfg.sl_atr) - 1)


def coups_expert_predits(jours, pred, D, fin_valide: int, cfg: JeuConfig):
    """Chaque jour, les minutes que l'expert PREDIT les meilleures.

    Au plus `expert_k`, sans chevauchement, et seulement si le R net
    predit est positif. Meme format que `coups_expert`.
    """
    H, L = int(cfg.horizon_max), int(cfg.lookback)
    ri, rj = _ref(cfg)
    out = []
    for a0, b0 in jours:
        ts = np.arange(max(a0, L - 1), b0)
        ts = ts[ts + 1 + H < _lim(fin_valide, ts)]
        if len(ts) == 0:
            continue
        p_ = pred[ts]
        ok = np.isfinite(p_).all(1)
        ts, p_ = ts[ok], p_[ok]
        if len(ts) == 0:
            continue
        sens = p_.argmax(1)
        v = p_.max(1)
        occupe = np.zeros(b0 - a0 + H + 2, bool)
        pris = 0
        for q in np.argsort(-v, kind="stable"):
            if v[q] <= 0.0 or pris >= cfg.expert_k:
                break
            t0 = int(ts[q])
            d0 = int(D[t0, sens[q], ri, rj])
            if occupe[t0 - a0: t0 - a0 + d0 + 1].any():
                continue
            occupe[t0 - a0: t0 - a0 + d0 + 1] = True
            out.append((t0, int(sens[q]), ri, rj, float(v[q]), int(b0)))
            pris += 1
    return out


def joue_expert(jours, pred, R, D, S, fin_valide: int, seuil: float,
                cfg: JeuConfig):
    """L'expert joue SEUL, minute apres minute, SANS voir la suite du jour.

    Il ouvre le coup de reference des que son R predit depasse `seuil`,
    avec les memes regles que le PPO : jetons, vie, un coup a la fois.
    Rend (scores, coups) au format de `joue`.
    """
    H, L = int(cfg.horizon_max), int(cfg.lookback)
    ri, rj = _ref(cfg)
    scores, coups = [], []
    for g, (a0, b0) in enumerate(jours):
        t, jet, sc = max(int(a0), L - 1), cfg.jetons, 0.0
        while t < b0 and jet > 0 and sc > -cfg.vie_R:
            p_ = pred[t]
            if (t + 1 + H < _lim(fin_valide, t) and np.isfinite(p_).all()
                    and float(p_.max()) >= seuil):
                s_ = int(p_.argmax())
                r = float(R[t, s_, ri, rj])
                d = int(D[t, s_, ri, rj])
                coups.append((g, t, s_, ri, rj, r, d, int(S[t, s_, ri, rj])))
                sc += r
                jet -= 1
                t += d
            else:
                t += 1
        scores.append(sc)
    return np.asarray(scores), coups


def imite_expert(policy, optims, pos, jours, Xn, fin_valide, cfg, device, rng):
    """L'apprentissage par imitation : l'ENTREE de l'expert, et elle seule.

    Les tetes de barrieres n'y apprennent rien : le choix a posteriori de
    l'expert est biaise vers le coup le plus risque. Voir l'en-tete.
    """
    n_j = max(len(jours), 1)
    print(f"  expert  {len(pos):,} coups enseignes sur {len(jours)} journees "
          f"({len(pos)/n_j:.1f} par jour), choisis sur des minutes que "
          f"l'expert n'a pas apprises", flush=True)
    t_pos = {p[0] for p in pos}
    H, L = int(cfg.horizon_max), int(cfg.lookback)
    neg = []
    for a0, b0 in jours:
        for t0 in range(max(a0, L - 1), b0, cfg.expert_pas_neg):
            if t0 not in t_pos and t0 + 1 + H < _lim(fin_valide, t0):
                neg.append((t0, b0))
    tt = np.array([p[0] for p in pos] + [q[0] for q in neg], np.int64)
    fin = np.array([p[5] for p in pos] + [q[1] for q in neg], np.int64)
    act = np.array([ACHETER if p[1] == 0 else VENDRE for p in pos]
                   + [ATTENDRE] * len(neg), np.int64)
    ii = np.array([p[2] for p in pos] + [0] * len(neg), np.int64)
    jj = np.array([p[3] for p in pos] + [0] * len(neg), np.int64)
    w = np.where(act != ATTENDRE, cfg.expert_poids_pos, 1.0).astype(np.float32)
    # UNE PARTIE NEUVE : tous les jetons, score nul. Le PPO apprendra le
    # reste de l'etat en jouant.
    et = etat_jeu(np.full(len(tt), cfg.jetons), np.zeros(len(tt)), fin - tt, cfg)
    print(f"  expert  imitation sur {len(tt):,} minutes : {len(pos):,} coups, "
          f"{len(neg):,} attentes (une minute sur {cfg.expert_pas_neg}), "
          f"poids des coups x{cfg.expert_poids_pos:g}", flush=True)
    T = lambda z, dt=torch.float32: torch.as_tensor(z, dtype=dt, device=device)
    policy.train()
    for ep in range(cfg.expert_epochs):
        perm = rng.permutation(len(tt))
        pertes, justes_c, n_c = [], 0, 0
        for d0 in range(0, len(perm), 512):
            b = perm[d0:d0 + 512]
            x = T(observations(Xn, tt[b], et[b], L))
            le, _, ltp, lsl = policy.jeu(x)
            ab, wb = T(act[b], torch.long), T(w[b])
            pe = (F.cross_entropy(le, ab, reduction="none") * wb).sum() / wb.sum()
            cb = ab != ATTENDRE
            if bool(cb.any()):
                justes_c += int((le[cb].argmax(-1) == ab[cb]).sum())
                n_c += int(cb.sum())
            _pas(policy, optims, pe, cfg)
            pertes.append(float(pe))
        print(f"  expert  passe {ep + 1}/{cfg.expert_epochs}  perte "
              f"{np.mean(pertes):.4f}  coups de l'expert reconnus "
              f"{100 * justes_c / max(n_c, 1):.1f}%", flush=True)
    policy.eval()


# ======================================================================
# LE BILAN D'UNE SERIE DE PARTIES
# ======================================================================
def bilan(scores, coups, close, atr, sp, cfg: JeuConfig,
          frac: float = 1.0, ordre=None) -> Dict[str, float]:
    ksl = np.asarray(cfg.sl_atr)
    n = len(coups)
    b = {"parties": len(scores), "score": float(np.mean(scores)) if len(scores) else 0.0,
         "gagnees": float(np.mean(scores > 0)) if len(scores) else 0.0,
         "perdues": float(np.mean(scores < 0)) if len(scores) else 0.0,
         "coups": n}
    if n == 0:
        return b | {"net": float("nan"), "brut": float("nan"), "cout": float("nan"),
                    "pf": float("nan"), "achat": float("nan"), "sorties": (0, 0, 0),
                    "tp": (0, 0, 0), "sl": (0, 0, 0), "duree": float("nan"),
                    "gain_R": float("nan")}
    c = np.array(coups)
    t, s, i, j = (c[:, k].astype(np.int64) for k in (1, 2, 3, 4))
    r, du, so = c[:, 5], c[:, 6], c[:, 7].astype(np.int64)
    net = r * ksl[j] * atr[t] / close[t] * 1e4
    # LE COUT REELLEMENT PAYE : pendant la rampe, une part seulement.
    cout = frac * (sp[t] + cfg.glissement_entree_bps
                   + np.where(so != 0, cfg.glissement_sortie_bps, 0.0))
    g, p = r[r > 0].sum(), -r[r < 0].sum()
    # LE DRAWDOWN — la pire baisse du compte, sommet a creux, en dollars et
    # en % du capital, les coups joues dans l'ordre du temps. Demande du
    # proprietaire, 2026-09-27, avec les comptes ci-dessous.
    # EN MULTI-MARCHES, l'indice d'une ligne ne suit pas le temps (les
    # marches sont empiles) : `ordre` donne l'instant de chaque ligne.
    ordre = np.argsort(t if ordre is None else np.asarray(ordre)[t], kind="stable")
    eq = cfg.capital + np.cumsum(r[ordre] * cfg.risque_dollars)
    eq = np.concatenate([[cfg.capital], eq])
    pic = np.maximum.accumulate(eq)
    dd_d = float((eq - pic).min())
    dd_p = float(((eq - pic) / pic).min())
    # LA MISE REDUITE EN BAISSE, jouee dans le meme ordre. Voir `seuil_baisse`.
    ep = compte_prudent(r[ordre], cfg)
    pic_p = np.maximum.accumulate(ep)
    return b | {
        "prudent_total": float(ep[-1] - cfg.capital),
        "prudent_dd_dollars": float((ep - pic_p).min()),
        "prudent_dd_pct": float(((ep - pic_p) / pic_p).min()),
        "gagnants": int((r > 0).sum()), "perdants": int((r < 0).sum()),
        "longs": int((s == 0).sum()), "shorts": int((s == 1).sum()),
        "win_rate": float((r > 0).mean()), "dd_dollars": dd_d, "dd_pct": dd_p,
        "total_dollars": float(r.sum() * cfg.risque_dollars),
        "net": float(net.mean()), "brut": float((net + cout).mean()),
        "cout": float(cout.mean()), "pf": float(g / p) if p > 0 else float("inf"),
        "achat": float(np.mean(s == 0)),
        "sorties": tuple(float(np.mean(so == k)) for k in range(3)),
        "tp": tuple(float(np.mean(i == k)) for k in range(len(cfg.tp_atr))),
        "sl": tuple(float(np.mean(j == k)) for k in range(len(cfg.sl_atr))),
        "duree": float(np.median(du)), "gain_R": float(r.mean()),
        "t_brut": float((net + cout).mean() / ((net + cout).std(ddof=1)
                        / np.sqrt(n) + 1e-12)) if n > 1 else float("nan"),
    }


def compte_prudent(r_ordonnes: np.ndarray, cfg: JeuConfig) -> np.ndarray:
    """Le compte, coup apres coup, avec la mise reduite en baisse.

    Avant chaque coup : si le compte est a plus de `seuil_baisse` sous son
    plus haut, la mise vaut `mise_en_baisse` ; sinon elle est entiere.
    Rend le compte apres chaque coup, capital de depart en tete.
    """
    eq = [float(cfg.capital)]
    pic = float(cfg.capital)
    for r in r_ordonnes:
        baisse = (pic - eq[-1]) / pic
        m = float(cfg.mise_en_baisse) if baisse >= float(cfg.seuil_baisse) else 1.0
        eq.append(eq[-1] + float(r) * cfg.risque_dollars * m)
        pic = max(pic, eq[-1])
    return np.asarray(eq)


def ligne_prudente(b, cfg: JeuConfig) -> str:
    """Le meme bilan, mise reduite en baisse — gain, drawdown, et leur rapport."""
    if b["coups"] == 0:
        return "aucun trade"
    r_fixe = b["total_dollars"] / abs(b["dd_dollars"]) if b["dd_dollars"] < 0 else float("inf")
    r_prud = (b["prudent_total"] / abs(b["prudent_dd_dollars"])
              if b["prudent_dd_dollars"] < 0 else float("inf"))
    return (f"mise reduite de moitie a -{100 * cfg.seuil_baisse:g} % du plus haut : total "
            f"{b['prudent_total']:+.2f} $  |  drawdown max {b['prudent_dd_dollars']:+.2f} $ "
            f"({100 * b['prudent_dd_pct']:+.1f} %)  |  gain / drawdown {r_prud:.1f} "
            f"(mise fixe : {r_fixe:.1f})")


def ligne_detail(b, cfg: JeuConfig) -> str:
    """Le bilan lisible : gagnants, perdants, longs, shorts, PF, drawdown."""
    if b["coups"] == 0:
        return "aucun trade"
    return (f"{b['coups']} trades : {b['gagnants']} gagnants, {b['perdants']} perdants "
            f"(win rate {100 * b['win_rate']:.1f} %)  |  {b['longs']} longs, "
            f"{b['shorts']} shorts  |  profit factor {b['pf']:.2f}  |  drawdown max "
            f"{b['dd_dollars']:+.2f} $ ({100 * b['dd_pct']:+.1f} %)  |  total "
            f"{b['total_dollars']:+.2f} $ sur {b['parties']} jours")


def ligne_bilan(b, cfg: JeuConfig) -> str:
    if b["coups"] == 0:
        return (f"parties {b['parties']}  AUCUN COUP JOUE  score +0.000 R/partie")
    return (f"score {b['score']:+.3f} R/partie ({b['score'] * cfg.risque_dollars:+.2f}$)  "
            f"gagnees {100 * b['gagnees']:.0f}% perdues {100 * b['perdues']:.0f}% "
            f"sur {b['parties']}  coups {b['coups']} ({b['coups'] / max(b['parties'], 1):.1f}"
            f"/partie, {b['gain_R']:+.3f} R/coup)  net {b['net']:+.2f} bps  "
            f"brut {b['brut']:+.2f} (t {b.get('t_brut', float('nan')):+.1f})  "
            f"cout {b['cout']:.2f}  PF {b['pf']:.2f}")


def ligne_style(b, cfg: JeuConfig) -> str:
    if b["coups"] == 0:
        return "aucun coup"
    so = b["sorties"]
    return (f"achat {100 * b['achat']:.0f}% vente {100 * (1 - b['achat']):.0f}%  |  "
            f"sorties objectif {100 * so[0]:.0f}% stop {100 * so[1]:.0f}% "
            f"temps {100 * so[2]:.0f}%  duree {b['duree'] * cfg.minutes_par_barre:.0f} min  |  objectif "
            + "/".join(f"{100 * x:.0f}" for x in b["tp"])
            + f"% sur {'/'.join(f'{x:g}' for x in cfg.tp_atr)} ATR  stop "
            + "/".join(f"{100 * x:.0f}" for x in b["sl"])
            + f"% sur {'/'.join(f'{x:g}' for x in cfg.sl_atr)} ATR")


# ======================================================================
# LE WALK-FORWARD
# ======================================================================
def main() -> int:
    cfg = JeuConfig()
    if tuple(cfg.marches):
        return main_multi(cfg)
    rng = np.random.default_rng(cfg.graine)
    torch.manual_seed(cfg.graine)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifeste = f"run_{cfg.prefixe}.json"
    if os.path.exists(manifeste):
        raise FileExistsError(f"{manifeste} existe : nouveau prefixe, ou lancer.ps1")
    with open(manifeste, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=1, default=str)

    print("=" * 70)
    print(f"  KAIROS EN JEU — BTCUSD M{cfg.minutes_par_barre} : une journee = une partie")
    print("=" * 70)
    print(f"  regles : {cfg.jetons} coups par partie, fin de partie a "
          f"-{cfg.vie_R:g} R, coup = sens + objectif {cfg.tp_atr} ATR + stop "
          f"{cfg.sl_atr} ATR, temps limite {cfg.horizon_max} bougies "
          f"({cfg.horizon_max * cfg.minutes_par_barre} min)")
    print(f"  barrieres placees sur l'ATR M{cfg.minutes_par_barre}, jamais sous {cfg.atr_min_bps:g} bps "
          f"(stop minimum {min(cfg.sl_atr) * cfg.atr_min_bps:g} bps)")
    print(f"  un R = {cfg.risque_dollars:.0f}$ ({cfg.risque_pct:g}% de "
          f"{cfg.capital:.0f}$)  |  quatre tetes : achat, vente, gain (objectif), "
          f"perte (stop), chacune son optimiseur  |  {device}", flush=True)

    d = pd.read_pickle(cfg.cache)
    cols = list(dict.fromkeys(["time", "open", "high", "low", "close", "atr_14",
                               "spread_bar"] + colonnes_jeu(cfg)))
    d = d[cols].reset_index(drop=True)
    N = len(d)
    o, h, l, c = (d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    # L'ATR DES BARRIERES, plancher compris : la table, l'expert et le
    # bilan lisent tous le meme. Voir `atr_min_bps`.
    atr = atr_effectif(d["atr_14"].to_numpy(np.float64), c, cfg)
    sp = d["spread_bar"].to_numpy(np.float64)
    n_tr, n_va, n_te = (int(N * x) for x in (cfg.part_train, cfg.part_val, cfg.part_test))
    pas_wf = n_te
    print(f"[CACHE] {N:,} bougies de {cfg.minutes_par_barre} min, "
          f"{len(colonnes_jeu(cfg))} features, "
          f"{d['time'].iloc[0]} -> {d['time'].iloc[-1]}", flush=True)

    X = d[colonnes_jeu(cfg)].to_numpy(np.float32)
    mean = X[:n_tr].astype(np.float64).mean(0)
    std = X[:n_tr].astype(np.float64).std(0)
    stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    Xn = safe_normalize(X, stats).astype(np.float32)
    del X
    np.savez(f"{cfg.prefixe}_norm.npz", mean=stats["mean"], std=stats["std"],
             features=np.array(colonnes_jeu(cfg)))
    print(f"  normalisation figee sur le train du fold 1 [0 : {n_tr:,})", flush=True)

    t0 = time.time()
    R0, D0, _ = table_coups(o, h, l, sp, atr, cfg, 0.0)
    R1, D1, S1 = table_coups(o, h, l, sp, atr, cfg, 1.0)
    print(f"  table des coups : {N:,} bougies x {R1[0].size} coups, sans cout et "
          f"au cout reel, {time.time() - t0:.0f} s", flush=True)

    # L'EXPERT, APPRIS SUR LE TRAIN DU FOLD 1 SEULEMENT. Voir l'en-tete.
    t_ex = time.time()
    y_ex = cibles_expert(R1)
    pred, modeles_expert = expert_realiste(Xn, y_ex, n_tr, N, cfg)
    for s_, nom in enumerate(("achat", "vente")):
        modeles_expert[s_].booster_.save_model(f"expert_{cfg.prefixe}_{nom}.txt")
    # LE RANG GLISSANT SUR ~7 JOURS, en bougies.
    FE = features_expert(pred, fenetre=10_000 // int(cfg.minutes_par_barre))
    # Les rangs BRUTS, pour la porte d'entree (pas les colonnes normalisees).
    rangs_ex = FE[:, 2:4].copy()
    m_e = FE[:n_tr].astype(np.float64).mean(0)
    s_e = FE[:n_tr].astype(np.float64).std(0)
    FEn = np.clip((FE - m_e) / (s_e + 1e-8), -5.0, 5.0).astype(np.float32)
    Xn = np.concatenate([Xn, FEn[:, :cfg.n_expert]], axis=1)
    stats = {"mean": np.concatenate([stats["mean"], m_e[:cfg.n_expert].astype(np.float32)]),
             "std": np.concatenate([stats["std"], s_e[:cfg.n_expert].astype(np.float32)])}
    noms_entree = colonnes_jeu(cfg) + NOMS_EXPERT[:cfg.n_expert]
    np.savez(f"{cfg.prefixe}_norm.npz", mean=stats["mean"], std=stats["std"],
             features=np.array(noms_entree))
    ics = []
    for s_ in range(2):
        ok = np.isfinite(y_ex[n_tr:n_tr + n_va, s_])
        ics.append(pd.Series(pred[n_tr:n_tr + n_va, s_][ok]).corr(
            pd.Series(y_ex[n_tr:n_tr + n_va, s_][ok]), method="spearman"))
    print(f"  expert  LightGBM appris sur le train du fold 1, {time.time() - t_ex:.0f} s ; "
          f"correlation avec le R moyen en validation : achat {ics[0]:+.3f}, "
          f"vente {ics[1]:+.3f} ; {cfg.n_expert} colonnes ajoutees au modele ; "
          f"porte d'entree : rang de l'expert >= {cfg.porte_rang_expert:g} dans "
          f"le sens joue "
          f"({len(noms_entree)} en tout), modeles sauvegardes "
          f"expert_{cfg.prefixe}_*.txt", flush=True)

    policy = PolitiqueJeu(cfg).to(device)
    optims = optimiseurs(policy, cfg)
    n_par = sum(q.numel() for v in policy.groupes_jeu().values() for q in v)
    print(f"  modele : {n_par:,} parametres entraines, "
          + ", ".join(f"{k} {sum(q.numel() for q in v):,}"
                      for k, v in policy.groupes_jeu().items()), flush=True)

    precedent = None
    for fold in range(cfg.n_folds):
        s0 = fold * pas_wf
        a_tr, a_va, a_te = s0, s0 + n_tr, s0 + n_tr + n_va
        f_te = min(a_te + n_te, N)
        suffixe = f"_wf{fold + 1}"
        # UNE PARTIE = UNE JOURNEE, et au moins 40 % de ses bougies.
        _mb = int(0.4 * cfg.barres_par_jour)
        j_tr = journees(d["time"], a_tr, a_va, _mb)
        j_va = journees(d["time"], a_va, a_te, _mb)
        j_te = journees(d["time"], a_te, f_te, _mb)
        print(f"\n--- Fold {fold + 1} : train [{a_tr:,} : {a_va:,}) {len(j_tr)} parties  "
              f"validation [{a_va:,} : {a_te:,}) {len(j_va)} parties  "
              f"test {len(j_te)} parties ---", flush=True)
        if precedent is not None:
            policy.load_state_dict(torch.load(precedent, map_location=device,
                                              weights_only=False)["modele"])
            optims = optimiseurs(policy, cfg)
            print(f"  reprend {precedent}", flush=True)
        depart_zero = precedent is None
        if depart_zero:
            pos = coups_expert_predits(j_tr, pred, D1, a_va, cfg)
            ri_, rj_ = _ref(cfg)
            r_pos = [float(R1[q[0], q[1], ri_, rj_]) for q in pos]
            print(f"  expert  ses coups d'entrainement ont rapporte "
                  f"{np.mean(r_pos) if r_pos else float('nan'):+.3f} R en moyenne "
                  f"(predits sans les avoir appris)", flush=True)
            imite_expert(policy, optims, pos, j_tr, Xn, a_va, cfg, device, rng)
        best_path = f"best_{cfg.prefixe}{suffixe}.pth"
        record = 0.0
        garde_rec = None
        ep0 = 0 if depart_zero else 1
        for epoch in range(ep0, cfg.epochs + 1):
            t_ep = time.time()
            frac = (min(1.0, max(0.0, (epoch - 1) / cfg.rampe_cout))
                    if (depart_zero and cfg.rampe_cout > 0) else 1.0)
            st = None
            b_tr = None
            if epoch >= 1:
                Rf = (1 - frac) * R0 + frac * R1 if frac < 1.0 else R1
                Df = D1 if frac >= 0.5 else D0
                choix = rng.choice(len(j_tr), size=min(cfg.parties_par_epoch, len(j_tr)),
                                   replace=False)
                sc, cp, tr = joue(policy, departs_tires(
                                      j_tr[choix], rng, 240 // int(cfg.minutes_par_barre)), Xn,
                                  Rf, Df, S1, a_va, cfg, device, explore=True,
                                  collecte=True, rangs=rangs_ex)
                b_tr = bilan(sc, cp, c, atr, sp, cfg, frac=frac)
                st = maj_ppo(policy, optims, avantages(tr, cfg), Xn, cfg, device, rng)
                del Rf
            gen = torch.Generator(device=device)
            gen.manual_seed(cfg.graine)
            sv, cv, _ = joue(policy, j_va, Xn, R1, D1, S1, a_te, cfg, device,
                             explore=False, gen=gen, rangs=rangs_ex)
            bv = bilan(sv, cv, c, atr, sp, cfg)
            nom = "EXPERT IMITE" if epoch == 0 else f"cout {100 * frac:.0f}%"
            print(f"\nEPOCH {epoch:03d}  {nom:>12}  VAL  {ligne_bilan(bv, cfg)}  "
                  f"{(time.time() - t_ep) / 60:.1f} min", flush=True)
            print(f"  bilan  {ligne_detail(bv, cfg)}", flush=True)
            print(f"  bilan  {ligne_prudente(bv, cfg)}", flush=True)
            print(f"  jeu  validation  {ligne_style(bv, cfg)}", flush=True)
            if st is not None:
                print(f"  jeu  entrainement  {ligne_bilan(b_tr, cfg)}", flush=True)
                print(f"  jeu  PPO  {st['n']:,} decisions ({st['n_coups']:,} coups, "
                      f"sur {st['n_total']:,})  H entree {st['H']:.3f}/1.099  "
                      f"H barrieres {st['Hb']:.3f}/1.099  KL {st['kl']:+.4f}  "
                      f"clip {100 * st['clip']:.0f}%  critique {st['v']:.4f}", flush=True)
            etat = {"modele": policy.state_dict(), "config": asdict(cfg),
                    "features": noms_entree, "mean": stats["mean"],
                    "std": stats["std"], "epoch": epoch, "fold": fold + 1}
            torch.save(etat, f"last_{cfg.prefixe}{suffixe}.pth")
            if bv["coups"] >= cfg.min_coups_val and bv["score"] > record:
                record = bv["score"]
                torch.save(etat, best_path)
                garde_rec = epoch
                print(f"  sauvegarde  NOUVEAU MEILLEUR : {bv['score']:+.3f} R/partie "
                      f"en validation, {bv['coups']} coups -> {best_path}", flush=True)
            else:
                pourquoi = (f"{bv['coups']} coups, il en faut {cfg.min_coups_val}"
                            if bv["coups"] < cfg.min_coups_val else
                            f"{bv['score']:+.3f} R/partie ne bat pas {record:+.3f}")
                print(f"  garde  rien de sauvegarde : {pourquoi}"
                      + (f" (meilleur : epoch {garde_rec})" if garde_rec is not None else ""),
                      flush=True)

        # LE TEST DU FOLD, UNE FOIS, sur le meilleur jeu de poids — ou le
        # dernier si aucun n'a battu zero. Rien n'y est choisi.
        src = best_path if os.path.exists(best_path) else f"last_{cfg.prefixe}{suffixe}.pth"
        policy.load_state_dict(torch.load(src, map_location=device,
                                          weights_only=False)["modele"])
        gen = torch.Generator(device=device)
        gen.manual_seed(cfg.graine)
        s_t, c_t, _ = joue(policy, j_te, Xn, R1, D1, S1, f_te, cfg, device,
                           explore=False, gen=gen, rangs=rangs_ex)
        bt = bilan(s_t, c_t, c, atr, sp, cfg)
        print(f"\nTEST fold {fold + 1} ({os.path.basename(src)})  {ligne_bilan(bt, cfg)}",
              flush=True)
        print(f"TEST fold {fold + 1} bilan  {ligne_detail(bt, cfg)}", flush=True)
        print(f"TEST fold {fold + 1} bilan  {ligne_prudente(bt, cfg)}", flush=True)
        print(f"TEST fold {fold + 1}  {ligne_style(bt, cfg)}", flush=True)
        with open(f"test_{cfg.prefixe}{suffixe}.json", "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in bt.items()}, fh, indent=1, default=str)
        precedent = src
    print("\nFIN du walk-forward", flush=True)
    return 0


def main_multi(cfg: JeuConfig) -> int:
    """Le jeu multi-marches. Voir `marches`."""
    rng = np.random.default_rng(cfg.graine)
    torch.manual_seed(cfg.graine)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifeste = f"run_{cfg.prefixe}.json"
    if os.path.exists(manifeste):
        raise FileExistsError(f"{manifeste} existe : nouveau prefixe, ou lancer.ps1")
    with open(manifeste, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=1, default=str)
    cols = colonnes_jeu(cfg)
    print("=" * 70)
    print(f"  KAIROS EN JEU — {len(cfg.marches)} MARCHES M{cfg.minutes_par_barre} : "
          f"une journee = une partie par marche, le jour = leur somme")
    print("=" * 70)
    print(f"  marches : {', '.join(cfg.marches)}")
    print(f"  regles : {cfg.jetons} coups par marche et par jour, fin de partie a "
          f"-{cfg.vie_R:g} R, coup = sens + objectif {cfg.tp_atr} ATR + stop "
          f"{cfg.sl_atr} ATR, temps limite {cfg.horizon_max} bougies "
          f"({cfg.horizon_max * cfg.minutes_par_barre} min)")
    print(f"  un R = {cfg.risque_dollars:.0f}$ ({cfg.risque_pct:g}% de "
          f"{cfg.capital:.0f}$)  |  quatre tetes : achat, vente, gain (objectif), "
          f"perte (stop), chacune son optimiseur  |  {device}", flush=True)

    d = pd.read_pickle(cfg.cache)
    d = d[d["marche"].isin(cfg.marches)].reset_index(drop=True)
    # SANS DOUBLON : `spread_bar` est aussi une feature. Deux colonnes du meme
    # nom feraient de `d["spread_bar"]` un tableau a deux colonnes.
    d = d[list(dict.fromkeys(["time", "marche", "open", "high", "low", "close",
                              "atr_14", "spread_bar"] + cols))].reset_index(drop=True)
    N = len(d)
    marche = d["marche"].to_numpy()
    blocs = blocs_marches(marche)
    t_ns = d["time"].values.astype("int64")
    dates = d["time"].dt.normalize().values
    o, h, l, c = (d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    sp = d["spread_bar"].to_numpy(np.float64)
    # LE PLANCHER PAR MARCHE. Voir `plancher_couts`.
    plancher = np.empty(N)
    for nom, a, b in blocs:
        cout_med = float(np.median(sp[a:b])) + cfg.glissement_entree_bps + cfg.glissement_sortie_bps
        plancher[a:b] = max(cfg.plancher_min_bps, cfg.plancher_couts * cout_med)
        print(f"  {nom:8s} {b - a:6d} bougies  {d['time'].iloc[a]:%Y-%m-%d} -> "
              f"{d['time'].iloc[b - 1]:%Y-%m-%d}  cout med {cout_med:5.2f} bps  "
              f"plancher des barrieres {plancher[a]:5.1f} bps", flush=True)
    atr = np.maximum(d["atr_14"].to_numpy(np.float64), plancher * 1e-4 * c)

    # LES FENETRES EN DATES, communes a tous les marches.
    t0, t1 = int(t_ns.min()), int(t_ns.max()) + 1
    span = t1 - t0
    bornes = []
    for f in range(cfg.n_folds):
        a_ = t0 + int(f * cfg.part_test * span)
        va = a_ + int(cfg.part_train * span)
        te = va + int(cfg.part_val * span)
        fi = min(te + int(cfg.part_test * span), t1)
        bornes.append((a_, va, te, fi))
    dans_train1 = t_ns < bornes[0][1]

    X = d[cols].to_numpy(np.float32)
    stats = {"mean": X[dans_train1].astype(np.float64).mean(0).astype(np.float32),
             "std": X[dans_train1].astype(np.float64).std(0).astype(np.float32)}
    Xn = safe_normalize(X, stats).astype(np.float32)
    del X

    t_tab = time.time()
    tabs = {}
    for frac in (0.0, 1.0):
        parts = [table_coups(o[a:b], h[a:b], l[a:b], sp[a:b], atr[a:b], cfg, frac)
                 for _, a, b in blocs]
        tabs[frac] = tuple(np.concatenate([p_[k] for p_ in parts]) for k in range(3))
    R0, D0, _ = tabs[0.0]
    R1, D1, S1 = tabs[1.0]
    del tabs
    print(f"  table des coups : {N:,} bougies x {R1[0].size} coups, marche par "
          f"marche, {time.time() - t_tab:.0f} s", flush=True)

    t_ex = time.time()
    fin_tr1 = fin_segment(t_ns, blocs, bornes[0][1])
    y_ex = cibles_expert(R1)
    # UN EXPERT PAR MARCHE — 2026-09-27, run kairos_multi_m15_02. Le run
    # m15_01 n'en avait qu'un pour les sept : correlation +0.019 a l'achat
    # et +0.010 a la vente en validation, contre +0.088 et +0.043 pour
    # l'expert du BTC seul. Un seul modele moyennait des marches qui ne se
    # comportent pas pareil, et le BTC avait perdu ses sources. Chaque marche
    # a desormais le sien, appris sur ses seules bougies ; celui du BTC lit
    # en plus Binance et Coinbase (`extras_btc`). Le SAINT reste commun.
    pred = np.full((N, 2), np.nan, np.float32)
    va1 = (t_ns >= bornes[0][1]) & (t_ns < bornes[0][2])
    for nom, a, b in blocs:
        Xb = Xn[a:b]
        ex = extras_btc(d["time"].iloc[a:b]) if nom == "BTCUSD" else None
        if ex is not None:
            Xb = np.concatenate([Xb, ex], axis=1)
        pb, ms = expert_multi(Xb, y_ex[a:b], t_ns[a:b], dans_train1[a:b],
                              fin_tr1[a:b] - a, cfg)
        pred[a:b] = pb
        for s_, sens in enumerate(("achat", "vente")):
            ms[s_].booster_.save_model(f"expert_{cfg.prefixe}_{nom}_{sens}.txt")
        ic_m = []
        for s_ in range(2):
            ok = va1[a:b] & np.isfinite(y_ex[a:b, s_])
            ic_m.append(pd.Series(pb[ok, s_]).corr(pd.Series(y_ex[a:b][ok, s_]),
                                                   method="spearman"))
        print(f"  expert  {nom:8s} correlation en validation : achat {ic_m[0]:+.3f}  "
              f"vente {ic_m[1]:+.3f}" + ("  (+ Binance et Coinbase)" if ex is not None else ""),
              flush=True)
    FE = np.concatenate([features_expert(pred[a:b], fenetre=10_000 // int(cfg.minutes_par_barre))
                         for _, a, b in blocs])
    rangs_ex = FE[:, 2:4].copy()
    m_e = FE[dans_train1].astype(np.float64).mean(0)
    s_e = FE[dans_train1].astype(np.float64).std(0)
    FEn = np.clip((FE - m_e) / (s_e + 1e-8), -5.0, 5.0).astype(np.float32)
    Xn = np.concatenate([Xn, FEn[:, :cfg.n_expert]], axis=1)
    stats = {"mean": np.concatenate([stats["mean"], m_e[:cfg.n_expert].astype(np.float32)]),
             "std": np.concatenate([stats["std"], s_e[:cfg.n_expert].astype(np.float32)])}
    noms_entree = cols + NOMS_EXPERT[:cfg.n_expert]
    np.savez(f"{cfg.prefixe}_norm.npz", mean=stats["mean"], std=stats["std"],
             features=np.array(noms_entree))
    ics = []
    for s_ in range(2):
        ok = va1 & np.isfinite(y_ex[:, s_])
        ics.append(pd.Series(pred[ok, s_]).corr(pd.Series(y_ex[ok, s_]), method="spearman"))
    print(f"  expert  sept LightGBM appris sur le train du fold 1, {time.time() - t_ex:.0f} s ; "
          f"correlation avec le R moyen en validation, tous marches : achat {ics[0]:+.3f}, "
          f"vente {ics[1]:+.3f} ; porte d'entree : rang de l'expert >= "
          f"{cfg.porte_rang_expert:g} dans le sens joue ({len(noms_entree)} entrees)",
          flush=True)

    policy = PolitiqueJeu(cfg).to(device)
    optims = optimiseurs(policy, cfg)
    n_par = sum(q.numel() for v in policy.groupes_jeu().values() for q in v)
    print(f"  modele : {n_par:,} parametres entraines", flush=True)

    def _bilan(scores, jours, coups, frac=1.0):
        return bilan(par_jour(scores, jours, dates), coups, c, atr, sp, cfg,
                     frac=frac, ordre=t_ns)

    _fmt = lambda x: pd.Timestamp(x).strftime("%Y-%m-%d")
    precedent = None
    _mb = int(0.4 * cfg.barres_par_jour)
    for fold, (a_, va, te, fi) in enumerate(bornes):
        suffixe = f"_wf{fold + 1}"
        j_tr = journees_multi(d["time"], blocs, a_, va, _mb)
        j_va = journees_multi(d["time"], blocs, va, te, _mb)
        j_te = journees_multi(d["time"], blocs, te, fi, _mb)
        fin_tr = fin_segment(t_ns, blocs, va)
        fin_va = fin_segment(t_ns, blocs, te)
        fin_te = fin_segment(t_ns, blocs, fi)
        n_jv = len(np.unique(dates[j_va[:, 0]])) if len(j_va) else 0
        print(f"\n--- Fold {fold + 1} : train {_fmt(a_)} -> {_fmt(va)} ({len(j_tr)} parties)  "
              f"validation -> {_fmt(te)} ({len(j_va)} parties, {n_jv} jours)  "
              f"test -> {_fmt(fi)} ({len(j_te)} parties) ---", flush=True)
        if precedent is not None:
            policy.load_state_dict(torch.load(precedent, map_location=device,
                                              weights_only=False)["modele"])
            optims = optimiseurs(policy, cfg)
            print(f"  reprend {precedent}", flush=True)
        depart_zero = precedent is None
        if depart_zero:
            pos = coups_expert_predits(j_tr, pred, D1, fin_tr, cfg)
            ri_, rj_ = _ref(cfg)
            r_pos = [float(R1[q[0], q[1], ri_, rj_]) for q in pos]
            print(f"  expert  ses coups d'entrainement ont rapporte "
                  f"{np.mean(r_pos) if r_pos else float('nan'):+.3f} R en moyenne "
                  f"(predits sans les avoir appris)", flush=True)
            imite_expert(policy, optims, pos, j_tr, Xn, fin_tr, cfg, device, rng)
        best_path = f"best_{cfg.prefixe}{suffixe}.pth"
        record, garde_rec = 0.0, None
        for epoch in range(0 if depart_zero else 1, cfg.epochs + 1):
            t_ep = time.time()
            frac = (min(1.0, max(0.0, (epoch - 1) / cfg.rampe_cout))
                    if (depart_zero and cfg.rampe_cout > 0) else 1.0)
            st = b_tr = None
            if epoch >= 1:
                Rf = (1 - frac) * R0 + frac * R1 if frac < 1.0 else R1
                Df = D1 if frac >= 0.5 else D0
                choix = rng.choice(len(j_tr), size=min(cfg.parties_par_epoch, len(j_tr)),
                                   replace=False)
                sc, cp, tr = joue(policy, departs_tires(
                                      j_tr[choix], rng, 240 // int(cfg.minutes_par_barre)),
                                  Xn, Rf, Df, S1, fin_tr, cfg, device, explore=True,
                                  collecte=True, rangs=rangs_ex)
                b_tr = bilan(sc, cp, c, atr, sp, cfg, frac=frac, ordre=t_ns)
                st = maj_ppo(policy, optims, avantages(tr, cfg), Xn, cfg, device, rng)
                del Rf
            gen = torch.Generator(device=device)
            gen.manual_seed(cfg.graine)
            sv, cv, _ = joue(policy, j_va, Xn, R1, D1, S1, fin_va, cfg, device,
                             explore=False, gen=gen, rangs=rangs_ex)
            bv = _bilan(sv, j_va, cv)
            nom = "EXPERT IMITE" if epoch == 0 else f"cout {100 * frac:.0f}%"
            print(f"\nEPOCH {epoch:03d}  {nom:>12}  VAL  {ligne_bilan(bv, cfg)}  "
                  f"{(time.time() - t_ep) / 60:.1f} min", flush=True)
            print(f"  bilan  {ligne_detail(bv, cfg)}", flush=True)
            print(f"  bilan  {ligne_prudente(bv, cfg)}", flush=True)
            print(f"  bilan  {ligne_marches(cv, marche, cfg)}", flush=True)
            print(f"  jeu  validation  {ligne_style(bv, cfg)}", flush=True)
            if st is not None:
                print(f"  jeu  entrainement  {ligne_bilan(b_tr, cfg)}", flush=True)
                print(f"  jeu  PPO  {st['n']:,} decisions ({st['n_coups']:,} coups)  "
                      f"H entree {st['H']:.3f}/1.099  H barrieres {st['Hb']:.3f}/1.386  "
                      f"KL {st['kl']:+.4f}  clip {100 * st['clip']:.0f}%", flush=True)
            etat = {"modele": policy.state_dict(), "config": asdict(cfg),
                    "features": noms_entree, "mean": stats["mean"], "std": stats["std"],
                    "epoch": epoch, "fold": fold + 1}
            torch.save(etat, f"last_{cfg.prefixe}{suffixe}.pth")
            if bv["coups"] >= cfg.min_coups_val and bv["score"] > record:
                record, garde_rec = bv["score"], epoch
                torch.save(etat, best_path)
                print(f"  sauvegarde  NOUVEAU MEILLEUR : {bv['score']:+.3f} R/partie "
                      f"en validation, {bv['coups']} coups -> {best_path}", flush=True)
            else:
                pourquoi = (f"{bv['coups']} coups, il en faut {cfg.min_coups_val}"
                            if bv["coups"] < cfg.min_coups_val else
                            f"{bv['score']:+.3f} R/partie ne bat pas {record:+.3f}")
                print(f"  garde  rien de sauvegarde : {pourquoi}"
                      + (f" (meilleur : epoch {garde_rec})" if garde_rec is not None else ""),
                      flush=True)
        src = best_path if os.path.exists(best_path) else f"last_{cfg.prefixe}{suffixe}.pth"
        policy.load_state_dict(torch.load(src, map_location=device,
                                          weights_only=False)["modele"])
        gen = torch.Generator(device=device)
        gen.manual_seed(cfg.graine)
        s_t, c_t, _ = joue(policy, j_te, Xn, R1, D1, S1, fin_te, cfg, device,
                           explore=False, gen=gen, rangs=rangs_ex)
        bt = _bilan(s_t, j_te, c_t)
        print(f"\nTEST fold {fold + 1} ({os.path.basename(src)})  {ligne_bilan(bt, cfg)}",
              flush=True)
        print(f"TEST fold {fold + 1} bilan  {ligne_detail(bt, cfg)}", flush=True)
        print(f"TEST fold {fold + 1} bilan  {ligne_prudente(bt, cfg)}", flush=True)
        print(f"TEST fold {fold + 1} bilan  {ligne_marches(c_t, marche, cfg)}", flush=True)
        print(f"TEST fold {fold + 1}  {ligne_style(bt, cfg)}", flush=True)
        with open(f"test_{cfg.prefixe}{suffixe}.json", "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in bt.items()}, fh, indent=1, default=str)
        precedent = src
    print("\nFIN du walk-forward", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
