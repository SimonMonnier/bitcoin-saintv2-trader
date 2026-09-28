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

import copy
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
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
    # LE JEU EN H1, PARTIES D'UNE SEMAINE — 2026-09-28, pistes 1 et 2 des
    # pistes non explorees (accord du proprietaire) : neuf ans d'historique
    # Binance au lieu de deux (`prepare_btc_h1_binance`), et un horizon ou le
    # cout pese peu (ATR H1 median 77.5 bps pour ~3.5 de cout, contre 26 en
    # M15). Les couts restent ceux de Vantage : spread mesure par creneau,
    # glissements, et le swap acheteur (-20 %/an) sur la duree de chaque coup.
    # Le run M15 precedent : prefixe kairos_jeu_m15_07, cache
    # data_cache_BTCUSD_M15.pkl, 15 min, partie "jour", horizon 32, sans swap.
    # LE MULTI-MARCHES H1 — 2026-09-28, demande du proprietaire : « revenir a
    # la configuration ou on a fait +400 $ au test du fold 1 » (run h1_01 :
    # une position a la fois, 10 jetons par semaine, l'expert enseigne 6 coups
    # sans chevauchement) « et ajouter des indices qui ont de la volatilite et
    # un faible spread » pour augmenter le nombre de trades et le profit.
    # Chaque marche joue sa semaine avec ses 10 jetons. Voir `prepare_multi_h1`.
    # RETOUR AU BTC SEUL — 2026-09-28, regle fixee par le proprietaire avant
    # la fin du run multi_h1_03 : « si ce n'est pas mieux que le BTC seul au
    # test du fold 1, relance le BTC seul pour retrouver les +400 $ ». Tests
    # du multi a 13 marches : +130.82 $ (PF 1.02, drawdown -65.6 %), -271.67 $,
    # -152.68 $ ; le BTC seul (h1_01) faisait +447.56 $ au fold 1 (PF 1.38,
    # drawdown -9.1 %). Configuration de h1_01 a l'identique.
    # LE JEU EN M5, UNE JOURNEE PAR PARTIE — 2026-09-28, demande du
    # proprietaire : « passer sur du M5, puisqu'on a des donnees M5 sur Coinbase
    # et Binance que l'on n'a pas sur MT5 ». Prix Binance depuis 2017, ecart
    # Coinbase / Binance, couts Vantage (voir `prepare_btc_m5`). Le reste : les
    # regles du jeu H1 ramenees a la journee (10 jetons, une position, porte
    # 0.90, 1 % de mise), expert a 4 coups par jour comme en M15. AVERTI : en M5
    # l'ATR median du BTC vaut ~22 bps pour ~3.5 de cout (16 %), contre 77 en H1.
    # Le run H1 precedent : kairos_jeu_h1_blocs_01, cache
    # data_cache_BTCUSD_H1_BINANCE.pkl, 60 min, partie "semaine", horizon 72,
    # expert_k 6, expert_pas_neg 1.
    # PLUS DE TRADES, MEME DIX JETONS — 2026-09-28 (run m5_02), demande du
    # proprietaire : « une selectivite plus large ». Seuls la porte (0.90 ->
    # 0.85) et l'expert (4 -> 10 coups enseignes par jour) changent. Le run
    # m5_01 jouait 1.8 a 3 coups par jour sur ses 10 jetons ; test du bloc 1
    # +465.49 $, PF 1.31.
    # PLUSIEURS CRYPTOS EN M5 — 2026-09-28 (run multi_m5_01), demande du
    # proprietaire : les cryptos de Vantage presentes sur Binance et Coinbase,
    # avec exactement les features du BTC, un seul modele et un petit cerveau
    # par marche. Voir `prepare_multi_m5` (BTC, ETH, ZEC : les seules dont le
    # spread Vantage reste sous l'ATR M5) et `main_multi_blocs`. Le BTC seul :
    # prefixe kairos_jeu_m5_02, `marches` = (), `cache` =
    # data_cache_BTCUSD_M5_BINANCE.pkl.
    prefixe: str = "kairos_multi_m5_01"
    # LA VALIDATION CROISEE PURGEE — 2026-09-28, demande du proprietaire :
    # « entrainer le modele sur des periodes aleatoires pour qu'il apprenne
    # tous les types de marches ». Le walk-forward (h1_05) : +447.56, -441.18,
    # -172.80 $. En "blocs", l'historique est coupe en `n_blocs` blocs ; chaque
    # bloc est teste a son tour par un modele NEUF entraine sur TOUS les
    # autres (passe et avenir), la validation etant le bloc le plus eloigne,
    # avec une zone tampon de `purge_semaines` + l'horizon + la memoire
    # retiree de l'entrainement autour du test et de la validation.
    # UN RESULTAT NEGATIF EST DEFINITIF (meme en voyant l'avenir des autres
    # blocs, pas d'avantage) ; un resultat positif est un PLAFOND, a confirmer
    # en walk-forward et en demo. "walk" = le walk-forward d'avant.
    validation: str = "blocs"
    n_blocs: int = 10
    purge_semaines: int = 1
    # LE JEU EN BOUGIES DE 15 MINUTES — 2026-09-27, demande du proprietaire :
    # « recommence le jeu avec des bougies de 15 minutes, et pas M1 pour
    # entrer ». Le modele voit des bougies M15 (contextes H1 et H4, voir
    # `prepare_btc_m15`) et ne decide qu'a leur cloture. Une partie reste
    # une journee : 96 decisions au lieu de 1440.
    #
    # CE QUE CELA CHANGE AU COUT : l'ATR M15 median vaut 26 bps, l'ATR M1
    # environ 6, pour le meme cout de ~3.3 bps par coup. Rapporte au
    # mouvement d'une bougie, il pese quatre fois moins.
    # MULTI M5 (run multi_m5_01) : data_cache_MULTI_M5.pkl. Le BTC seul en
    # M5 : data_cache_BTCUSD_M5_BINANCE.pkl.
    cache: str = "data_cache_MULTI_M5.pkl"
    minutes_par_barre: int = 5
    # UNE PARTIE = UN "jour" OU UNE "semaine" (lundi 0 h -> lundi 0 h UTC).
    # En H1 un jour ne fait que 24 decisions : la semaine en fait 168, et
    # laisse aux coups de plusieurs jours le temps de se resoudre.
    partie: str = "jour"
    # LE SWAP DU COURTIER, en bps du prix par jour de detention, compte au
    # prorata de la duree du coup. Vantage BTC : -20 %/an a l'achat, 0 a la
    # vente. Sans objet en M15 (coups de quelques heures) : 0 dans ces runs.
    swap_achat_bps_jour: float = 20.0 / 365 * 100
    swap_vente_bps_jour: float = 0.0
    # LE JEU MULTI-MARCHES — 2026-09-27, demande du proprietaire : un seul
    # jeu qui trade le BTC, l'ETH, l'or et les indices a petit spread et
    # bonne volatilite, pour maximiser le nombre de trades et le profit en
    # gardant un drawdown bas. Voir `prepare_multi_m15` et `main_multi`.
    #
    # CHAQUE MARCHE JOUE SA JOURNEE, avec ses jetons, sa vie et la porte de
    # l'expert ; le resultat du JOUR est la somme des marches, et le
    # drawdown se mesure sur le compte commun, coups dans l'ordre du temps.
    # Vide = le jeu d'un seul marche (`cache`).
    #
    # DESACTIVE LE 2026-09-27 (run kairos_multi_m15_02 arrete a l'epoch 2) :
    # meme avec un expert par marche, correlation de -0.02 a +0.05 en
    # validation (2024-09 -> 2025-05), 16 coups par jour, profit factor 0.92,
    # drawdown -108 % a 0.5 % de mise. Retour au BTC seul, le seul jeu positif
    # au test. Pour rejouer le multi-marches : les sept marches ici, et
    # `cache` = data_cache_MULTI_M15.pkl.
    # REACTIVE LE 2026-09-28 EN H1 (run kairos_multi_h1_01) : le BTC et six
    # indices a faible spread (voir `telecharge_h1_mt5`). Le BTC seul en H1 :
    # `marches` = () et `cache` = data_cache_BTCUSD_H1_BINANCE.pkl.
    # 7 -> 13 MARCHES LE 2026-09-28 (run kairos_multi_h1_03), demande du
    # proprietaire : « beaucoup plus d'indices, chacun avec son petit cerveau ».
    # () LE 2026-09-28 (run h1_05) : retour au BTC seul, voir `prefixe`. Le
    # multi a 13 marches : ces marches et `cache` = data_cache_MULTI_H1.pkl.
    # BTC, ETH, ZEC EN M5 LE 2026-09-28 (run multi_m5_01), demande du
    # proprietaire. Voir `prepare_multi_m5`.
    marches: Tuple[str, ...] = ("BTCUSD", "ETHUSD", "ZECUSD")
    # UN GROS MODELE DIVISE EN PETITS MODELES — 2026-09-28, demande du
    # proprietaire (run kairos_multi_h1_02). Le tronc SAINT reste commun a tous
    # les marches ; chaque marche a SES tetes : achat, vente, objectif, stop et
    # valeur. Le tronc apprend ce qui vaut partout, chaque marche garde des
    # decisions a lui. Le marche se lit dans ses colonnes `m_*`. Au run
    # multi_h1_01 (tetes communes), le BTC tombait a PF 1.09 en validation,
    # contre 1.43 quand il jouait seul avec son propre modele.
    tetes_par_marche: bool = True
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
    # 10 -> 1000 LE 2026-09-28 (run h1_04, jamais mene a terme) pour ouvrir
    # autant de trades que la marge le permet. REMIS A 10 LE 2026-09-28 (run
    # kairos_multi_h1_01) : retour a la configuration du run h1_01.
    jetons: int = 10
    vie_R: float = 3.0
    # PLUSIEURS POSITIONS A LA FOIS — 2026-09-28, demande du proprietaire
    # (run h1_02) : le run h1_01 n'utilisait que 2 a 3 de ses 10 jetons par
    # semaine, parce qu'un coup de ~12 h bloquait toute autre decision. Avec
    # `positions_max` > 1, le temps avance d'UNE bougie a chaque decision et
    # chaque coup occupe une place jusqu'a sa resolution. Le score que voit la
    # politique, et celui qui juge la fin de partie a -`vie_R`, ne comptent que
    # les coups RESOLUS : les coups encore ouverts ne lui revelent rien de leur
    # issue. 1 = le jeu d'avant, a l'identique.
    # 3 -> 10 LE 2026-09-28 (run h1_03), demande du proprietaire : autant de
    # positions simultanees que de jetons. 10 -> 50 (run h1_04) : une borne
    # technique ; en pratique c'est la marge qui limite. REMIS A 1 LE
    # 2026-09-28 (run kairos_multi_h1_01) : la configuration du run h1_01.
    positions_max: int = 1
    # COMME EN LIVE — le compte demo Vantage, lu dans MT5 le 2026-09-28 :
    # levier 1:500, BTCUSD contrat 1 BTC, lot minimum 0.01, pas 0.01 (0.01
    # lot ~ 842 $ de notionnel pour 1.69 $ de marge). Le jeu n'ouvre un coup
    # que si la marge libre couvre sa marge (au stop le plus serre, le plus
    # gourmand) ; le bilan rejoue les coups sur UN compte, mise de
    # `risque_pct` de l'equite du moment arrondie aux lots de `pas_lot`, jamais
    # sous `lot_min` : avec 1000 $, un stop large risque donc PLUS que 1 %.
    levier: float = 500.0
    lot_min: float = 0.01
    pas_lot: float = 0.01
    contrat: float = 1.0
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
    # EN BOUGIES : 32 bougies M15, huit heures. EN H1 : 72 bougies, trois jours.
    # EN M5 : 96 bougies, huit heures (comme le M15).
    horizon_max: int = 96
    # Glissements ESPERES (la moitie des bornes de `training.PPOConfig`) :
    # entree toujours, sortie au stop et au temps, jamais a l'objectif.
    glissement_entree_bps: float = 0.5
    glissement_sortie_bps: float = 1.0
    capital: float = 1000.0
    # 1 % -> 0.5 % LE 2026-09-27 (run m15_05), demande du proprietaire :
    # reduire le drawdown. Le jeu apprend en R, donc les decisions ne
    # changent pas ; les dollars et le drawdown sont divises par deux, le
    # rapport gain / drawdown est inchange. C'est la mise du live.
    #
    # REMISE A 1 % LE 2026-09-27, demande du proprietaire : retrouver les gains
    # en dollars d'avant. Le drawdown redouble avec eux (~-7 a -15 % en
    # validation, -13.6 % au test du fold 1).
    risque_pct: float = 1.0
    # --- le walk-forward : les memes proportions que le run PPO ---
    part_train: float = 0.55
    part_val: float = 0.15
    part_test: float = 0.10
    n_folds: int = 3
    epochs: int = 40
    rampe_cout: int = 10
    parties_par_epoch: int = 256
    # --- l'expert ---
    # EN H1, 6 coups enseignes par SEMAINE (4 par jour en M15).
    # 6 -> 10 LE 2026-09-28 (run h1_03), demande du proprietaire : l'expert ne
    # jouait qu'un coup a la fois. Il enseigne desormais jusqu'a 10 coups par
    # semaine, autant que les jetons, et peut les faire se chevaucher jusqu'a
    # `positions_max` (voir `coups_expert_predits`). REMIS A 6 LE 2026-09-28
    # (run kairos_multi_h1_01) : la configuration du run h1_01.
    # EN M5 : 4 coups enseignes par JOUR (comme le M15).
    # 4 -> 10 LE 2026-09-28 (run m5_02), demande du proprietaire : plus de
    # trades avec les memes dix jetons. L'expert enseigne jusqu'a 10 coups par
    # jour, autant que les jetons ; le run m5_01 en enseignait 4 et le PPO en
    # jouait 2 a 3.
    expert_k: int = 10
    expert_R_min: float = 1.0          # l'ancien expert, qui lisait l'avenir
    # EN M15, une bougie sur deux : il n'y en a que 96 par jour. EN H1, toutes.
    # EN M5 : une bougie sur trois pour les attentes (288 par jour).
    expert_pas_neg: int = 3
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
    #
    # 0.90 -> 0.85 LE 2026-09-28 (run m5_02), demande du proprietaire : « une
    # selectivite plus large ». `mesure_porte_m5.py`, blocs 1 et 2 du run
    # m5_01, R net moyen des 16 coups du sens joue (cout plein) par tranche de
    # rang, le test NON LU :
    #
    #     rang          validation b1   validation b2   entrainement b1 / b2
    #     0.90 - 0.95   -0.000          +0.022          +0.018 / +0.001
    #     0.85 - 0.90   -0.003          -0.016          -0.008 / -0.005
    #     0.80 - 0.85   -0.014          -0.018          -0.018 / -0.019 (t -2.3)
    #     0.70 - 0.80   -0.021          -0.023          -0.032 / -0.030 (t -5)
    #
    # La tranche 0.85 - 0.90 est a l'equilibre : le PPO peut y choisir. Sous
    # 0.85, elle perd a chaque mesure. La porte s'arrete donc a 0.85.
    porte_rang_expert: float = 0.85
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
    #
    # RETIREE LE 2026-09-27, demande du proprietaire. Mesure au test : +29.40
    # contre +22.97 $ au fold 1, mais -32.30 contre -5.67 $ au fold 2. Avec
    # `mise_en_baisse` a 1.0 la regle ne change plus rien, et le bilan ne
    # l'affiche plus. `compte_prudent` reste pour qui voudrait la remesurer.
    seuil_baisse: float = 0.05
    mise_en_baisse: float = 1.0
    graine: int = 7

    @property
    def risque_dollars(self) -> float:
        return self.capital * self.risque_pct / 100.0

    @property
    def barres_par_jour(self) -> int:
        return 1440 // int(self.minutes_par_barre)

    @property
    def barres_par_partie(self) -> int:
        return self.barres_par_jour * (7 if self.partie == "semaine" else 1)

    @property
    def unite(self) -> str:
        return "semaines" if self.partie == "semaine" else "jours"


def colonnes_jeu(cfg: "JeuConfig") -> list:
    """Les features du cache que joue ce jeu : multi-marches, M15 ou M1."""
    if tuple(getattr(cfg, "marches", ())) and int(cfg.minutes_par_barre) == 60:
        from prepare_multi_h1 import FEATURE_COLS_MULTI_H1
        return list(FEATURE_COLS_MULTI_H1)
    if tuple(getattr(cfg, "marches", ())) and int(cfg.minutes_par_barre) == 5:
        from prepare_multi_m5 import FEATURE_COLS_MULTI_M5
        return list(FEATURE_COLS_MULTI_M5)
    if tuple(getattr(cfg, "marches", ())):
        from prepare_multi_m15 import FEATURE_COLS_MULTI
        return list(FEATURE_COLS_MULTI)
    if int(cfg.minutes_par_barre) == 60:
        from prepare_btc_h1_binance import FEATURE_COLS_H1
        return list(FEATURE_COLS_H1)
    if int(cfg.minutes_par_barre) == 5:
        from prepare_btc_m5 import FEATURE_COLS_M5
        return list(FEATURE_COLS_M5)
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
        # LES TETES PAR MARCHE. Voir `tetes_par_marche`. Chaque copie part
        # des memes poids ; l'entrainement les separe.
        marches = tuple(getattr(cfg, "marches", ()))
        self.n_marches = len(marches) if (marches and getattr(cfg, "tetes_par_marche", False)) else 1
        if self.n_marches > 1:
            cols = colonnes_jeu(cfg)
            self.register_buffer("idx_marche", torch.tensor(
                [cols.index(f"m_{m}") for m in marches], dtype=torch.long), persistent=False)

            def _dup(mod):
                return nn.ModuleList([copy.deepcopy(mod) for _ in range(self.n_marches)])
            self.mlp_achat_m, self.tete_achat_m = _dup(self.mlp_achat), _dup(self.tete_achat)
            self.mlp_vente_m, self.tete_vente_m = _dup(self.mlp_vente), _dup(self.tete_vente)
            self.mlp_m, self.critic_m = _dup(self.mlp), _dup(self.critic)
            self.lecteur_objectif_m, self.tete_objectif_m = (_dup(self.lecteur_objectif),
                                                             _dup(self.tete_objectif))
            self.lecteur_stop_m, self.tete_stop_m = _dup(self.lecteur_stop), _dup(self.tete_stop)

    def marche_de(self, x: torch.Tensor) -> torch.Tensor:
        """(B,) : l'indice du marche de chaque ligne, lu dans ses colonnes
        `m_*` (normalisees : la seule a 1 reste la plus grande)."""
        return x[:, -1, self.idx_marche].argmax(-1)

    @staticmethod
    def _par_marche(k, lecteurs, tetes, z):
        sortie = torch.stack([t(lec(z)) for lec, t in zip(lecteurs, tetes)], 1)
        return sortie[torch.arange(len(k), device=z.device), k]

    def jeu(self, x: torch.Tensor):
        """(logits d'entree (B,3), valeur (B,), objectif (B,2,K), stop (B,2,K)).

        L'axe 1 des barrieres est le sens : 0 achat, 1 vente. Les deux sont
        calcules — les tetes sont petites — pour que le sens tire ensuite
        choisisse les siennes sans second passage dans le tronc.
        """
        zn = self._lecture_tronc(x)
        if self.n_marches > 1:
            k = self.marche_de(x)
            pm = self._par_marche
            la = pm(k, self.mlp_achat_m, self.tete_achat_m, zn)
            lv = pm(k, self.mlp_vente_m, self.tete_vente_m, zn)
            le = torch.cat([la, lv, torch.zeros_like(la)], dim=-1)
            v = pm(k, self.mlp_m, self.critic_m, zn).squeeze(-1)
            un = torch.ones_like(la)
            zl, zs = torch.cat([zn, un], -1), torch.cat([zn, -un], -1)
            ltp = torch.stack([pm(k, self.lecteur_objectif_m, self.tete_objectif_m, zl),
                               pm(k, self.lecteur_objectif_m, self.tete_objectif_m, zs)], 1)
            lsl = torch.stack([pm(k, self.lecteur_stop_m, self.tete_stop_m, zl),
                               pm(k, self.lecteur_stop_m, self.tete_stop_m, zs)], 1)
            return le, v, ltp, lsl
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
        if self.n_marches > 1:
            g = {
                "achat": _p(self.mlp_achat_m, self.tete_achat_m),
                "vente": _p(self.mlp_vente_m, self.tete_vente_m),
                "gain": _p(self.lecteur_objectif_m, self.tete_objectif_m),
                "perte": _p(self.lecteur_stop_m, self.tete_stop_m),
                "valeur": _p(self.mlp_m, self.critic_m),
                "tronc": _p(self.embed, self.col_emb, *self.blocks, self.norm) + [self.cls],
            }
            if self.memoire is not None:
                g["tronc"] += list(self.memoire.parameters())
            return g
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
                duree = np.where(stop, a_sl + 1, np.where(obj, a_tp + 1, H + 1))
                # LE SWAP, au prorata de la duree du coup (en bougies, la
                # bougie d'entree comprise : a peine plus que la detention
                # reelle). Voir `swap_achat_bps_jour`.
                sw = float(getattr(cfg, "swap_achat_bps_jour" if sens > 0
                                   else "swap_vente_bps_jour", 0.0))
                swap = frac * sw / 1e4 * p0 * duree * float(cfg.minutes_par_barre) / 1440.0
                R[t, si, i, j] = ((sens * (x - p0) - swap) / (ksl[j] * a)).astype(np.float32)
                D[t, si, i, j] = duree
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


def semaines(temps: pd.Series, debut: int, fin: int, min_barres: int) -> np.ndarray:
    """Les semaines ENTIERES de [debut, fin), du lundi 0 h au lundi suivant :
    (n, 2) indices [a, b). Meme regle que `journees` pour les trous."""
    sem = pd.to_datetime(temps).dt.to_period("W-SUN").dt.start_time.to_numpy()
    df = pd.DataFrame({"s": sem, "i": np.arange(len(sem))})
    g = df.groupby("s")["i"].agg(["min", "max"])
    g = g[(g["min"] >= debut) & (g["max"] < fin)]
    g = g[(g["max"] - g["min"]) >= min_barres]
    return np.stack([g["min"].to_numpy(), g["max"].to_numpy() + 1], 1)


def parties(temps: pd.Series, debut: int, fin: int, cfg) -> np.ndarray:
    """Les parties de [debut, fin) selon `cfg.partie`, avec au moins 40 % de
    leurs bougies."""
    mb = int(0.4 * cfg.barres_par_partie)
    if getattr(cfg, "partie", "jour") == "semaine":
        return semaines(temps, debut, fin, mb)
    return journees(temps, debut, fin, mb)


def etat_jeu(jetons, score, reste_min, cfg: JeuConfig, occupees=None) -> np.ndarray:
    """Les cinq colonnes du bloc de position, qui portent l'etat de la partie.
    La cinquieme : la part des places occupees (`positions_max`), 0 sinon."""
    n = len(jetons)
    return np.stack([
        jetons / float(cfg.jetons),
        np.clip(score / cfg.vie_R, -1.0, 3.0),
        np.clip(reste_min / float(cfg.barres_par_partie), 0.0, 1.0),
        np.clip((score + cfg.vie_R) / cfg.vie_R, 0.0, 3.0),
        np.zeros(n) if occupees is None else np.asarray(occupees, np.float64),
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


def fraction_marge(close, atr, cfg: JeuConfig) -> np.ndarray:
    """(N, K_sl) : la marge d'un coup en fraction de l'equite, pour chaque
    stop : mise de `risque_pct` au stop, notionnel = mise / distance, marge =
    notionnel / levier."""
    dist = (np.asarray(cfg.sl_atr, np.float64)[None, :] * np.asarray(atr, np.float64)[:, None]
            / np.asarray(close, np.float64)[:, None])
    return (cfg.risque_pct / 100.0) / np.maximum(dist, 1e-12) / float(cfg.levier)


def joue(policy, jours: np.ndarray, Xn, R, D, S, fin_valide: int,
         cfg: JeuConfig, device, explore: bool, gen=None,
         collecte: bool = False, rangs=None, marge=None):
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
    # PLUSIEURS POSITIONS : chaque place garde la fin et le R de son coup,
    # qui n'entre dans le score VU qu'une fois resolu. Voir `positions_max`.
    P = int(getattr(cfg, "positions_max", 1))
    multi = P > 1
    occ_fin = np.full((G, max(P, 1)), -1, np.int64)
    occ_r = np.zeros((G, max(P, 1)))
    # LA MARGE (positions multiples) : celle de chaque coup ouvert, en
    # fraction de l'equite. `marge` None = pas de contrainte (les tests).
    occ_m = np.zeros((G, max(P, 1)))
    m_max = None if marge is None else np.asarray(marge).max(1)
    vu = np.zeros(G, np.float64)
    policy.eval()
    while actif.any():
        g = np.flatnonzero(actif)
        tt = t[g]
        if multi:
            fait = (occ_fin[g] >= 0) & (occ_fin[g] <= tt[:, None])
            vu[g] += (occ_r[g] * fait).sum(1)
            occ_r[g] = np.where(fait, 0.0, occ_r[g])
            occ_m[g] = np.where(fait, 0.0, occ_m[g])
            occ_fin[g] = np.where(fait, -1, occ_fin[g])
            libre = occ_fin[g] < 0
            n_occ = P - libre.sum(1)
            et = etat_jeu(jet[g], vu[g], fin[g] - tt, cfg, n_occ / float(P))
        else:
            et = etat_jeu(jet[g], score[g], fin[g] - tt, cfg)
        ob = observations(Xn, tt, et, L)
        peut_np = (jet[g] > 0) & (tt + 1 + H < _lim(fin_valide, tt))
        if multi:
            peut_np = peut_np & (n_occ < P)
            if m_max is not None:
                peut_np = peut_np & (1.0 - occ_m[g].sum(1) >= m_max[tt])
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
                if multi:
                    k = int(np.argmax(libre[q]))
                    occ_fin[g[q], k] = int(tt[q] + dur[q])
                    occ_r[g[q], k] = float(r[q])
                    if marge is not None:
                        occ_m[g[q], k] = float(marge[tt[q], j[q]])
        # En positions multiples, le temps avance d'une bougie ; la decision
        # dure donc une bougie pour l'actualisation du PPO.
        pas = np.ones(len(g), np.int64) if multi else dur
        t[g] = tt + pas
        score[g] += r
        jet[g] -= coup.astype(np.int64)
        juge = vu[g] if multi else score[g]
        fini = (t[g] >= fin[g]) | (jet[g] <= 0) | (juge <= -cfg.vie_R)
        actif[g[fini]] = False
        if collecte:
            for q in range(len(g)):
                # LES DEUX PORTES, codees : bit 0 achat, bit 1 vente.
                trans[g[q]].append((int(tt[q]), et[q],
                                    int(pa_np[q]) + 2 * int(pv_np[q]),
                                    int(a[q]), int(i[q]), int(j[q]),
                                    float(lp_e[q]), float(lp_i[q]),
                                    float(lp_j[q]), float(v[q]), float(r[q]),
                                    int(pas[q]), bool(fini[q])))
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


def journees_multi(temps: pd.Series, blocs, t0_ns, t1_ns, min_barres: int,
                   cfg=None) -> np.ndarray:
    """Les parties entieres de [t0, t1), marche par marche : des journees, ou
    celles de `cfg.partie` si `cfg` est donne."""
    t_ns = temps.values.astype("int64")
    out = []
    for _, a, b in blocs:
        i0 = a + int(np.searchsorted(t_ns[a:b], t0_ns, side="left"))
        i1 = a + int(np.searchsorted(t_ns[a:b], t1_ns, side="left"))
        if i1 - i0 < min_barres:
            continue
        tb = temps.iloc[a:b].reset_index(drop=True)
        j = (parties(tb, i0 - a, i1 - a, cfg) if cfg is not None
             else journees(tb, i0 - a, i1 - a, min_barres))
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

    Au plus `expert_k`, seulement si le R net predit est positif, un seul
    par bougie, et jamais plus de `positions_max` ouverts a la fois (1 : sans
    chevauchement, comme avant). Meme format que `coups_expert`.
    """
    P = max(int(getattr(cfg, "positions_max", 1)), 1)
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
        occupe = np.zeros(b0 - a0 + H + 2, np.int64)
        pris = 0
        for q in np.argsort(-v, kind="stable"):
            if v[q] <= 0.0 or pris >= cfg.expert_k:
                break
            t0 = int(ts[q])
            d0 = int(D[t0, sens[q], ri, rj])
            if occupe[t0 - a0: t0 - a0 + d0 + 1].max() >= P:
                continue
            occupe[t0 - a0: t0 - a0 + d0 + 1] += 1
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
    # EN POSITIONS MULTIPLES, les coups se chevauchent : le compte bouge a
    # leur RESOLUTION, pas a leur ouverture.
    t_ref = t + du.astype(np.int64) if int(getattr(cfg, "positions_max", 1)) > 1 else t
    if ordre is not None:
        t_ref = np.minimum(t_ref, len(ordre) - 1)
    ordre = np.argsort(t_ref if ordre is None else np.asarray(ordre)[t_ref], kind="stable")
    eq = cfg.capital + np.cumsum(r[ordre] * cfg.risque_dollars)
    eq = np.concatenate([[cfg.capital], eq])
    pic = np.maximum.accumulate(eq)
    dd_d = float((eq - pic).min())
    dd_p = float(((eq - pic) / pic).min())
    # LA MISE REDUITE EN BAISSE, jouee dans le meme ordre. Voir `seuil_baisse`.
    ep = compte_prudent(r[ordre], cfg)
    pic_p = np.maximum.accumulate(ep)
    # UN SEUL MARCHE : les lots, contrats et devises des indices different de
    # ceux du BTC, le compte « comme en live » ne vaut que pour le BTC seul.
    live = (compte_live(t, j, r, du, close, atr, cfg)
            if n and not tuple(getattr(cfg, "marches", ())) else {})
    return b | live | {
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


def compte_live(t, j, r, du, close, atr, cfg: JeuConfig) -> Dict[str, float]:
    """Les coups rejoues sur UN compte, comme en live.

    Chaque coup mise `risque_pct` de l'equite REALISEE au moment d'ouvrir,
    en lots arrondis vers le bas au `pas_lot`, jamais sous `lot_min` ; il
    n'est pris que si la marge libre (equite - marge des coups ouverts)
    couvre la sienne. Son resultat en dollars est son R fois le risque
    reellement pris (lots x contrat x distance du stop). Les pertes latentes
    des coups ouverts ne sont pas comptees dans la marge libre.
    """
    import heapq
    ksl = np.asarray(cfg.sl_atr, np.float64)
    E = float(cfg.capital)
    courbe, ouverts, risques = [E], [], []
    pris, sautes = 0, 0
    for k in np.argsort(t, kind="stable"):
        tk = int(t[k])
        while ouverts and ouverts[0][0] <= tk:
            _, _, pnl = heapq.heappop(ouverts)
            E += pnl
            courbe.append(E)
        if E <= 0:
            break
        dist = ksl[int(j[k])] * float(atr[tk])
        cible = E * cfg.risque_pct / 100.0
        lots = np.floor(cible / (dist * cfg.contrat) / cfg.pas_lot + 1e-9) * cfg.pas_lot
        lots = max(float(lots), float(cfg.lot_min))
        m = lots * cfg.contrat * float(close[tk]) / float(cfg.levier)
        if m > E - sum(o[1] for o in ouverts):
            sautes += 1
            continue
        risque = lots * cfg.contrat * dist
        risques.append(risque / E)
        heapq.heappush(ouverts, (tk + int(du[k]), m, float(r[k]) * risque))
        pris += 1
    while ouverts:
        _, _, pnl = heapq.heappop(ouverts)
        E += pnl
        courbe.append(E)
    courbe = np.asarray(courbe)
    pic = np.maximum.accumulate(courbe)
    return {"live_total": float(E - cfg.capital),
            "live_dd_dollars": float((courbe - pic).min()),
            "live_dd_pct": float(((courbe - pic) / pic).min()),
            "live_pris": pris, "live_marge": sautes,
            "live_risque_moy": float(np.mean(risques)) if risques else float("nan"),
            "live_risque_max": float(np.max(risques)) if risques else float("nan")}


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
            f"{b['total_dollars']:+.2f} $ sur {b['parties']} {cfg.unite}"
            + (f"  |  COMME EN LIVE ({cfg.risque_pct:g} % de l'equite, lots de "
               f"{cfg.lot_min:g}, levier 1:{cfg.levier:.0f}) : total {b['live_total']:+.2f} $, "
               f"pire baisse {100 * b['live_dd_pct']:+.1f} %, risque reel moyen "
               f"{100 * b['live_risque_moy']:.2f} % (max {100 * b['live_risque_max']:.1f} %)"
               + (f", {b['live_marge']} refuses faute de marge" if b.get("live_marge") else "")
               if "live_total" in b else ""))


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
def masque_blocs(N: int, K: int, k: int, purge: int):
    """(entrainement permis (N,), (debut, fin) du test, (debut, fin) de la
    validation) pour le bloc de test `k` : la validation est le bloc le plus
    eloigne ; `purge` bougies sont retirees de part et d'autre des deux."""
    b = np.linspace(0, N, K + 1).astype(np.int64)
    te = (int(b[k]), int(b[k + 1]))
    v = (k + K // 2) % K
    va = (int(b[v]), int(b[v + 1]))
    exclu = np.zeros(N, bool)
    for a, z in (te, va):
        exclu[max(0, a - purge):min(N, z + purge)] = True
    return ~exclu, te, va


def prochain_exclu(permis: np.ndarray) -> np.ndarray:
    """Pour chaque bougie, l'indice de la premiere bougie NON permise a partir
    d'elle (N s'il n'y en a pas) : un coup d'entrainement doit se resoudre
    avant, pour ne jamais lire la zone de test ou de validation."""
    N = len(permis)
    out = np.full(N, N, np.int64)
    prochain = N
    for i in range(N - 1, -1, -1):
        if not permis[i]:
            prochain = i
        out[i] = prochain
    return out


def main_blocs(cfg: JeuConfig) -> int:
    """La validation croisee purgee. Voir `validation`."""
    rng = np.random.default_rng(cfg.graine)
    torch.manual_seed(cfg.graine)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifeste = f"run_{cfg.prefixe}.json"
    if os.path.exists(manifeste):
        raise FileExistsError(f"{manifeste} existe : nouveau prefixe, ou lancer.ps1")
    with open(manifeste, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=1, default=str)
    print("=" * 70)
    print(f"  KAIROS EN JEU — BTCUSD M{cfg.minutes_par_barre} : une "
          f"{'semaine' if cfg.partie == 'semaine' else 'journee'} = une partie  |  "
          f"VALIDATION CROISEE PURGEE en {cfg.n_blocs} blocs")
    print("=" * 70)
    print(f"  regles : {cfg.jetons} coups par partie, {cfg.positions_max} position(s) a la fois, "
          f"fin de partie a -{cfg.vie_R:g} R, temps limite {cfg.horizon_max} bougies  |  un R = "
          f"{cfg.risque_dollars:.0f}$  |  {device}", flush=True)
    d = pd.read_pickle(cfg.cache)
    cols = list(dict.fromkeys(["time", "open", "high", "low", "close", "atr_14",
                               "spread_bar"] + colonnes_jeu(cfg)))
    d = d[cols].reset_index(drop=True)
    N = len(d)
    o, h, l, c = (d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    atr = atr_effectif(d["atr_14"].to_numpy(np.float64), c, cfg)
    sp = d["spread_bar"].to_numpy(np.float64)
    t_ns = d["time"].values.astype("int64")
    X = d[colonnes_jeu(cfg)].to_numpy(np.float32)
    marge = fraction_marge(c, atr, cfg)
    R0, D0, _ = table_coups(o, h, l, sp, atr, cfg, 0.0)
    R1, D1, S1 = table_coups(o, h, l, sp, atr, cfg, 1.0)
    y_ex = cibles_expert(R1)
    toutes = parties(d["time"], 0, N, cfg)
    purge = int(cfg.horizon_max + cfg.lookback + cfg.purge_semaines * cfg.barres_par_partie)
    _fmt = lambda i: pd.Timestamp(d["time"].iloc[min(i, N - 1)]).strftime("%Y-%m-%d")
    print(f"[CACHE] {N:,} bougies, {_fmt(0)} -> {_fmt(N - 1)} ; {len(toutes)} parties ; "
          f"zone tampon {purge} bougies autour du test et de la validation", flush=True)
    tests = []
    for k in range(cfg.n_blocs):
        permis, (te0, te1), (va0, va1) = masque_blocs(N, cfg.n_blocs, k, purge)
        fin_tr = prochain_exclu(permis)
        dedans = lambda a0, a1: np.array([w for w in toutes if w[0] >= a0 and w[1] <= a1],
                                         np.int64).reshape(-1, 2)
        j_tr = np.array([w for w in toutes if permis[w[0]:w[1]].all()], np.int64).reshape(-1, 2)
        j_va, j_te = dedans(va0, va1), dedans(te0, te1)
        print(f"\n--- Fold {k + 1} : test {_fmt(te0)} -> {_fmt(te1 - 1)} ({len(j_te)} parties)  "
              f"validation {_fmt(va0)} -> {_fmt(va1 - 1)} ({len(j_va)} parties)  "
              f"entrainement sur tout le reste ({len(j_tr)} parties) ---", flush=True)
        st_m = X[permis].astype(np.float64).mean(0)
        st_s = X[permis].astype(np.float64).std(0)
        Xn = safe_normalize(X, {"mean": st_m.astype(np.float32),
                                "std": st_s.astype(np.float32)}).astype(np.float32)
        t_ex = time.time()
        pred, _ = expert_multi(Xn, y_ex, t_ns, permis, fin_tr, cfg)
        FE = features_expert(pred, fenetre=10_000 // int(cfg.minutes_par_barre))
        rangs_ex = FE[:, 2:4].copy()
        m_e = FE[permis].astype(np.float64).mean(0)
        s_e = FE[permis].astype(np.float64).std(0)
        FEn = np.clip((FE - m_e) / (s_e + 1e-8), -5.0, 5.0).astype(np.float32)
        Xk = np.concatenate([Xn, FEn[:, :cfg.n_expert]], axis=1)
        ics = []
        for s_ in range(2):
            ok = np.zeros(N, bool)
            ok[te0:te1] = True
            ok &= np.isfinite(y_ex[:, s_])
            ics.append(pd.Series(pred[ok, s_]).corr(pd.Series(y_ex[ok, s_]), method="spearman"))
        print(f"  expert  LightGBM appris sur les autres blocs, {time.time() - t_ex:.0f} s ; "
              f"correlation sur le bloc de test : achat {ics[0]:+.3f}, vente {ics[1]:+.3f}",
              flush=True)
        policy = PolitiqueJeu(cfg).to(device)
        optims = optimiseurs(policy, cfg)
        pos = coups_expert_predits(j_tr, pred, D1, fin_tr, cfg)
        imite_expert(policy, optims, pos, j_tr, Xk, fin_tr, cfg, device, rng)
        suffixe = f"_bloc{k + 1}"
        best_path = f"best_{cfg.prefixe}{suffixe}.pth"
        record, garde_rec = 0.0, None
        for epoch in range(0, cfg.epochs + 1):
            t_ep = time.time()
            frac = min(1.0, max(0.0, (epoch - 1) / cfg.rampe_cout)) if cfg.rampe_cout > 0 else 1.0
            st = None
            if epoch >= 1:
                Rf = (1 - frac) * R0 + frac * R1 if frac < 1.0 else R1
                Df = D1 if frac >= 0.5 else D0
                choix = rng.choice(len(j_tr), size=min(cfg.parties_par_epoch, len(j_tr)),
                                   replace=False)
                marge_d = max(240 // int(cfg.minutes_par_barre), cfg.barres_par_partie // 6)
                sc, cp, tr = joue(policy, departs_tires(j_tr[choix], rng, marge_d), Xk,
                                  Rf, Df, S1, fin_tr, cfg, device, explore=True,
                                  collecte=True, rangs=rangs_ex, marge=marge)
                st = maj_ppo(policy, optims, avantages(tr, cfg), Xk, cfg, device, rng)
                del Rf
            gen = torch.Generator(device=device)
            gen.manual_seed(cfg.graine)
            sv, cv, _ = joue(policy, j_va, Xk, R1, D1, S1, va1, cfg, device,
                             explore=False, gen=gen, rangs=rangs_ex, marge=marge)
            bv = bilan(sv, cv, c, atr, sp, cfg)
            nom = "EXPERT IMITE" if epoch == 0 else f"cout {100 * frac:.0f}%"
            print(f"\nEPOCH {epoch:03d}  {nom:>12}  VAL  {ligne_bilan(bv, cfg)}  "
                  f"{(time.time() - t_ep) / 60:.1f} min", flush=True)
            print(f"  bilan  {ligne_detail(bv, cfg)}", flush=True)
            etat = {"modele": policy.state_dict(), "config": asdict(cfg), "epoch": epoch,
                    "bloc": k + 1}
            torch.save(etat, f"last_{cfg.prefixe}{suffixe}.pth")
            if bv["coups"] >= cfg.min_coups_val and bv["score"] > record:
                record, garde_rec = bv["score"], epoch
                torch.save(etat, best_path)
                print(f"  sauvegarde  NOUVEAU MEILLEUR : {bv['score']:+.3f} R/partie en "
                      f"validation, {bv['coups']} coups -> {best_path}", flush=True)
            else:
                print(f"  garde  rien de sauvegarde : {bv['score']:+.3f} R/partie ne bat pas "
                      f"{record:+.3f}" + (f" (meilleur : epoch {garde_rec})"
                                          if garde_rec is not None else ""), flush=True)
        src = best_path if os.path.exists(best_path) else f"last_{cfg.prefixe}{suffixe}.pth"
        policy.load_state_dict(torch.load(src, map_location=device, weights_only=False)["modele"])
        gen = torch.Generator(device=device)
        gen.manual_seed(cfg.graine)
        s_t, c_t, _ = joue(policy, j_te, Xk, R1, D1, S1, te1, cfg, device,
                           explore=False, gen=gen, rangs=rangs_ex, marge=marge)
        bt = bilan(s_t, c_t, c, atr, sp, cfg)
        tests.append((k + 1, _fmt(te0), _fmt(te1 - 1), bt, c_t))
        print(f"\nTEST fold {k + 1} ({os.path.basename(src)})  {ligne_bilan(bt, cfg)}", flush=True)
        print(f"TEST fold {k + 1} bilan  {ligne_detail(bt, cfg)}", flush=True)
        with open(f"test_{cfg.prefixe}{suffixe}.json", "w", encoding="utf-8") as fh:
            json.dump({k_: v_ for k_, v_ in bt.items()}, fh, indent=1, default=str)
    print("\n" + "=" * 70)
    print(f"  RESUME DES {cfg.n_blocs} BLOCS DE TEST (chacun jamais vu par son modele)")
    print("=" * 70)
    tous = []
    for k, a, z, bt, c_t in tests:
        tous += list(c_t)
        tot = bt.get("total_dollars", 0.0)
        print(f"  bloc {k:2d}  {a} -> {z}  {bt['coups']:4d} trades  PF "
              f"{bt.get('pf', float('nan')):5.2f}  total {tot:+9.2f} $", flush=True)
    positifs = sum(1 for *_, bt, _c in tests if bt.get("total_dollars", 0.0) > 0)
    somme = sum(bt.get("total_dollars", 0.0) for *_, bt, _c in tests)
    r = np.array([x[5] for x in tous]) if tous else np.zeros(0)
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else float("inf")
    print(f"  blocs gagnants : {positifs}/{cfg.n_blocs}  |  total {somme:+.2f} $  |  "
          f"{len(r)} trades  |  profit factor global {pf:.2f}", flush=True)
    print("\nFIN du walk-forward", flush=True)
    return 0


def masque_blocs_dates(t_ns: np.ndarray, t_ref: np.ndarray, K: int, k: int,
                       purge_ns: int):
    """`masque_blocs` EN DATES, pour plusieurs marches empiles.

    Les bornes des blocs sont celles du marche de reference (`t_ref`, ses
    instants dans l'ordre) : pour le BTC, les memes blocs que son jeu seul.
    Rend (entrainement permis (N,), (debut, fin) du test, (debut, fin) de la
    validation), les bornes en ns, la fin exclue."""
    n = len(t_ref)
    ib = np.linspace(0, n, K + 1).astype(np.int64)
    tb = [int(t_ref[i]) for i in ib[:-1]] + [int(t_ref[-1]) + 1]
    te = (tb[k], tb[k + 1])
    v = (k + K // 2) % K
    va = (tb[v], tb[v + 1])
    exclu = np.zeros(len(t_ns), bool)
    for a, z in (te, va):
        exclu |= (t_ns >= a - purge_ns) & (t_ns < z + purge_ns)
    return ~exclu, te, va


def main_multi_blocs(cfg: JeuConfig) -> int:
    """Le jeu multi-marches en validation croisee purgee. Voir `validation`
    et `marches`.

    LES BLOCS SONT DES DATES, celles du BTC seul (`main_blocs`) : chaque
    bloc de test est joue par un modele NEUF, entraine sur tous les autres
    blocs de tous les marches. Un expert par marche, appris sur SES bougies ;
    un tronc SAINT commun et des tetes par marche (`tetes_par_marche`).
    Chaque marche joue sa journee avec ses jetons ; le resultat du jour est
    leur somme. L'ATR des barrieres ne descend jamais sous `atr_min_bps`
    (comme le BTC seul), ni sous `plancher_couts` fois le cout median du
    marche : le cout reste sous ~0.2 R partout."""
    rng = np.random.default_rng(cfg.graine)
    torch.manual_seed(cfg.graine)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifeste = f"run_{cfg.prefixe}.json"
    if os.path.exists(manifeste):
        raise FileExistsError(f"{manifeste} existe : nouveau prefixe, ou lancer.ps1")
    with open(manifeste, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=1, default=str)
    cols = colonnes_jeu(cfg)
    _u = "semaine" if cfg.partie == "semaine" else "journee"
    print("=" * 70)
    print(f"  KAIROS EN JEU — {len(cfg.marches)} MARCHES M{cfg.minutes_par_barre} : une {_u} = "
          f"une partie par marche  |  VALIDATION CROISEE PURGEE en {cfg.n_blocs} blocs")
    print("=" * 70)
    print(f"  marches : {', '.join(cfg.marches)}  |  un petit cerveau par marche : "
          f"{'oui' if cfg.tetes_par_marche else 'non'}")
    print(f"  regles : {cfg.jetons} coups par marche et par {'semaine' if cfg.partie == 'semaine' else 'jour'}, "
          f"{cfg.positions_max} position(s) a la fois par marche, fin de partie a -{cfg.vie_R:g} R, "
          f"temps limite {cfg.horizon_max} bougies, porte {cfg.porte_rang_expert:g}, expert "
          f"{cfg.expert_k} coups  |  un R = {cfg.risque_dollars:.0f}$  |  {device}", flush=True)
    d = pd.read_pickle(cfg.cache)
    d = d[d["marche"].isin(cfg.marches)].copy()
    d["_o"] = d["marche"].map({m: i for i, m in enumerate(cfg.marches)})
    d = d.sort_values(["_o", "time"], kind="stable").drop(columns="_o")
    _plus = [k for k in ["swap_achat_bps_jour", "swap_vente_bps_jour"] if k in d.columns]
    d = d[list(dict.fromkeys(["time", "marche", "open", "high", "low", "close",
                              "atr_14", "spread_bar"] + cols + _plus))].reset_index(drop=True)
    N = len(d)
    marche = d["marche"].to_numpy()
    blocs = blocs_marches(marche)
    t_ns = d["time"].values.astype("int64")
    dates = d["time"].dt.normalize().values
    o, h, l, c = (d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    sp = d["spread_bar"].to_numpy(np.float64)
    plancher = np.empty(N)
    for nom, a, b in blocs:
        cout_med = float(np.median(sp[a:b])) + cfg.glissement_entree_bps + cfg.glissement_sortie_bps
        plancher[a:b] = max(float(cfg.atr_min_bps), cfg.plancher_couts * cout_med)
        sw = (f"swap {d['swap_achat_bps_jour'].iloc[a]:.2f} / {d['swap_vente_bps_jour'].iloc[a]:.2f} bps/jour"
              if _plus else "")
        print(f"  {nom:8s} {b - a:9,d} bougies  {d['time'].iloc[a]:%Y-%m-%d} -> "
              f"{d['time'].iloc[b - 1]:%Y-%m-%d}  cout med {cout_med:5.2f} bps  "
              f"ATR des barrieres >= {plancher[a]:5.1f} bps  {sw}", flush=True)
    atr = np.maximum(d["atr_14"].to_numpy(np.float64), plancher * 1e-4 * c)
    X = d[cols].to_numpy(np.float32)

    def _cfg_m(a):
        if not _plus:
            return cfg
        return replace(cfg, swap_achat_bps_jour=float(d["swap_achat_bps_jour"].iloc[a]),
                       swap_vente_bps_jour=float(d["swap_vente_bps_jour"].iloc[a]))
    t_tab = time.time()
    tabs = {}
    for frac in (0.0, 1.0):
        parts = [table_coups(o[a:b], h[a:b], l[a:b], sp[a:b], atr[a:b], _cfg_m(a), frac)
                 for _, a, b in blocs]
        tabs[frac] = tuple(np.concatenate([p_[k] for p_ in parts]) for k in range(3))
        del parts
    R0, D0, _ = tabs[0.0]
    R1, D1, S1 = tabs[1.0]
    del tabs
    y_ex = cibles_expert(R1)
    print(f"  table des coups : {N:,} bougies x {R1[0].size} coups, marche par marche, "
          f"{time.time() - t_tab:.0f} s", flush=True)
    toutes = journees_multi(d["time"], blocs, int(t_ns.min()), int(t_ns.max()) + 1,
                            int(0.4 * cfg.barres_par_partie), cfg)
    purge = int(cfg.horizon_max + cfg.lookback + cfg.purge_semaines * cfg.barres_par_partie)
    purge_ns = purge * int(cfg.minutes_par_barre) * 60 * 10**9
    _, a_ref, b_ref = blocs[0]
    _fmt = lambda x: pd.Timestamp(x).strftime("%Y-%m-%d")
    print(f"[CACHE] {N:,} bougies ; {len(toutes)} parties ; blocs en dates, ceux du "
          f"{blocs[0][0]} ; zone tampon {purge} bougies autour du test et de la validation",
          flush=True)
    tests = []
    for k in range(cfg.n_blocs):
        permis, (te0, te1), (va0, va1) = masque_blocs_dates(t_ns, t_ns[a_ref:b_ref],
                                                            cfg.n_blocs, k, purge_ns)
        fin_tr = np.empty(N, np.int64)
        for _, a, b in blocs:
            fin_tr[a:b] = a + prochain_exclu(permis[a:b])
        dedans = lambda z0, z1: np.array(
            [w for w in toutes if t_ns[w[0]] >= z0 and t_ns[w[1] - 1] < z1],
            np.int64).reshape(-1, 2)
        j_tr = np.array([w for w in toutes if permis[w[0]:w[1]].all()], np.int64).reshape(-1, 2)
        j_va, j_te = dedans(va0, va1), dedans(te0, te1)
        fin_va = fin_segment(t_ns, blocs, va1)
        fin_te = fin_segment(t_ns, blocs, te1)
        par_m = lambda j: " ".join(f"{nom} {int(((j[:, 0] >= a) & (j[:, 0] < b)).sum())}"
                                   for nom, a, b in blocs)
        print(f"\n--- Fold {k + 1} : test {_fmt(te0)} -> {_fmt(te1 - 1)} ({len(j_te)} parties : "
              f"{par_m(j_te)})  validation {_fmt(va0)} -> {_fmt(va1 - 1)} ({len(j_va)} parties)  "
              f"entrainement sur tout le reste ({len(j_tr)} parties) ---", flush=True)
        st_m = X[permis].astype(np.float64).mean(0)
        st_s = X[permis].astype(np.float64).std(0)
        Xn = safe_normalize(X, {"mean": st_m.astype(np.float32),
                                "std": st_s.astype(np.float32)}).astype(np.float32)
        t_ex = time.time()
        pred = np.full((N, 2), np.nan, np.float32)
        ics = []
        for nom, a, b in blocs:
            pb, _ = expert_multi(Xn[a:b], y_ex[a:b], t_ns[a:b], permis[a:b], fin_tr[a:b] - a, cfg)
            pred[a:b] = pb
            ic = []
            for s_ in range(2):
                ok = (t_ns[a:b] >= te0) & (t_ns[a:b] < te1) & np.isfinite(y_ex[a:b, s_])
                ic.append(pd.Series(pb[ok, s_]).corr(pd.Series(y_ex[a:b][ok, s_]), method="spearman")
                          if ok.sum() > 100 else float("nan"))
            ics.append(f"{nom} {ic[0]:+.3f}/{ic[1]:+.3f}")
        print(f"  expert  un LightGBM par marche, appris sur les autres blocs, "
              f"{time.time() - t_ex:.0f} s ; correlation sur le bloc de test (achat/vente) : "
              + "  ".join(ics), flush=True)
        FE = np.concatenate([features_expert(pred[a:b], fenetre=10_000 // int(cfg.minutes_par_barre))
                             for _, a, b in blocs])
        rangs_ex = FE[:, 2:4].copy()
        m_e = FE[permis].astype(np.float64).mean(0)
        s_e = FE[permis].astype(np.float64).std(0)
        FEn = np.clip((FE - m_e) / (s_e + 1e-8), -5.0, 5.0).astype(np.float32)
        Xk = np.concatenate([Xn, FEn[:, :cfg.n_expert]], axis=1)
        del Xn, FE, FEn
        policy = PolitiqueJeu(cfg).to(device)
        optims = optimiseurs(policy, cfg)
        pos = coups_expert_predits(j_tr, pred, D1, fin_tr, cfg)
        imite_expert(policy, optims, pos, j_tr, Xk, fin_tr, cfg, device, rng)

        def _bilan(scores, jours, coups, frac=1.0):
            return bilan(par_jour(scores, jours, dates), coups, c, atr, sp, cfg,
                         frac=frac, ordre=t_ns)
        suffixe = f"_bloc{k + 1}"
        best_path = f"best_{cfg.prefixe}{suffixe}.pth"
        record, garde_rec = 0.0, None
        for epoch in range(0, cfg.epochs + 1):
            t_ep = time.time()
            frac = min(1.0, max(0.0, (epoch - 1) / cfg.rampe_cout)) if cfg.rampe_cout > 0 else 1.0
            if epoch >= 1:
                Rf = (1 - frac) * R0 + frac * R1 if frac < 1.0 else R1
                Df = D1 if frac >= 0.5 else D0
                choix = rng.choice(len(j_tr), size=min(cfg.parties_par_epoch, len(j_tr)),
                                   replace=False)
                marge_d = max(240 // int(cfg.minutes_par_barre), cfg.barres_par_partie // 6)
                sc, cp, tr = joue(policy, departs_tires(j_tr[choix], rng, marge_d), Xk,
                                  Rf, Df, S1, fin_tr, cfg, device, explore=True,
                                  collecte=True, rangs=rangs_ex)
                maj_ppo(policy, optims, avantages(tr, cfg), Xk, cfg, device, rng)
                del Rf
            gen = torch.Generator(device=device)
            gen.manual_seed(cfg.graine)
            sv, cv, _ = joue(policy, j_va, Xk, R1, D1, S1, fin_va, cfg, device,
                             explore=False, gen=gen, rangs=rangs_ex)
            bv = _bilan(sv, j_va, cv)
            nom = "EXPERT IMITE" if epoch == 0 else f"cout {100 * frac:.0f}%"
            print(f"\nEPOCH {epoch:03d}  {nom:>12}  VAL  {ligne_bilan(bv, cfg)}  "
                  f"{(time.time() - t_ep) / 60:.1f} min", flush=True)
            print(f"  bilan  {ligne_detail(bv, cfg)}", flush=True)
            print(f"  bilan  {ligne_marches(cv, marche, cfg)}", flush=True)
            etat = {"modele": policy.state_dict(), "config": asdict(cfg), "epoch": epoch,
                    "bloc": k + 1}
            torch.save(etat, f"last_{cfg.prefixe}{suffixe}.pth")
            if bv["coups"] >= cfg.min_coups_val and bv["score"] > record:
                record, garde_rec = bv["score"], epoch
                torch.save(etat, best_path)
                print(f"  sauvegarde  NOUVEAU MEILLEUR : {bv['score']:+.3f} R/partie en "
                      f"validation, {bv['coups']} coups -> {best_path}", flush=True)
            else:
                print(f"  garde  rien de sauvegarde : {bv['score']:+.3f} R/partie ne bat pas "
                      f"{record:+.3f}" + (f" (meilleur : epoch {garde_rec})"
                                          if garde_rec is not None else ""), flush=True)
        src = best_path if os.path.exists(best_path) else f"last_{cfg.prefixe}{suffixe}.pth"
        policy.load_state_dict(torch.load(src, map_location=device, weights_only=False)["modele"])
        gen = torch.Generator(device=device)
        gen.manual_seed(cfg.graine)
        s_t, c_t, _ = joue(policy, j_te, Xk, R1, D1, S1, fin_te, cfg, device,
                           explore=False, gen=gen, rangs=rangs_ex)
        bt = _bilan(s_t, j_te, c_t)
        tests.append((k + 1, _fmt(te0), _fmt(te1 - 1), bt, c_t))
        print(f"\nTEST fold {k + 1} ({os.path.basename(src)})  {ligne_bilan(bt, cfg)}", flush=True)
        print(f"TEST fold {k + 1} bilan  {ligne_detail(bt, cfg)}", flush=True)
        print(f"TEST fold {k + 1} bilan  {ligne_marches(c_t, marche, cfg)}", flush=True)
        with open(f"test_{cfg.prefixe}{suffixe}.json", "w", encoding="utf-8") as fh:
            json.dump({k_: v_ for k_, v_ in bt.items()}, fh, indent=1, default=str)
        del Xk, pred, rangs_ex
    print("\n" + "=" * 70)
    print(f"  RESUME DES {cfg.n_blocs} BLOCS DE TEST (chacun jamais vu par son modele)")
    print("=" * 70)
    tous = []
    for k, a, z, bt, c_t in tests:
        tous += list(c_t)
        tot = bt.get("total_dollars", 0.0)
        print(f"  bloc {k:2d}  {a} -> {z}  {bt['coups']:5d} trades  PF "
              f"{bt.get('pf', float('nan')):5.2f}  total {tot:+9.2f} $  |  "
              f"{ligne_marches(c_t, marche, cfg)}", flush=True)
    positifs = sum(1 for *_, bt, _c in tests if bt.get("total_dollars", 0.0) > 0)
    somme = sum(bt.get("total_dollars", 0.0) for *_, bt, _c in tests)
    r = np.array([x[5] for x in tous]) if tous else np.zeros(0)
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else float("inf")
    print(f"  blocs gagnants : {positifs}/{cfg.n_blocs}  |  total {somme:+.2f} $  |  "
          f"{len(r)} trades  |  profit factor global {pf:.2f}", flush=True)
    print(f"  tous les blocs, {ligne_marches(tous, marche, cfg)}", flush=True)
    print("\nFIN du walk-forward", flush=True)
    return 0


def main() -> int:
    cfg = JeuConfig()
    if tuple(cfg.marches) and getattr(cfg, "validation", "walk") == "blocs":
        return main_multi_blocs(cfg)
    if tuple(cfg.marches):
        return main_multi(cfg)
    if getattr(cfg, "validation", "walk") == "blocs":
        return main_blocs(cfg)
    rng = np.random.default_rng(cfg.graine)
    torch.manual_seed(cfg.graine)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifeste = f"run_{cfg.prefixe}.json"
    if os.path.exists(manifeste):
        raise FileExistsError(f"{manifeste} existe : nouveau prefixe, ou lancer.ps1")
    with open(manifeste, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=1, default=str)

    print("=" * 70)
    print(f"  KAIROS EN JEU — BTCUSD M{cfg.minutes_par_barre} : une "
          f"{'semaine' if cfg.partie == 'semaine' else 'journee'} = une partie")
    print("=" * 70)
    print(f"  regles : {cfg.jetons} coups par partie, {cfg.positions_max} position(s) a la "
          f"fois, fin de partie a "
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

    marge = fraction_marge(c, atr, cfg)
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
        j_tr = parties(d["time"], a_tr, a_va, cfg)
        j_va = parties(d["time"], a_va, a_te, cfg)
        j_te = parties(d["time"], a_te, f_te, cfg)
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
                marge_d = max(240 // int(cfg.minutes_par_barre), cfg.barres_par_partie // 6)
                sc, cp, tr = joue(policy, departs_tires(j_tr[choix], rng, marge_d), Xn,
                                  Rf, Df, S1, a_va, cfg, device, explore=True,
                                  collecte=True, rangs=rangs_ex, marge=marge)
                b_tr = bilan(sc, cp, c, atr, sp, cfg, frac=frac)
                st = maj_ppo(policy, optims, avantages(tr, cfg), Xn, cfg, device, rng)
                del Rf
            gen = torch.Generator(device=device)
            gen.manual_seed(cfg.graine)
            sv, cv, _ = joue(policy, j_va, Xn, R1, D1, S1, a_te, cfg, device,
                             explore=False, gen=gen, rangs=rangs_ex, marge=marge)
            bv = bilan(sv, cv, c, atr, sp, cfg, ordre=np.arange(N))
            nom = "EXPERT IMITE" if epoch == 0 else f"cout {100 * frac:.0f}%"
            print(f"\nEPOCH {epoch:03d}  {nom:>12}  VAL  {ligne_bilan(bv, cfg)}  "
                  f"{(time.time() - t_ep) / 60:.1f} min", flush=True)
            print(f"  bilan  {ligne_detail(bv, cfg)}", flush=True)
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
                           explore=False, gen=gen, rangs=rangs_ex, marge=marge)
        bt = bilan(s_t, c_t, c, atr, sp, cfg, ordre=np.arange(N))
        print(f"\nTEST fold {fold + 1} ({os.path.basename(src)})  {ligne_bilan(bt, cfg)}",
              flush=True)
        print(f"TEST fold {fold + 1} bilan  {ligne_detail(bt, cfg)}", flush=True)
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
    _u = "semaine" if cfg.partie == "semaine" else "journee"
    print(f"  KAIROS EN JEU — {len(cfg.marches)} MARCHES M{cfg.minutes_par_barre} : "
          f"une {_u} = une partie par marche, la {_u} = leur somme")
    print("=" * 70)
    print(f"  marches : {', '.join(cfg.marches)}")
    print(f"  regles : {cfg.jetons} coups par marche et par {'semaine' if cfg.partie == 'semaine' else 'jour'}, "
          f"{cfg.positions_max} position(s) a la fois, fin de partie a "
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
    _plus = [k for k in ["swap_achat_bps_jour", "swap_vente_bps_jour"] if k in d.columns]
    if int(cfg.minutes_par_barre) == 60:
        from prepare_multi_h1 import EXTRAS_BTC_H1
        _plus += [k for k in EXTRAS_BTC_H1 if k in d.columns]
    d = d[list(dict.fromkeys(["time", "marche", "open", "high", "low", "close",
                              "atr_14", "spread_bar"] + cols + _plus))].reset_index(drop=True)
    N = len(d)
    marche = d["marche"].to_numpy()
    blocs = blocs_marches(marche)
    t_ns = d["time"].values.astype("int64")
    # LA PARTIE COMMUNE : le jour, ou la semaine (du lundi, en UTC).
    dates = (d["time"].dt.to_period("W-SUN").dt.start_time.values if cfg.partie == "semaine"
             else d["time"].dt.normalize().values)
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
        # LE SWAP DE CHAQUE MARCHE, lu dans le cache (voir `prepare_multi_h1`).
        def _cfg_m(a):
            if "swap_achat_bps_jour" not in d.columns:
                return cfg
            return replace(cfg, swap_achat_bps_jour=float(d["swap_achat_bps_jour"].iloc[a]),
                           swap_vente_bps_jour=float(d["swap_vente_bps_jour"].iloc[a]))
        parts = [table_coups(o[a:b], h[a:b], l[a:b], sp[a:b], atr[a:b], _cfg_m(a), frac)
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
        if nom != "BTCUSD":
            ex = None
        elif int(cfg.minutes_par_barre) == 60:
            from prepare_multi_h1 import EXTRAS_BTC_H1
            ex = d[EXTRAS_BTC_H1].iloc[a:b].to_numpy(np.float32)
        else:
            ex = extras_btc(d["time"].iloc[a:b])
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
              f"vente {ic_m[1]:+.3f}" + ("  (+ flux et financement Binance)" if ex is not None else ""),
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
    print(f"  expert  {len(blocs)} LightGBM appris sur le train du fold 1, {time.time() - t_ex:.0f} s ; "
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
    _mb = int(0.4 * cfg.barres_par_partie)
    for fold, (a_, va, te, fi) in enumerate(bornes):
        suffixe = f"_wf{fold + 1}"
        j_tr = journees_multi(d["time"], blocs, a_, va, _mb, cfg)
        j_va = journees_multi(d["time"], blocs, va, te, _mb, cfg)
        j_te = journees_multi(d["time"], blocs, te, fi, _mb, cfg)
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
                marge_d = max(240 // int(cfg.minutes_par_barre), cfg.barres_par_partie // 6)
                sc, cp, tr = joue(policy, departs_tires(j_tr[choix], rng, marge_d),
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
        print(f"TEST fold {fold + 1} bilan  {ligne_marches(c_t, marche, cfg)}", flush=True)
        print(f"TEST fold {fold + 1}  {ligne_style(bt, cfg)}", flush=True)
        with open(f"test_{cfg.prefixe}{suffixe}.json", "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in bt.items()}, fh, indent=1, default=str)
        precedent = src
    print("\nFIN du walk-forward", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
