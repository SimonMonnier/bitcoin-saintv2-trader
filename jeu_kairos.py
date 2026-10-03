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

N_ETAT = 5          # taille historique ; les nouveaux runs ajoutent des etats causaux


def n_etat(cfg) -> int:
    """Conserve les 5 etats historiques et active les extensions par run."""
    return (N_ETAT
            + 3 * int(bool(getattr(cfg, "observe_compte", False)))
            + 3 * int(bool(getattr(cfg, "observe_calibration", False))))


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
    # LE MEME POINT DE DEPART (run multi_m5_02) : voir `debut_commun`.
    # TROIS MODELES INDEPENDANTS, UN COMPTE COMMUN (run multi_m5_03) : voir
    # `modeles_par_marche`.
    # BTC ET ETH SEULS, DEPUIS 2017 (run multi_m5_04) : voir `marches`.
    # UN MODELE APRES L'AUTRE (run multi_m5_05) : voir `main_modeles_par_marche`.
    # LES MODELES PAS A PAS, DANS UN SEUL PROCESSUS (run multi_m5_06) : voir
    # `main_pas_a_pas`.
    # RETOUR AU BTC SEUL — 2026-09-28 (run m5_03), demande du proprietaire :
    # « l'ETH ne rapporte rien, lance que le BTC ». La configuration du run
    # m5_02 (porte 0.85, expert a 10 coups, 10 jetons, une position). Le BTC
    # et l'ETH pas a pas : prefixe kairos_multi_m5_06, `marches` = ("BTCUSD",
    # "ETHUSD"), `cache` = data_cache_MULTI_M5.pkl.
    # Run distinct : le M5_09 reste le modele actuellement branche en demo.
    # M5_19 rejoue strictement la version PPO precedente, sous un prefixe
    # neuf afin de conserver l'audit du run interrompu M5_17.
    # DEUX NOTES SEPAREES ET SAUVEGARDE SANS RUINE (run m5_20) : voir
    # `note_mise_separee`, `mur_mise` et `dd_max_sauvegarde`. Le run m5_19
    # (une seule note pour les huit tetes) : prefixe
    # kairos_jeu_m5_19_thermostat_ppo_lotmin, les trois a False / 1.0.
    # LA NOTE DU MOIS ET LA SERIE NOIRE (run m5_21) : voir `note_mise_mois`
    # et `serie_noire`. Le run m5_20 : prefixe kairos_jeu_m5_20_mise_separee,
    # les deux a False.
    # LA SORTIE EN DEUX TEMPS ET LE LOT MINIMUM A 0.02 (run m5_22) : voir
    # `sortie_deux_temps`. Le run m5_21 : prefixe
    # kairos_jeu_m5_21_mois_serie_noire, sortie_deux_temps False, lot_min 0.01.
    # LA PORTE A 0.90 ET L'OBJECTIF PROCHE CHOISI PAR LE MODELE (run m5_23) :
    # voir `porte_rang_expert` et `tp1_par_objectif`. Le run m5_22 : prefixe
    # kairos_jeu_m5_22_deux_temps, porte 0.85, tp_atr (1, 2, 4, 8),
    # tp1_par_objectif ().
    # QUATRE PAIRES D'OBJECTIFS DE PLUS (run m5_24) : voir `tp1_par_objectif`.
    # Le run m5_23 : prefixe kairos_jeu_m5_23_porte90_tp1, les neuf premieres
    # paires.
    # DOUZE PAIRES, SANS « 4 ATR EN UNE FOIS » (run m5_25). Le run m5_24 :
    # prefixe kairos_jeu_m5_24_paires, treize paires.
    # LA TETE D'ESPERANCE (run m5_26) : voir `apprendre_esperance`. Le run
    # m5_25 : prefixe kairos_jeu_m5_25_paires12, apprendre_esperance False.
    # LES TETES DU DEJA-VU ET METEO (run m5_27) : voir `apprendre_dejavu_meteo`.
    # Le run m5_26 : prefixe kairos_jeu_m5_26_esperance, apprendre_esperance
    # True, apprendre_dejavu_meteo False.
    # LE BUDGET DE BAISSE (run m5_28) : voir `budget_baisse`. Le run m5_27 :
    # prefixe kairos_jeu_m5_27_dejavu_meteo, apprendre_dejavu_meteo True,
    # budget_baisse False.
    # LA TETE DE CONVICTION (run m5_29) : voir `tete_conviction`. Le run
    # m5_28 : prefixe kairos_jeu_m5_28_budget_baisse, budget_baisse True,
    # mise_budget 0.005, tete_conviction False.
    # LA CONVICTION CONTINUE (run m5_30) : un multiplicateur libre au lieu du
    # menu x0.5 a x4 du run m5_29 (prefixe kairos_jeu_m5_29_conviction).
    # LA CONVICTION DU MODELE FINAL COPIEE DES FOLDS (run m5_31) : voir
    # `copie_conviction`. Le run m5_30 (prefixe
    # kairos_jeu_m5_30_conviction_continue) la reapprenait sur l'expert final.
    prefixe: str = "kairos_jeu_m5_31_conviction_folds"
    # Reproduction demandee du run M5_03, avant les corrections de deroulement
    # et de compte introduites dans la version 2.
    # La version 2 rejoue l'equite, la marge et les positions ouvertes : elle
    # est necessaire pour que la tete de lot recoive une recompense reelle.
    jeu_version: int = 2
    # Le modele voit enfin la situation de son compte : equite relative,
    # drawdown depuis le plus haut et part de marge libre. Les checkpoints
    # precedents restent a 5 etats et ne sont donc jamais incompatibles.
    observe_compte: bool = True
    # LA VALIDATION CROISEE PURGEE — 2026-09-28, demande du proprietaire :
    # « entrainer le modele sur des periodes aleatoires pour qu'il apprenne
    # tous les types de marches ». Le walk-forward (h1_05) : +447.56, -441.18,
    # -172.80 $. En "blocs", l'historique est coupe en `n_blocs` blocs ; chaque
    # bloc est teste a son tour par un modele NEUF entraine sur TOUS les
    # autres (passe et avenir), la validation etant le bloc le plus eloigne,
    # avec une zone tampon de `purge_semaines` + l'horizon + la memoire
    # (ou `purge_barres_fixes` lorsqu'une reproduction historique est demandee)
    # retiree de l'entrainement autour du test et de la validation.
    # UN RESULTAT NEGATIF EST DEFINITIF (meme en voyant l'avenir des autres
    # blocs, pas d'avantage) ; un resultat positif est un PLAFOND, a confirmer
    # en walk-forward et en demo. "walk" = le walk-forward d'avant.
    validation: str = "blocs"
    n_blocs: int = 10
    # APRES LA MESURE CROISEE, le modele destine au live ne repart pas de
    # zero. Il part du champion de validation et traverse les dix regimes
    # dans l'ordre, en rejouant equitablement les regimes precedents. Les
    # folds restent independants : transmettre leurs poids entre eux ferait
    # fuir leur zone de test dans le fold suivant et rendrait les resultats
    # de validation mensongers.
    deploiement_continu: bool = True
    # 0 = une duree robuste, la mediane des epoques championnes des folds.
    # Au minimum dix epoques afin que chaque regime soit le regime courant
    # au moins une fois dans le curriculum.
    epochs_deploiement: int = 0
    # Le deploy apprend aussi les decisions des dix enseignants, mais chaque
    # decision provient exclusivement du bloc de test que son enseignant
    # n'avait jamais vu : distillation OOF, sans fuite de validation.
    distillation_oof_epochs: int = 2
    purge_semaines: int = 1
    # Compatibilite avec les resultats historiques M5 : l'utilisateur demande
    # explicitement de revenir a la zone tampon de 388 bougies (un jour +
    # horizon et memoire), au lieu de la purge hebdomadaire complete.
    purge_barres_fixes: int = 388
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
    cache: str = "data_cache_BTCUSD_M5_BINANCE.pkl"
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
    # SANS LE ZEC LE 2026-09-28 (run multi_m5_04), demande du proprietaire :
    # « recuperer 2017-2019 et enlever le ZECUSD ». Le ZEC n'est cote chez
    # Binance que depuis mars 2019 : le point de depart commun (`debut_commun`)
    # le retirait au BTC et a l'ETH, et avec lui l'historique le plus agite
    # (correlation de l'expert du BTC au test du bloc 1 : +0.088 / +0.070 sur
    # 2019, +0.119 / +0.113 sur 2017-2018). Sans lui, BTC et ETH commencent
    # tous deux le 2017-08-24 : memes blocs que le BTC seul.
    # () LE 2026-09-28 (run m5_03) : le BTC seul, voir `prefixe`.
    marches: Tuple[str, ...] = ()
    # LE MEME POINT DE DEPART POUR TOUS LES MARCHES — 2026-09-28 (run
    # multi_m5_02), demande du proprietaire : « que les trois cryptos
    # commencent depuis le meme point de depart, qu'il n'y ait pas une crypto
    # qui ne fasse rien, meme si on perd un peu d'historique ». Le run
    # multi_m5_01 testait son bloc 1 (2017-08 -> 2018-07) sans le ZEC, cote
    # chez Binance seulement depuis mars 2019. Avec True, l'historique de
    # TOUS les marches commence a la premiere bougie du marche le plus
    # recent (le ZEC : 2019-03-28) ; les blocs en dates sont alors coupes
    # dans cette periode commune. Les features, calculees avant la coupe,
    # gardent leur amorce.
    debut_commun: bool = True
    # TROIS MODELES INDEPENDANTS, UN COMPTE COMMUN — 2026-09-28 (run
    # multi_m5_03), demande du proprietaire. Au run multi_m5_02 (un tronc
    # commun, des tetes par marche), le BTC ne gagnait plus ce qu'il gagnait
    # seul : le tronc, l'optimiseur de chaque role, la normalisation et le
    # choix de la meilleure epoch etaient partages avec l'ETH et le ZEC.
    #
    # Avec True, CHAQUE MARCHE A SON MODELE, entraine exactement comme le BTC
    # seul (`main_blocs`) : son tronc, ses tetes, ses optimiseurs, sa
    # normalisation, son expert, ses parties d'entrainement. Chacun garde SON
    # meilleur modele : a une epoch, on peut en sauvegarder un, deux, trois
    # ou aucun, chacun seulement s'il bat son propre record en validation.
    # Les trois tournent en parallele (`main_modeles_par_marche`), et leurs
    # coups sont rejoues sur UN COMPTE COMMUN (`bilan_commun`) : un seul
    # solde, la mise de chaque coup prise sur ce solde, la marge partagee, le
    # drawdown, le win rate et le reste mesures sur ce compte.
    modeles_par_marche: bool = True
    # Rempli par `config_marche` pour le modele d'UN marche : son nom, et le
    # dossier ou il depose ses coups pour le compte commun.
    marche_seul: str = ""
    echanges: str = ""
    # Les marches du compte commun, pour leur point de depart commun.
    marches_communs: Tuple[str, ...] = ()
    # Le seul bloc que joue ce processus (1 a `n_blocs`) ; 0 = tous.
    bloc_seul: int = 0
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
    # 0.01 -> 0.02 LE 2026-10-03 (run m5_22), demande du proprietaire : un
    # coup doit pouvoir etre coupe en deux. Voir `sortie_deux_temps`.
    lot_min: float = 0.02
    pas_lot: float = 0.01
    lot_max: float = 100.0
    contrat: float = 1.0
    # Tete de taille : elle choisit un des niveaux entre le minimum et le lot
    # maximum calcule a l'entree (equite, stop et marge). Le lot envoye reste
    # toujours un multiple de 0.01 ; aucun plafond fixe en lots n'est impose.
    apprendre_lot: bool = True
    niveaux_lot: int = 20
    # Sixieme tete : part de l'equite que le coup a le droit de perdre au
    # stop. Ce n'est pas une limite fixe a 1 % : la politique choisit elle
    # meme l'un de ces pourcentages pour chaque occasion. 100 % est le
    # plafond de solvabilite theorique ; le courtier, la marge et le pas de
    # lot restent des bornes dures.
    apprendre_risque: bool = True
    niveaux_risque_pct: Tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0,
                                               35.0, 50.0, 75.0, 100.0)
    # TETE D'ARBITRAGE : budget commun qui pilote les deux propositions
    # (lot et risque). Ce n'est pas un plafond fixe par trade : 100 % est la
    # capacite sure calculee avec equite, stop, marge et limites MT5 ; la
    # tete choisit quelle part de cette capacite est justifiee par le signal.
    apprendre_allocation: bool = True
    niveaux_allocation_pct: Tuple[float, ...] = (10.0, 25.0, 45.0, 70.0, 100.0)
    allocation_aux_coef: float = 0.35
    # HUITIEME TETE — thermostat de confiance. Elle ne touche ni au sens ni
    # aux barrieres : elle convertit l'historique CAUSAL des erreurs recentes
    # du modele en un multiplicateur de l'enveloppe lot/risque. Le minimum
    # reste 20 %, donc elle ne peut pas apprendre a se refugier a zero.
    apprendre_confiance: bool = True
    niveaux_confiance_pct: Tuple[float, ...] = (20.0, 40.0, 65.0, 85.0, 100.0)
    # Trois etats bornes : R recent, surprise negative (valeur predite moins
    # R realise), et serie de pertes. Ils ne lisent que des positions deja
    # cloturees ; jeu, validation et live les mettent a jour de la meme facon.
    observe_calibration: bool = True
    # LA TETE D'ESPERANCE — 2026-10-03 (run m5_26), demande du proprietaire :
    # « miser plus, avec une tete en plus ». Au run m5_25, les quatre tetes de
    # mise, apprises par la seule note du mois (douze mois par epoch, tres
    # bruitee), restaient collees au lot minimum : 0.5 % du compte risque par
    # coup, quand la regle de Kelly en aurait permis ~6.7 %.
    #
    # Avec True, une NEUVIEME tete predit, pour chaque coup possible (sens,
    # paire d'objectifs, stop), son resultat en multiples du risque pris :
    # moyenne et dispersion. Elle apprend comme un eleve avec un corrige —
    # apres chaque coup joue, on lui montre le vrai resultat (`entraine_esperance`),
    # sur des milliers de coups par epoch. Elle est derriere le mur (`mur_mise`).
    # Sa prediction donne la MISE DE BASE du coup : `kelly_fraction` fois la
    # mise de Kelly, moyenne / (moyenne^2 + variance). Les quatre tetes de mise
    # AJUSTENT autour de cette base au lieu de partir du lot minimum : le lot
    # choisit un multiplicateur `multiplicateurs_lot` (x0.25 a x2), l'allocation
    # et la confiance aussi (`multiplicateurs_allocation`, `..._confiance`), la
    # tete de risque reste un plafond. Serie noire, marge et lot minimum
    # restent des bornes dures.
    #
    # COUPEE LE 2026-10-03 (run m5_27), demande du proprietaire. Au run m5_26,
    # bloc 1, la correlation de rang entre sa prediction et le resultat reel en
    # validation est passee de +0.23 (epoch 2) a -0.11 (epoch 7) : elle
    # predisait le resultat du coup, c'est-a-dire la direction, ce que le
    # signal lui-meme ne fait qu'a peine. Voir `apprendre_dejavu_meteo`.
    apprendre_esperance: bool = False
    kelly_fraction: float = 0.5
    # x0.25 a x2, x0.5 a x1.5 au run m5_26 (esperance) ; depuis le run m5_28,
    # ceux du budget de baisse.
    multiplicateurs_lot: Tuple[float, ...] = tuple(
        float(x) for x in np.round(np.geomspace(0.5, 1.5, 20), 4))
    multiplicateurs_allocation: Tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
    multiplicateurs_confiance: Tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
    esperance_passes: int = 2
    # LES TETES DU DEJA-VU ET METEO — 2026-10-03 (run m5_27), demande du
    # proprietaire : « invente un truc avec des nouvelles tetes ». La tete
    # d'esperance predisait le resultat du coup et a echoue. Ces deux tetes ne
    # predisent pas la direction : elles predisent ce qui se prevoit.
    #
    # LE DEJA-VU : « ce marche, je l'ai deja vu ? ». La tete reconstitue la
    # bougie que le modele regarde (ses features de marche et d'expert) a partir
    # du seul resume qu'en garde le tronc. Sur un marche semblable a
    # l'entrainement elle y arrive ; sur un marche jamais vu elle se trompe :
    # son erreur (en log) mesure la nouveaute. Le jeu H1 l'a montre : +447 $ au
    # fold 1, -441 et -173 $ aux folds 2 et 3, le modele perd quand le marche
    # change de nature.
    #
    # LA METEO : de combien le cours va bouger dans les `horizon_max` bougies
    # qui suivent (plus haut moins plus bas, en log de bps), quel que soit le
    # sens. La volatilite vient par vagues : c'est ce qui se prevoit le mieux.
    # Le cout Vantage est fixe en bps, le mouvement grandit avec elle : les
    # jours agites, il ne pese presque plus rien.
    #
    # Les deux tetes sont DERRIERE LE MUR : elles lisent le tronc detache et
    # apprennent par leur seule cible (`entraine_dejavu_meteo`), sans toucher
    # le signal ni sa note. Leurs predictions, normalisees, deviennent deux
    # ENTREES de plus des quatre tetes de mise, qui apprennent avec la note du
    # mois a en tirer des mises plus grosses ou plus petites. La veille affiche
    # le profit factor des coups de validation par tranche de chaque tete.
    #
    # COUPEES LE 2026-10-03 (run m5_28), demande du proprietaire. Au run m5_27
    # (blocs 1 a 3), elles prevoyaient bien leur cible — meteo correlee a
    # +0.35 / +0.50 au mouvement reel en validation — mais aucune ne triait les
    # coups : la tranche qui payait le mieux changeait de bloc en bloc
    # (deja-vu : les plus nouveaux aux blocs 1 et 2, les plus familiers au
    # bloc 3 ; meteo : les jours calmes au bloc 1, agites au bloc 2), et les
    # tetes de mise misaient pareil dans toutes les tranches.
    apprendre_dejavu_meteo: bool = False
    dejavu_meteo_passes: int = 2
    dejavu_meteo_etats: int = 16384
    # LE BUDGET DE BAISSE — 2026-10-03 (run m5_28), demande du proprietaire :
    # « invente autre chose qui marche cette fois ». Trois tetes ont cherche
    # QUELS coups meritent plus (esperance, deja-vu, meteo) : aucune n'a
    # trouve de regle stable. Ce qui est stable, c'est l'avantage MOYEN du
    # signal. La mesure du 2026-10-03 sur les coups du run m5_27 : le meme
    # signal, rejoue au test a mise FIXE, rapporte bien plus que la mise
    # apprise, parce que les tetes de mise collaient au lot minimum (mediane
    # 0.31 a 0.77 % du compte) ; elles montaient aussi parfois a 3 % sur un
    # signal perdant (bloc 4, validation -96 %).
    #
    # La regle : apres chaque validation, ses coups sont rejoues
    # `budget_baisse_tirages` fois, les journees dans le desordre, a chaque
    # mise possible. La MISE DE BASE est la plus forte pour laquelle la pire
    # baisse ne depasse `budget_baisse_pct` qu'une fois sur 10
    # (`budget_baisse_quantile`). Elle sert a l'epoch suivante, au test (celle
    # du modele garde), au modele final (la mediane des dix folds) et au live.
    # Quand le signal perd en validation, la mise tombe d'elle-meme au minimum.
    # Les quatre tetes de mise restent : elles choisissent un multiplicateur
    # (lot x0.5 a x1.5, allocation et confiance x0.8 a x1.2, produit borne a
    # [0.5, 1.5]) et la mesure rejoue leurs vrais multiplicateurs. La serie
    # noire, la tete de risque, la marge et le lot minimum restent des bornes.
    #
    # COUPE LE 2026-10-03 (run m5_29), demande du proprietaire : une mise tiree
    # des resultats passes n'est pas ce qu'il veut. `mise_budget` reste la
    # mise de base, fixe : 1 % du compte risque au stop, le R du jeu.
    budget_baisse: bool = False
    budget_baisse_pct: float = 0.30
    budget_baisse_quantile: float = 0.10
    budget_baisse_tirages: int = 200
    mise_budget: float = 0.01
    # LA TETE DE CONVICTION — 2026-10-03 (run m5_29), demande du proprietaire :
    # « je veux que ce signal soit exploite par les tetes de mise, qu'elles
    # gerent elles-memes le moment ou elles doublent, triplent, quadruplent ».
    #
    # LE SIGNAL. Mesure du 2026-10-03 sur les coups du run m5_27 : quand le
    # rang glissant de l'expert (sa note du moment parmi les 7 derniers jours)
    # depasse 0.99, le coup paie mieux dans les 3 blocs de validation (PF 1.06
    # / 1.20 / 1.14 contre 1.02 / 1.06 / 0.97) et dans les 3 tests (1.32 /
    # 1.60 / 1.58 contre 1.28 / 1.05 / 1.37), 22 mois sur 33 en validation.
    # Les tetes de mise avaient ce rang dans leur entree, noye parmi des
    # centaines d'autres chiffres, et la note du mois, trop bruitee, ne le
    # leur a jamais appris.
    #
    # LA TETE. Une cinquieme tete de mise, petite, qui ne lit QUE la conviction
    # de l'expert (le rang du sens choisi) et le sens. Elle calcule un
    # multiplicateur de la mise, en plus des quatre autres tetes. Elle ne peut
    # pas apprendre le bruit du tronc qui a perdu la tete d'esperance : elle
    # n'en lit rien.
    # CONTINU DEPUIS LE RUN m5_30, demande du proprietaire : « pourquoi c'est
    # limite a x4, pourquoi ce n'est pas dynamique ». Au run m5_29, elle
    # choisissait dans un menu (x0.5, 1, 1.5, 2, 3, 4). Elle calcule
    # desormais un multiplicateur libre, de x`conviction_min` a
    # x`conviction_max` (un garde-fou numerique : la serie noire coupe bien
    # avant), au point ou la croissance du compte est la meilleure.
    #
    # SON APPRENTISSAGE (`entraine_conviction`). La taille de la mise ne change
    # pas l'issue du coup : on connait donc EXACTEMENT ce qu'aurait donne
    # n'importe quel multiplicateur sur chaque coup joue, log(1 + mise x m x R).
    # La tete monte la pente de cette croissance, sur chaque coup, sans
    # tirage : le bruit qui a noye la note du mois disparait. Elle vise la
    # fraction `conviction_kelly` de la mise de Kelly (demi-Kelly) : la
    # croissance est comptee comme si la mise etait deux fois plus forte, ce
    # qui lui fait encaisser ses erreurs d'estimation.
    # LE BUDGET CONSTANT. Sans contrainte, la tete miserait le maximum partout
    # (les coups d'entrainement gagnent). Un prix du risque `lambda`, ajuste
    # en continu, maintient le multiplicateur MOYEN a 1 : pour quadrupler un
    # coup, elle doit miser moins sur d'autres. Elle apprend OU mettre le
    # risque, pas combien en prendre. Serie noire, marge, tete de risque et
    # lot minimum restent des bornes.
    tete_conviction: bool = True
    conviction_min: float = 0.1
    conviction_max: float = 25.0
    conviction_kelly: float = 0.5
    conviction_passes: int = 2
    conviction_pas_lambda: float = 0.01
    # DEUX NOTES SEPAREES — 2026-10-02 (run m5_20), demande du proprietaire :
    # garder le signal d'achat et de vente, et la prise de risque au-dela de
    # 1 %, sans que le compte finisse ruine. Au run m5_19, une seule note (le
    # gain en dollars rapporte a 1 % du compte) entrainait les huit tetes :
    # miser plus rapportait toujours plus en moyenne, le risque choisi montait
    # a 63-78 % du compte par coup et la validation du bloc 1 finissait a
    # -87 / -100 % dans 19 epochs sur les 20 dernieres.
    #
    # Avec True : l'entree, l'objectif, le stop et la valeur gardent EXACTEMENT
    # leur note. Les quatre tetes de mise (lot, risque, allocation, confiance)
    # sont notees sur la croissance du compte en pourcentage, le logarithme du
    # solde apres le coup sur le solde avant (`avantage_mise`) : la regle qui
    # mene a la mise de Kelly. Un gain de +5 % vaut +0.049, une perte de 50 %
    # vaut -0.69, la ruine vaut -6.9. Miser gros reste possible et paye quand
    # le signal est fort ; tout miser ne paye jamais.
    note_mise_separee: bool = True
    # LE MUR : les tetes de mise LISENT le tronc mais ne le modifient pas en
    # apprenant (gradient coupe). Sans lui, leur apprentissage remonterait
    # dans le tronc, donc dans le signal d'achat et de vente.
    mur_mise: bool = True
    # LA NOTE DU MOIS — 2026-10-03 (run m5_21), demande du proprietaire. Au
    # run m5_20, la note des tetes de mise etait donnee coup par coup, sur
    # des journees qui repartaient toutes de 1 000 $ : une ruine pesait le
    # poids d'un seul coup, noyee parmi des milliers de petits gains, et le
    # compte de validation a fini vide dans 6 epochs sur 7 apres l'epoch 15.
    #
    # Avec True, chaque epoch joue en plus `mois_par_epoch` mois de
    # `mois_jours` journees d'affilee, chacun sur UN compte (positions et
    # solde gardes d'un jour a l'autre). Ces mois n'entrainent QUE les tetes
    # de mise : chaque decision de mise y est notee sur la croissance du
    # compte depuis cette decision jusqu'a la fin du mois (log du solde final
    # sur le solde avant le coup). Une ruine note tres mal toutes les mises
    # qui l'ont construite. Le signal (entree, objectif, stop, valeur)
    # n'apprend que sur les journees, comme avant, et ne voit pas cette note.
    note_mise_mois: bool = True
    # LE SIGNAL EN MODE REEL PENDANT LES MOIS — 2026-10-03 (run m5_24). Au run
    # m5_23, les mois etaient joues en exploration : le signal tirait ses
    # coups au hasard selon ses probabilites, bien plus faibles que ses
    # meilleures decisions (validation, live). Les tetes de mise y voyaient un
    # avantage presque nul et misaient au plus petit (0.5-0.7 % du compte par
    # coup, lot minimum dans la plupart des cas). Avec True, le signal joue
    # sa meilleure decision et seules les tetes de mise explorent.
    mois_signal_reel: bool = True
    mois_jours: int = 30
    mois_par_epoch: int = 12
    # LA SERIE NOIRE — 2026-10-03 (run m5_21), demande du proprietaire.
    # Avant chaque coup, la part du compte risquee ne depasse jamais ce que
    # la pire serie de pertes du modele laisse survivre : si les
    # `serie_noire_n` prochains coups perdaient tous, le compte ne perdrait
    # pas plus de `serie_noire_perte_max`. Plafond = (1 - (1 - 0.5)^(1/N)) / k,
    # ou k est la perte REELLE d'un coup perdant rapportee a la perte prevue
    # au stop (gaps, spread, glissements), la plus forte observee. N et k sont
    # remesures sur chaque validation (jamais sur le test) et suivent le
    # modele jusqu'au live. La tete de risque garde le choix sous ce plafond.
    serie_noire: bool = True
    serie_noire_n: int = 10
    serie_noire_n_min: int = 5
    serie_noire_k: float = 1.0
    serie_noire_perte_max: float = 0.50
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
    # LE MENU DES OBJECTIFS DEPUIS LE RUN m5_23 : chaque choix est une paire
    # (objectif proche, objectif final), voir `tp1_par_objectif`. `tp_atr`
    # porte l'objectif final de chaque choix.
    tp_atr: Tuple[float, ...] = (1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 8.0, 8.0,
                                 4.0, 8.0, 8.0)
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
    # LA SORTIE EN DEUX TEMPS — 2026-10-03 (run m5_22), demande du
    # proprietaire : un compromis entre l'ancien style (objectif proche,
    # win rate de 66 %) et celui du run m5_21 (objectif a 6-8 ATR, stop a
    # 1 ATR, win rate de 20 %). Quand l'objectif choisi est plus loin que
    # `tp1_atr` :
    #   1. la moitie du coup sort a l'objectif proche (`tp1_atr` ATR) ;
    #   2. le stop de l'autre moitie remonte alors au prix d'entree, SPREAD ET
    #      GLISSEMENT DE SORTIE COMPRIS, plus `be_marge_bps` : ressortir a ce
    #      stop ne fait pas perdre (sauf trou de cotation) ;
    #   3. cette moitie court vers l'objectif choisi, au plus `horizon_max`.
    # Un objectif choisi a `tp1_atr` ou moins reste un coup en une fois : le
    # modele choisit lui-meme son melange des deux styles. Codes de sortie :
    # 0 objectif, 1 stop, 2 temps, 3 objectif proche puis entree, 4 objectif
    # proche puis temps.
    sortie_deux_temps: bool = True
    tp1_atr: float = 1.0
    be_marge_bps: float = 1.0
    # L'OBJECTIF PROCHE CHOISI PAR LE MODELE — 2026-10-03 (run m5_23),
    # demande du proprietaire. Au run m5_22, l'objectif proche etait fixe a
    # 1 ATR : 40 % des coups le touchaient puis ressortaient a l'entree pour
    # +0.13 R seulement. Un objectif proche par choix du menu (`tp_atr`) :
    # 0 = coup en une fois. Le modele choisit ainsi en un seul geste son
    # objectif proche et son objectif final :
    #   1 en une fois, 2 en une fois, 0.5 puis 2, 1 puis 2, 0.5 puis 4,
    #   1 puis 4, 1.5 puis 4, 1 puis 8, 1.5 puis 8 (en ATR).
    # QUATRE PAIRES DE PLUS LE 2026-10-03 (run m5_24), demande du proprietaire :
    # « les profits sont trop faibles ». 4 en une fois, 2 puis 4, 0.5 puis 8,
    # 2 puis 8. Les neuf premieres paires gardent leur place. Le run m5_23
    # (bloc 1, epochs 11-18, couts pleins) : +0.64 a +1.12 $/jour, PF 1.08 a
    # 1.16, win rate 73-76 %, mise reelle 0.5-0.7 % du compte.
    # Vide = `tp1_atr` pour tous les objectifs plus lointains (run m5_22).
    # « 4 EN UNE FOIS » RETIREE LE 2026-10-03 (run m5_25), demande du
    # proprietaire : au run m5_24, le modele s'y est jete (41 puis 68 puis
    # 76 % des coups, stop serre a 1 ATR) et le win rate est tombe de 69 a
    # 27 % en trois epochs — la derive du run m5_21. Sans sortie en deux
    # temps, un coup qui monte puis redescend finit au stop.
    tp1_par_objectif: Tuple[float, ...] = (0.0, 0.0, 0.5, 1.0, 0.5, 1.0, 1.5, 1.0, 1.5,
                                           2.0, 0.5, 2.0)
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
    # Reference de 1 R (et non plus un plafond par trade quand `apprendre_lot`
    # est actif) : le lot est alors choisi par la tete sous seule contrainte de
    # marge/equite et des bornes MT5.
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
    # RETOUR AUX 16 COUPLES, demande du proprietaire : aucune exclusion.
    # Ce champ ne sert plus qu'a reproduire la cible historique du run m5_04.
    expert_ratio_min: float = 0.0
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
    # 0.85 -> 0.90 LE 2026-10-03 (run m5_23), demande du proprietaire : moins
    # de coups, mais plus forts. Mesure du 2026-09-28 (`mesure_porte_m5.py`) :
    # la tranche 0.85 - 0.90 perdait en entrainement (-0.008 / -0.005 R) et en
    # validation (-0.003 / -0.016 R) ; celles au-dessus de 0.90 gagnaient en
    # entrainement. La mise suivant l'avantage (note du mois), des coups plus
    # forts appellent des mises plus grosses.
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
    # LA SAUVEGARDE SANS RUINE — 2026-10-02 (run m5_20), demande du
    # proprietaire. Le modele garde est celui dont le compte de validation
    # grossit le plus (sur un seul compte, c'est le meme classement que le
    # profit par jour), parmi les epochs dont la pire baisse en validation
    # reste au-dessus de -`dd_max_sauvegarde`. Au run m5_19, l'epoch gardee
    # au bloc 1 (la 18, -35.7 %) etait entouree d'epochs a -84 et -100 %.
    # 1.0 = pas de limite, comme avant.
    dd_max_sauvegarde: float = 0.50
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
    # GOUVERNEUR D'EXPOSITION — il ne modifie jamais le signal ni la
    # recompense PPO. Il reduit seulement le plafond de lot disponible a la
    # tete de risque quand le compte recule depuis son plus haut. A 20 % de
    # drawdown, une nouvelle degradation bloque les entrees jusqu'au prochain
    # jour ; le lendemain repart avec l'exposition minimale, pas a 100 %.
    gouverneur_exposition: bool = False
    gouverneur_dd_frein: float = 0.05
    gouverneur_dd_stop: float = 0.20
    gouverneur_exposition_min: float = 0.25
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
    """Les features du cache que joue ce jeu : BTC M5 (la configuration
    actuelle), BTC H1, multi-marches H1 / M5, ou M1 (les tests du moteur). Le
    jeu M15 a ete retire du depot le 2026-10-02 ; il reste dans l'historique git."""
    if tuple(getattr(cfg, "marches", ())) and int(cfg.minutes_par_barre) == 60:
        from prepare_multi_h1 import FEATURE_COLS_MULTI_H1
        return list(FEATURE_COLS_MULTI_H1)
    if tuple(getattr(cfg, "marches", ())) and int(cfg.minutes_par_barre) == 5:
        from prepare_multi_m5 import FEATURE_COLS_MULTI_M5
        return list(FEATURE_COLS_MULTI_M5)
    if int(cfg.minutes_par_barre) == 60 and not tuple(getattr(cfg, "marches", ())):
        from prepare_btc_h1_binance import FEATURE_COLS_H1
        return list(FEATURE_COLS_H1)
    if int(cfg.minutes_par_barre) == 5 and not tuple(getattr(cfg, "marches", ())):
        from prepare_btc_m5 import FEATURE_COLS_M5
        return list(FEATURE_COLS_M5)
    # Le jeu en bougies d'une minute sur les colonnes du SAINT : celui des
    # tests du moteur (`test_jeu.py`).
    if int(cfg.minutes_par_barre) == 1:
        return list(FEATURE_COLS)
    raise ValueError(f"pas de jeu de features pour {cfg.minutes_par_barre} min "
                     f"({len(getattr(cfg, 'marches', ()))} marches)")


# ======================================================================
# LE MODELE — le SAINT du run, deux tetes de barrieres en plus
# ======================================================================
class PolitiqueJeu(SAINTPolicySingleHead):
    """Le tronc et les tetes d'achat et de vente du run PPO ; les tetes de
    coupure des gains et des pertes choisissent desormais l'objectif et le
    stop, AU MOMENT DU COUP, en lisant le tronc et le sens."""

    def __init__(self, cfg: JeuConfig):
        super().__init__(
            n_features=len(colonnes_jeu(cfg)) + cfg.n_expert + n_etat(cfg),
            d_model=cfg.d_model,
            num_blocks=cfg.num_blocks, heads=cfg.heads, n_freq=cfg.n_freq,
            mlp_dim=cfg.mlp_dim, lecture="colonnes", dropout=0.05, ff_mult=2,
            max_len=cfg.lookback, n_actions=N_ACTIONS, n_ref=0)
        dl = self.dim_lecture
        # LES TETES DU DEJA-VU ET METEO. Voir `apprendre_dejavu_meteo`. Elles
        # ne servent qu'avec les quatre tetes de mise, dont elles nourrissent
        # l'entree, et sur un seul marche.
        _marches = tuple(getattr(cfg, "marches", ()))
        self.apprendre_dejavu_meteo = (
            bool(getattr(cfg, "apprendre_dejavu_meteo", False))
            and not (len(_marches) > 1 and getattr(cfg, "tetes_par_marche", False))
            and all(bool(getattr(cfg, f"apprendre_{n}", False))
                    for n in ("lot", "risque", "allocation", "confiance")))
        n_prev = 2 if self.apprendre_dejavu_meteo else 0

        def _lecteur(extra=0):
            return nn.Sequential(nn.Linear(dl + 1 + extra, cfg.mlp_dim), nn.GELU(),
                                 nn.Linear(cfg.mlp_dim, 64), nn.GELU())
        self.lecteur_objectif = _lecteur()
        self.tete_objectif = nn.Linear(64, len(cfg.tp_atr))
        self.lecteur_stop = _lecteur()
        self.tete_stop = nn.Linear(64, len(cfg.sl_atr))
        self.apprendre_lot = bool(getattr(cfg, "apprendre_lot", False))
        if self.apprendre_lot:
            self.lecteur_lot = _lecteur(n_prev)
            self.tete_lot = nn.Linear(64, int(cfg.niveaux_lot))
        self.apprendre_risque = bool(getattr(cfg, "apprendre_risque", False))
        if self.apprendre_risque:
            self.lecteur_risque = _lecteur(n_prev)
            self.tete_risque = nn.Linear(64, len(cfg.niveaux_risque_pct))
        self.apprendre_allocation = bool(getattr(cfg, "apprendre_allocation", False))
        if self.apprendre_allocation:
            self.lecteur_allocation = _lecteur(n_prev)
            self.tete_allocation = nn.Linear(64, len(cfg.niveaux_allocation_pct))
        self.apprendre_confiance = bool(getattr(cfg, "apprendre_confiance", False))
        if self.apprendre_confiance:
            self.lecteur_confiance = _lecteur(n_prev)
            self.tete_confiance = nn.Linear(64, len(cfg.niveaux_confiance_pct))
        # LA TETE D'ESPERANCE. Voir `apprendre_esperance`.
        # Elle ne sert qu'avec les quatre tetes de mise, qu'elle remet au centre.
        self.apprendre_esperance = (bool(getattr(cfg, "apprendre_esperance", False))
                                    and self.apprendre_lot and self.apprendre_risque
                                    and self.apprendre_allocation and self.apprendre_confiance)
        self.k_objectifs, self.k_stops = len(cfg.tp_atr), len(cfg.sl_atr)
        if self.apprendre_esperance:
            self.lecteur_esperance = _lecteur()
            self.tete_esperance = nn.Linear(64, 2 * self.k_objectifs * self.k_stops)
            nn.init.zeros_(self.tete_esperance.weight)
            nn.init.zeros_(self.tete_esperance.bias)
        if self.apprendre_dejavu_meteo:
            # elles lisent le tronc seul, sans le sens
            def _lecteur_nu():
                return nn.Sequential(nn.Linear(dl, cfg.mlp_dim), nn.GELU(),
                                     nn.Linear(cfg.mlp_dim, 64), nn.GELU())
            self.n_vu = len(colonnes_jeu(cfg)) + int(cfg.n_expert)
            self.lecteur_dejavu, self.tete_dejavu = _lecteur_nu(), nn.Linear(64, self.n_vu)
            self.lecteur_meteo, self.tete_meteo = _lecteur_nu(), nn.Linear(64, 1)
            # moyenne et ecart du deja-vu, puis de la meteo, sur l'entrainement ;
            # le dernier vaut 1 une fois mesures. Ils voyagent avec le modele.
            self.register_buffer("norme_previsions", torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]))
        # LA TETE DE CONVICTION. Voir `tete_conviction`. Elle ne lit que la
        # conviction de l'expert et le sens ; au depart elle mise x1.
        self.tete_conviction = (bool(getattr(cfg, "tete_conviction", False))
                                and not (len(_marches) > 1 and getattr(cfg, "tetes_par_marche", False))
                                and all(bool(getattr(cfg, f"apprendre_{n}", False))
                                        for n in ("lot", "risque", "allocation", "confiance")))
        if self.tete_conviction:
            self.reseau_conviction = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 32),
                                                   nn.GELU(), nn.Linear(32, 1))
            nn.init.zeros_(self.reseau_conviction[-1].weight)
            nn.init.zeros_(self.reseau_conviction[-1].bias)
            # log(multiplicateur) = centre + demi-largeur x tanh(sortie + decalage) :
            # borne en douceur a [min, max], et x1 pour une sortie nulle
            lo, hi = math.log(float(cfg.conviction_min)), math.log(float(cfg.conviction_max))
            self._conv_c, self._conv_h = (lo + hi) / 2.0, (hi - lo) / 2.0
            self._conv_d = math.atanh(-self._conv_c / self._conv_h)
            # le prix du risque (en R) et s'il a ete pose : ils voyagent avec le modele
            self.register_buffer("conviction_lambda", torch.tensor(0.0))
            self.register_buffer("conviction_pret", torch.tensor(0.0))
        # LE MUR des tetes de mise. Voir `mur_mise`.
        self.mur_mise = bool(getattr(cfg, "mur_mise", False))
        for m in list(self.lecteur_objectif) + list(self.lecteur_stop):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        for m in (self.tete_objectif, self.tete_stop,
                  *((self.tete_lot,) if self.apprendre_lot else ()),
                  *((self.tete_risque,) if self.apprendre_risque else ()),
                  *((self.tete_allocation,) if self.apprendre_allocation else ()),
                  *((self.tete_confiance,) if self.apprendre_confiance else ())):
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
            if self.apprendre_lot:
                self.lecteur_lot_m, self.tete_lot_m = _dup(self.lecteur_lot), _dup(self.tete_lot)
            if self.apprendre_risque:
                self.lecteur_risque_m, self.tete_risque_m = _dup(self.lecteur_risque), _dup(self.tete_risque)
            if self.apprendre_allocation:
                self.lecteur_allocation_m, self.tete_allocation_m = _dup(self.lecteur_allocation), _dup(self.tete_allocation)
            if self.apprendre_confiance:
                self.lecteur_confiance_m, self.tete_confiance_m = _dup(self.lecteur_confiance), _dup(self.tete_confiance)

    def marche_de(self, x: torch.Tensor) -> torch.Tensor:
        """(B,) : l'indice du marche de chaque ligne, lu dans ses colonnes
        `m_*` (normalisees : la seule a 1 reste la plus grande)."""
        return x[:, -1, self.idx_marche].argmax(-1)

    @staticmethod
    def _par_marche(k, lecteurs, tetes, z):
        sortie = torch.stack([t(lec(z)) for lec, t in zip(lecteurs, tetes)], 1)
        return sortie[torch.arange(len(k), device=z.device), k]

    def conviction(self, c: torch.Tensor) -> torch.Tensor:
        """(B,) : le multiplicateur de conviction, continu, a partir de
        `entrees_conviction`. Voir `tete_conviction`."""
        g = self.reseau_conviction(c).squeeze(-1)
        return torch.exp(self._conv_c + self._conv_h * torch.tanh(g + self._conv_d))

    def previsions(self, zn: torch.Tensor, x: torch.Tensor):
        """(erreur de reconstitution (B,), meteo predite (B,)) a partir du
        tronc DETACHE. Voir `apprendre_dejavu_meteo`."""
        zd = zn.detach()
        rec = self.tete_dejavu(self.lecteur_dejavu(zd))
        mse = ((rec - x[:, -1, :self.n_vu]) ** 2).mean(-1)
        met = self.tete_meteo(self.lecteur_meteo(zd)).squeeze(-1)
        return mse, met

    def entrees_mise(self, zn: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """(B, 2) : le deja-vu et la meteo normalises, tels que les lisent les
        tetes de mise. Ils sont aussi gardes dans `dernieres_previsions` (deja-vu
        en log, meteo en log de bps) pour le journal des coups et le live."""
        with torch.no_grad():
            mse, met = self.previsions(zn, x)
            vu = torch.log(mse + 1e-6)
            self.dernieres_previsions = (vu, met)
            pn = self.norme_previsions
            return torch.stack([(vu - pn[0]) / pn[1], (met - pn[2]) / pn[3]], -1).clamp(-5.0, 5.0)

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
            zlm, zsm = (zl.detach(), zs.detach()) if self.mur_mise else (zl, zs)
            ltp = torch.stack([pm(k, self.lecteur_objectif_m, self.tete_objectif_m, zl),
                               pm(k, self.lecteur_objectif_m, self.tete_objectif_m, zs)], 1)
            lsl = torch.stack([pm(k, self.lecteur_stop_m, self.tete_stop_m, zl),
                               pm(k, self.lecteur_stop_m, self.tete_stop_m, zs)], 1)
            if self.apprendre_lot:
                llo = torch.stack([pm(k, self.lecteur_lot_m, self.tete_lot_m, zlm),
                                   pm(k, self.lecteur_lot_m, self.tete_lot_m, zsm)], 1)
                if self.apprendre_risque:
                    lri = torch.stack([pm(k, self.lecteur_risque_m, self.tete_risque_m, zlm),
                                       pm(k, self.lecteur_risque_m, self.tete_risque_m, zsm)], 1)
                    if self.apprendre_allocation:
                        lal = torch.stack([pm(k, self.lecteur_allocation_m, self.tete_allocation_m, zlm),
                                           pm(k, self.lecteur_allocation_m, self.tete_allocation_m, zsm)], 1)
                        if self.apprendre_confiance:
                            lco = torch.stack([pm(k, self.lecteur_confiance_m, self.tete_confiance_m, zlm),
                                               pm(k, self.lecteur_confiance_m, self.tete_confiance_m, zsm)], 1)
                            return le, v, ltp, lsl, llo, lri, lal, lco
                        return le, v, ltp, lsl, llo, lri, lal
                    return le, v, ltp, lsl, llo, lri
                return le, v, ltp, lsl, llo
            return le, v, ltp, lsl
        la = self.tete_achat(self.mlp_achat(zn))
        lv = self.tete_vente(self.mlp_vente(zn))
        le = torch.cat([la, lv, torch.zeros_like(la)], dim=-1)
        v = self.critic(self.mlp(zn)).squeeze(-1)
        un = torch.ones_like(la)
        zl, zs = torch.cat([zn, un], -1), torch.cat([zn, -un], -1)
        zlm, zsm = (zl.detach(), zs.detach()) if self.mur_mise else (zl, zs)
        # les tetes de mise lisent en plus le deja-vu et la meteo
        zlq, zsq = zlm, zsm
        if self.apprendre_dejavu_meteo:
            f = self.entrees_mise(zn, x)
            zlq, zsq = torch.cat([zlm, f], -1), torch.cat([zsm, f], -1)
        ltp = torch.stack([self.tete_objectif(self.lecteur_objectif(zl)),
                           self.tete_objectif(self.lecteur_objectif(zs))], 1)
        lsl = torch.stack([self.tete_stop(self.lecteur_stop(zl)),
                           self.tete_stop(self.lecteur_stop(zs))], 1)
        if self.apprendre_lot:
            llo = torch.stack([self.tete_lot(self.lecteur_lot(zlq)),
                               self.tete_lot(self.lecteur_lot(zsq))], 1)
            if self.apprendre_risque:
                lri = torch.stack([self.tete_risque(self.lecteur_risque(zlq)),
                                   self.tete_risque(self.lecteur_risque(zsq))], 1)
                if self.apprendre_allocation:
                    lal = torch.stack([self.tete_allocation(self.lecteur_allocation(zlq)),
                                       self.tete_allocation(self.lecteur_allocation(zsq))], 1)
                    if self.apprendre_confiance:
                        lco = torch.stack([self.tete_confiance(self.lecteur_confiance(zlq)),
                                           self.tete_confiance(self.lecteur_confiance(zsq))], 1)
                        if self.apprendre_esperance:
                            # (B, sens, objectif, stop, [moyenne, log-variance])
                            les = torch.stack([self.tete_esperance(self.lecteur_esperance(zlm)),
                                               self.tete_esperance(self.lecteur_esperance(zsm))], 1)
                            les = les.view(len(les), 2, self.k_objectifs, self.k_stops, 2)
                            return le, v, ltp, lsl, llo, lri, lal, lco, les
                        return le, v, ltp, lsl, llo, lri, lal, lco
                    return le, v, ltp, lsl, llo, lri, lal
                return le, v, ltp, lsl, llo, lri
            return le, v, ltp, lsl, llo
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
                **({"lot": _p(self.lecteur_lot_m, self.tete_lot_m)} if self.apprendre_lot else {}),
                **({"risque": _p(self.lecteur_risque_m, self.tete_risque_m)} if self.apprendre_risque else {}),
                **({"allocation": _p(self.lecteur_allocation_m, self.tete_allocation_m)} if self.apprendre_allocation else {}),
                **({"confiance": _p(self.lecteur_confiance_m, self.tete_confiance_m)} if self.apprendre_confiance else {}),
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
            **({"lot": _p(self.lecteur_lot, self.tete_lot)} if self.apprendre_lot else {}),
            **({"risque": _p(self.lecteur_risque, self.tete_risque)} if self.apprendre_risque else {}),
            **({"allocation": _p(self.lecteur_allocation, self.tete_allocation)} if self.apprendre_allocation else {}),
            **({"confiance": _p(self.lecteur_confiance, self.tete_confiance)} if self.apprendre_confiance else {}),
            **({"esperance": _p(self.lecteur_esperance, self.tete_esperance)}
               if getattr(self, "apprendre_esperance", False) else {}),
            **({"dejavu": _p(self.lecteur_dejavu, self.tete_dejavu),
                "meteo": _p(self.lecteur_meteo, self.tete_meteo)}
               if getattr(self, "apprendre_dejavu_meteo", False) else {}),
            **({"conviction": _p(self.reseau_conviction)}
               if getattr(self, "tete_conviction", False) else {}),
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


def tp1_objectif(cfg, i: int) -> float:
    """L'objectif proche (en ATR) du choix `i` du menu des objectifs ; 0 si
    le coup sort en une fois. Voir `sortie_deux_temps` et `tp1_par_objectif`."""
    if not bool(getattr(cfg, "sortie_deux_temps", False)):
        return 0.0
    par = tuple(getattr(cfg, "tp1_par_objectif", ()) or ())
    k1 = float(par[i]) if par else float(getattr(cfg, "tp1_atr", 0.0))
    return k1 if 0.0 < k1 < float(cfg.tp_atr[i]) - 1e-9 else 0.0


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
        # LA SORTIE EN DEUX TEMPS. Voir `sortie_deux_temps`.
        deux = bool(getattr(cfg, "sortie_deux_temps", False))
        proches = {}
        if deux:
            mbe = float(getattr(cfg, "be_marge_bps", 0.0)) / 1e4
            # sortir au bid a ce niveau, glissement compris, rend au moins le
            # prix paye (l'ask d'entree) : le spread est couvert
            be = p0 / (1 - sx) * (1 + mbe) if sens > 0 else p0 / (1 + sx) * (1 - mbe)
            for k1 in sorted({tp1_objectif(cfg, i) for i in range(len(ktp))} - {0.0}):
                tp1 = p0 + k1 * a if sens > 0 else p0 - k1 * a
                t1 = np.full(len(t), H, np.int16)
                for k in range(H):
                    m = (t1 == H) & ((hi[e + k] >= tp1) if sens > 0 else (lo[e + k] <= tp1))
                    t1[m] = k
                # le stop a l'entree ne vaut qu'APRES la bougie de l'objectif proche
                t_be = np.full(len(t), H, np.int16)
                for k in range(1, H):
                    m = (t_be == H) & (t1 < k) & ((lo[e + k] <= be) if sens > 0 else (hi[e + k] >= be))
                    t_be[m] = k
                proches[k1] = (tp1, t1, t_be)
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
                r_ = (sens * (x - p0) - swap) / (ksl[j] * a)
                d_ = duree
                s_ = np.where(obj, 0, np.where(stop, 1, 2))
                k1 = tp1_objectif(cfg, i)
                if deux and k1 > 0.0:
                    tp1, t1, t_be = proches[k1]
                    # l'objectif proche AVANT le stop (meme bougie : le stop)
                    tp1_ok = (t1 < a_sl) & (t1 < H)
                    d1 = t1 + 1
                    # moitie 2 : l'objectif choisi (meme bougie que l'objectif
                    # proche : il est atteint, le prix y passe en montant),
                    # sinon le stop a l'entree, sinon le temps
                    loin = (a_tp < H) & ((a_tp == t1) | (a_tp < t_be))
                    entree = ~loin & (t_be < H)
                    b_be = e + np.minimum(t_be, H - 1)
                    if sens > 0:
                        x_be = np.minimum(be, o[b_be]) * (1 - sx)
                    else:
                        x_be = np.maximum(be, oa[b_be]) * (1 + sx)
                    x2 = np.where(loin, tp[:, i], np.where(entree, x_be, x_to))
                    d2 = np.where(loin, a_tp + 1, np.where(entree, t_be + 1, H + 1))
                    gain = 0.5 * sens * (tp1 - p0) + 0.5 * sens * (x2 - p0)
                    sw2 = frac * sw / 1e4 * p0 * (0.5 * d1 + 0.5 * d2) * float(cfg.minutes_par_barre) / 1440.0
                    r_ = np.where(tp1_ok, (gain - sw2) / (ksl[j] * a), r_)
                    d_ = np.where(tp1_ok, np.maximum(d1, d2), d_)
                    s_ = np.where(tp1_ok, np.where(loin, 0, np.where(entree, 3, 4)), s_)
                R[t, si, i, j] = r_.astype(np.float32)
                D[t, si, i, j] = d_
                S[t, si, i, j] = s_
    return R, D, S


def table_a_cout(frac, sans_cout, cout_reel, calcule):
    """Une execution coherente au cout demande : resultat, duree ET sortie.

    Le spread deplace les niveaux touches. Interpoler les R de deux
    executions peut donc inventer un gain qui ne correspond ni a la duree
    ni au type de sortie. Les extremites reutilisent leurs tables ; chaque
    cout intermediaire est simule exactement, sans garder toute la rampe
    en memoire. ``calcule`` preserve les frontieres et swaps par marche.
    """
    frac = float(frac)
    if not np.isfinite(frac) or not 0.0 <= frac <= 1.0:
        raise ValueError("la fraction de cout doit etre finie et comprise entre 0 et 1")
    if frac == 0.0:
        return sans_cout
    if frac == 1.0:
        return cout_reel
    return calcule(frac)


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


def etat_jeu(jetons, score, reste_min, cfg: JeuConfig, occupees=None,
              equite=None, pic_equite=None, marge_libre=None,
              calibration=None) -> np.ndarray:
    """Etat de jeu, avec le compte pour les runs qui le demandent.

    Les cinq premieres colonnes restent exactement celles des checkpoints
    historiques. M5_12 ajoute equite/capital, drawdown depuis le pic et marge
    libre/equite ; ce sont des donnees deja disponibles au moment de decider.
    """
    n = len(jetons)
    base = np.stack([
        jetons / float(cfg.jetons),
        np.clip(score / cfg.vie_R, -1.0, 3.0),
        np.clip(reste_min / float(cfg.barres_par_partie), 0.0, 1.0),
        np.clip((score + cfg.vie_R) / cfg.vie_R, 0.0, 3.0),
        np.zeros(n) if occupees is None else np.asarray(occupees, np.float64),
    ], 1).astype(np.float32)
    morceaux = [base]
    if bool(getattr(cfg, "observe_compte", False)):
        capital = max(float(cfg.capital), 1e-12)
        eq = np.full(n, capital) if equite is None else np.asarray(equite, np.float64)
        pic = (np.maximum(eq, capital) if pic_equite is None else
               np.maximum(np.asarray(pic_equite, np.float64), eq))
        libre = eq if marge_libre is None else np.asarray(marge_libre, np.float64)
        morceaux.append(np.stack([
            np.clip(eq / capital, 0.0, 10.0),
            np.clip((pic - eq) / np.maximum(pic, 1e-12), 0.0, 1.0),
            np.clip(libre / np.maximum(eq, 1e-12), 0.0, 1.0),
        ], 1).astype(np.float32))
    if bool(getattr(cfg, "observe_calibration", False)):
        # Colonnes deja normalisees : R recent dans [-1,1], surprise et
        # serie de pertes dans [0,1]. Une partie sans historique commence
        # neutre, sans information venue du futur.
        cal = np.zeros((n, 3), np.float32) if calibration is None else np.asarray(calibration, np.float32)
        if cal.shape != (n, 3):
            raise ValueError("calibration doit etre de forme (n, 3)")
        morceaux.append(np.column_stack([
            np.clip(cal[:, 0], -1.0, 1.0),
            np.clip(cal[:, 1], 0.0, 1.0),
            np.clip(cal[:, 2], 0.0, 1.0),
        ]).astype(np.float32))
    return np.concatenate(morceaux, axis=1)


def gouverneur_exposition(equite, pic_equite, dd_debut_jour, cfg: JeuConfig):
    """(echelle, bloque, dd) du garde-fou commun jeu/live.

    La tete conserve le choix du niveau de risque et du lot. Cette echelle
    borne seulement l'enveloppe qu'elle peut convertir en volume. Le blocage
    ne se declenche que si le drawdown GLOBAL se degrade durant la journee ;
    une journee nouvelle peut donc reprendre, mais a l'exposition reduite.
    """
    eq = max(float(equite), 0.0)
    pic = max(float(pic_equite), eq, 1e-12)
    dd = float(np.clip((pic - eq) / pic, 0.0, 1.0))
    if not bool(getattr(cfg, "gouverneur_exposition", False)):
        return 1.0, False, dd
    frein = max(0.0, float(getattr(cfg, "gouverneur_dd_frein", 0.05)))
    stop = max(frein + 1e-9, float(getattr(cfg, "gouverneur_dd_stop", 0.20)))
    minimum = float(np.clip(getattr(cfg, "gouverneur_exposition_min", 0.25), 0.0, 1.0))
    if dd <= frein:
        echelle = 1.0
    else:
        progression = min(1.0, (dd - frein) / (stop - frein))
        echelle = 1.0 - progression * (1.0 - minimum)
    # On ne bloque pas une nouvelle journee qui commence deja sous le pic :
    # elle ne peut reprendre qu'au plafond reduit. En revanche, une perte
    # supplementaire sous le seuil stop coupe les nouvelles entrees du jour.
    bloque = dd >= stop and dd > float(dd_debut_jour) + 1e-9
    return float(echelle), bool(bloque), dd


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


class DonneesMarge(np.ndarray):
    """Marge avec prix/ATR causaux pour appliquer le meme budget au jeu."""


def fraction_marge(close, atr, cfg: JeuConfig) -> np.ndarray:
    """(N, K_sl) : la marge d'un coup en fraction de l'equite, pour chaque
    stop : mise de `risque_pct` au stop, notionnel = mise / distance, marge =
    notionnel / levier."""
    dist = (np.asarray(cfg.sl_atr, np.float64)[None, :] * np.asarray(atr, np.float64)[:, None]
            / np.asarray(close, np.float64)[:, None])
    m = ((cfg.risque_pct / 100.0) / np.maximum(dist, 1e-12) / float(cfg.levier)).view(DonneesMarge)
    m.prix, m.atr = np.asarray(close), np.asarray(atr)
    return m


def _joue_historique(policy, jours: np.ndarray, Xn, R, D, S, fin_valide: int,
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
            le, v, ltp, lsl = policy.jeu(x)[:4]
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
def joue(policy, jours, Xn, R, D, S, fin_valide, cfg, device, explore,
         gen=None, collecte=False, rangs=None, marge=None, prix=None, atr=None,
         chainer=False, groupes=None, suivi=None, explore_mise=None):
    if int(getattr(cfg, "jeu_version", 1)) < 2:
        return _joue_historique(policy, jours, Xn, R, D, S, fin_valide, cfg,
                               device, explore, gen, collecte, rangs, marge)
    from jeu_rollout import joue as joue_continu
    return joue_continu(policy, jours, Xn, R, D, S, fin_valide, cfg, device,
                        explore, gen, collecte, rangs, marge, prix, atr,
                        chainer=chainer, groupes=groupes, suivi=suivi,
                        explore_mise=explore_mise)


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
            vk, rk, dk, fk = tr[k][9:13]
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


def avantage_mise(r_coups: np.ndarray, coup: np.ndarray, cfg) -> np.ndarray:
    """La note des tetes de mise : la croissance du compte, normalisee.

    `r_coups` est le resultat de chaque coup en unites de `risque_pct` du
    solde a l'entree (le R du jeu version 2), donc le coup a change le solde
    de r x risque_pct %. Sa note est log(solde apres / solde avant), plancher
    a log(0.001) pour la ruine, centree et reduite sur les coups du lot. Les
    attentes ne choisissent pas de mise : leur note vaut 0. Voir
    `note_mise_separee`.
    """
    r = np.asarray(r_coups, np.float64)
    coup = np.asarray(coup, bool)
    g = np.log(np.maximum(1.0 + r * float(cfg.risque_pct) / 100.0, 1e-3))
    out = np.zeros(len(r), np.float32)
    if coup.sum() >= 2:
        mg = float(g[coup].mean())
        sg = float(g[coup].std()) + 1e-8
        out[coup] = ((g[coup] - mg) / sg).astype(np.float32)
    return out


def fraction_esperance(moyenne, log_variance, cfg) -> np.ndarray:
    """La mise de base d'un coup, en part du compte risquee au stop : la
    fraction de Kelly (`kelly_fraction`) de moyenne / (moyenne^2 + variance),
    jamais negative. Voir `apprendre_esperance`."""
    m = np.asarray(moyenne, np.float64)
    var = np.exp(np.clip(np.asarray(log_variance, np.float64), -8.0, 6.0))
    f = m / np.maximum(m * m + var, 1e-6)
    return float(getattr(cfg, "kelly_fraction", 0.5)) * np.clip(f, 0.0, 1.0)


def entraine_esperance(policy, optims, trans, Xn, R, cfg, device, rng) -> Dict[str, float]:
    """La tete d'esperance apprend le vrai resultat des coups joues.

    Cible : le resultat net du coup en multiples du risque au stop, a cout
    plein (`R`), pour le sens, la paire d'objectifs et le stop choisis.
    Perte : vraisemblance gaussienne (moyenne et log-variance). Seul le
    groupe « esperance » apprend : le reste du modele n'est pas touche."""
    if not getattr(policy, "apprendre_esperance", False) or "esperance" not in optims:
        return {}
    lignes = [x for ep in (trans or []) for x in ep if x[3] != ATTENDRE]
    if len(lignes) < 32:
        return {}
    tt = np.asarray([x[0] for x in lignes], np.int64)
    et = np.stack([x[1] for x in lignes]).astype(np.float32)
    sens = (np.asarray([x[3] for x in lignes]) == VENDRE).astype(np.int64)
    ii = np.asarray([x[4] for x in lignes], np.int64)
    jj = np.asarray([x[5] for x in lignes], np.int64)
    y = np.asarray(R[tt, sens, ii, jj], np.float32)
    ok = np.isfinite(y)
    tt, et, sens, ii, jj, y = tt[ok], et[ok], sens[ok], ii[ok], jj[ok], np.clip(y[ok], -3.0, 10.0)
    T = lambda z, dt=torch.float32: torch.as_tensor(z, dtype=dt, device=device)
    opt = optims["esperance"]
    params = list(policy.groupes_jeu()["esperance"])
    policy.train()
    pertes = []
    for _ in range(int(getattr(cfg, "esperance_passes", 2))):
        ordre = rng.permutation(len(y))
        for d0 in range(0, len(ordre), 512):
            b = ordre[d0:d0 + 512]
            les = policy.jeu(T(observations(Xn, tt[b], et[b], int(cfg.lookback))))[8]
            ar = torch.arange(len(b), device=device)
            p = les[ar, T(sens[b], torch.long), T(ii[b], torch.long), T(jj[b], torch.long)]
            mu, lv = p[:, 0], p[:, 1].clamp(-8.0, 6.0)
            perte = 0.5 * (lv + (T(y[b]) - mu) ** 2 * torch.exp(-lv)).mean()
            opt.zero_grad(set_to_none=True)
            perte.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            pertes.append(float(perte.detach()))
    policy.eval()
    return {"perte": float(np.mean(pertes)), "n": int(len(y))}


def mesure_esperance(coups) -> Tuple[float, float]:
    """(correlation de rang entre la moyenne predite et le resultat reel en
    multiples du risque, mise de base moyenne en %) sur des coups du jeu."""
    c = [q for q in coups if len(q) > 23]
    if len(c) < 20:
        return float("nan"), float("nan")
    c = np.asarray([q[:24] for q in c], np.float64)
    prevue = c[:, 8] * c[:, 15]
    reel = c[:, 10] / np.maximum(prevue, 1e-12)
    rho = pd.Series(c[:, 21]).corr(pd.Series(reel), method="spearman")
    return float(rho), float(100 * np.mean(c[:, 23]))


def cible_meteo(h, l, c, cfg) -> np.ndarray:
    """La cible de la tete meteo : log de l'amplitude (plus haut moins plus
    bas) des `horizon_max` bougies qui suivent, en bps du cours. NaN la ou
    l'avenir manque. Voir `apprendre_dejavu_meteo`."""
    H = int(cfg.horizon_max)
    hi = pd.Series(np.asarray(h, np.float64)).rolling(H, min_periods=H).max().shift(-H).to_numpy()
    lo = pd.Series(np.asarray(l, np.float64)).rolling(H, min_periods=H).min().shift(-H).to_numpy()
    amp = (hi - lo) / np.asarray(c, np.float64) * 1e4
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(np.maximum(amp, 1e-3)).astype(np.float32)


def entraine_dejavu_meteo(policy, optims, trans, Xn, meteo, cfg, device, rng) -> Dict[str, float]:
    """Les tetes du deja-vu et meteo apprennent sur les etats d'entrainement.

    Deja-vu : reconstituer la bougie regardee (erreur quadratique). Meteo :
    l'amplitude a venir (`cible_meteo`, erreur quadratique). Le tronc est lu
    sans gradient ; seuls les groupes « dejavu » et « meteo » apprennent.
    Ensuite, la normalisation que lisent les tetes de mise est remesuree sur
    ces etats (moyenne glissante). Voir `apprendre_dejavu_meteo`."""
    if not getattr(policy, "apprendre_dejavu_meteo", False) or "dejavu" not in optims:
        return {}
    lignes = [x for ep in (trans or []) for x in ep]
    if len(lignes) < 64:
        return {}
    pick = rng.choice(len(lignes), size=min(len(lignes), int(cfg.dejavu_meteo_etats)), replace=False)
    tt = np.asarray([lignes[k][0] for k in pick], np.int64)
    et = np.stack([lignes[k][1] for k in pick]).astype(np.float32)
    ym = np.asarray(meteo[tt], np.float32)
    okm = np.isfinite(ym)
    if float(policy.norme_previsions[4]) < 0.5 and okm.any():
        # la meteo part du mouvement moyen, pas de zero (sa cible vaut ~6)
        with torch.no_grad():
            policy.tete_meteo.bias.fill_(float(ym[okm].mean()))
    ym = np.nan_to_num(ym)
    T = lambda z, dt=torch.float32: torch.as_tensor(z, dtype=dt, device=device)
    noms = ("dejavu", "meteo")
    params = [q for n in noms for q in policy.groupes_jeu()[n]]
    policy.eval()   # le tronc est seulement lu : pas de dropout
    pv, pm = [], []
    for _ in range(int(getattr(cfg, "dejavu_meteo_passes", 2))):
        ordre = rng.permutation(len(tt))
        for d0 in range(0, len(ordre), 512):
            b = ordre[d0:d0 + 512]
            x = T(observations(Xn, tt[b], et[b], int(cfg.lookback)))
            with torch.no_grad():
                zn = policy._lecture_tronc(x)
            mse, met = policy.previsions(zn, x)
            perte_vu = mse.mean()
            m = T(okm[b], torch.bool)
            perte_met = ((met - T(ym[b])) ** 2)[m].mean() if bool(m.any()) else met.sum() * 0.0
            for n in noms:
                optims[n].zero_grad(set_to_none=True)
            (perte_vu + perte_met).backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            for n in noms:
                optims[n].step()
            pv.append(float(perte_vu.detach()))
            pm.append(float(perte_met.detach()))
    # la normalisation des entrees des tetes de mise
    vus, mets = [], []
    with torch.no_grad():
        for d0 in range(0, len(tt), 1024):
            x = T(observations(Xn, tt[d0:d0 + 1024], et[d0:d0 + 1024], int(cfg.lookback)))
            mse, met = policy.previsions(policy._lecture_tronc(x), x)
            vus.append(torch.log(mse + 1e-6).cpu().numpy())
            mets.append(met.cpu().numpy())
    vus, mets = np.concatenate(vus), np.concatenate(mets)
    neuf = torch.tensor([vus.mean(), vus.std() + 1e-6, mets.mean(), mets.std() + 1e-6, 1.0],
                        dtype=policy.norme_previsions.dtype, device=policy.norme_previsions.device)
    if float(policy.norme_previsions[4]) > 0.5:
        neuf = 0.7 * policy.norme_previsions + 0.3 * neuf
    policy.norme_previsions.copy_(neuf)
    var = float(np.var(ym[okm])) if okm.any() else float("nan")
    return {"perte_vu": float(np.mean(pv)), "perte_meteo": float(np.mean(pm)),
            "r2_meteo": 1.0 - float(np.mean(pm[-max(1, len(pm) // 2):])) / max(var, 1e-9),
            "n": int(len(tt))}


def index_dejavu_meteo(cfg) -> int:
    """La place du deja-vu dans un coup du jeu (la meteo suit)."""
    return 24 if bool(getattr(cfg, "apprendre_esperance", False)) else 21


def mesure_dejavu_meteo(coups, cfg, meteo=None) -> Dict[str, object]:
    """Sur des coups du jeu : le profit factor, la mise et le nombre de coups
    par quintile du deja-vu (du plus familier au plus nouveau) et de la meteo
    (du plus calme au plus agite), plus la correlation de rang entre la meteo
    predite et l'amplitude reelle qui a suivi (`meteo`)."""
    i0 = index_dejavu_meteo(cfg)
    c = [q for q in coups if len(q) > i0 + 1]
    if len(c) < 25:
        return {}
    c = np.asarray([q[:i0 + 2] for q in c], np.float64)
    r = c[:, 10] / np.maximum(c[:, 8] * c[:, 15], 1e-12)
    mise = 100 * c[:, 8] * c[:, 15] / np.maximum(c[:, 11], 1e-12)
    out = {"n": int(len(c))}
    for nom, col in (("dejavu", c[:, i0]), ("meteo", c[:, i0 + 1])):
        pf, mi = [], []
        for part in np.array_split(np.argsort(col, kind="stable"), 5):
            g, p = r[part][r[part] > 0].sum(), -r[part][r[part] < 0].sum()
            pf.append(float(g / p) if p > 0 else float("inf"))
            mi.append(float(mise[part].mean()))
        out[nom] = {"pf": pf, "mise": mi}
    if meteo is not None:
        reel = np.asarray(meteo, np.float64)[c[:, 1].astype(np.int64)]
        ok = np.isfinite(reel)
        out["rho_meteo"] = (float(pd.Series(c[ok, i0 + 1]).corr(pd.Series(reel[ok]), method="spearman"))
                            if ok.sum() > 10 else float("nan"))
    return out


def entrees_conviction(rang, sens) -> np.ndarray:
    """(B, 2) : la conviction de l'expert pour le sens choisi (son rang
    glissant, en logit centre : 0.95 -> 0, 0.99 -> +0.8, 0.999 -> +2) et le
    sens (+1 achat, -1 vente). Voir `tete_conviction`."""
    r = np.clip(np.asarray(rang, np.float64), 0.5, 0.999)
    z = (np.log(r / (1.0 - r)) - np.log(19.0)) / 2.0
    return np.stack([z, np.where(np.asarray(sens) == 1, -1.0, 1.0)], -1).astype(np.float32)


def entraine_conviction(policy, optims, trans, R, rangs, cfg, device, rng) -> Dict[str, object]:
    """La tete de conviction apprend sur les coups joues.

    Pour chaque coup, l'issue R (cout plein) ne depend pas de la mise : la
    croissance du compte qu'aurait donnee n'importe quel multiplicateur m est
    connue. La tete monte la pente de k x log(1 + mise x m x R / k) / mise
    (k = `conviction_kelly` : demi-Kelly), moins un prix du risque `lambda`
    x (m - 1). Apres chaque lot, `lambda` monte si le multiplicateur moyen
    depasse 1 et baisse sinon : le risque moyen reste celui de la mise de
    base. Seul le groupe « conviction » apprend. Voir `tete_conviction`."""
    if not getattr(policy, "tete_conviction", False) or "conviction" not in optims or rangs is None:
        return {}
    lignes = [x for ep in (trans or []) for x in ep if x[3] != ATTENDRE]
    if len(lignes) < 64:
        return {}
    tt = np.asarray([x[0] for x in lignes], np.int64)
    sens = (np.asarray([x[3] for x in lignes]) == VENDRE).astype(np.int64)
    ii = np.asarray([x[4] for x in lignes], np.int64)
    jj = np.asarray([x[5] for x in lignes], np.int64)
    y = np.asarray(R[tt, sens, ii, jj], np.float64)
    ok = np.isfinite(y)
    tt, sens, y = tt[ok], sens[ok], np.clip(y[ok], -3.0, 10.0)
    rang = np.asarray(rangs)[tt, sens]
    c = entrees_conviction(rang, sens)
    f = float(cfg.mise_budget)
    k = float(getattr(cfg, "conviction_kelly", 0.5))
    T = lambda z, dt=torch.float32: torch.as_tensor(z, dtype=dt, device=device)
    lam = policy.conviction_lambda
    if float(policy.conviction_pret) < 0.5:
        lam.fill_(float(np.mean(y)))
        policy.conviction_pret.fill_(1.0)
    opt = optims["conviction"]
    params = list(policy.groupes_jeu()["conviction"])
    pertes = []
    for _ in range(int(getattr(cfg, "conviction_passes", 2))):
        ordre = rng.permutation(len(y))
        for d0 in range(0, len(ordre), 512):
            b = ordre[d0:d0 + 512]
            m = policy.conviction(T(c[b]))
            u = k * torch.log(torch.clamp(1.0 + f * m * T(y[b]) / k, min=1e-6)) / f - lam * (m - 1.0)
            perte = -u.mean()
            opt.zero_grad(set_to_none=True)
            perte.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                lam += float(cfg.conviction_pas_lambda) * (m.mean() - 1.0)
            pertes.append(float(perte.detach()))
    with torch.no_grad():
        m = policy.conviction(T(c)).cpu().numpy().astype(np.float64)
    out = {"perte": float(np.mean(pertes)), "n": int(len(y)), "lambda": float(lam),
           "mult_moyen": float(m.mean()), "mult_max": float(m.max())}
    for nom, a, z in (("bas", 0.0, 0.98), ("milieu", 0.98, 0.99), ("haut", 0.99, 1.01)):
        m_ = (rang >= a) & (rang < z)
        out["mult_" + nom] = float(m[m_].mean()) if m_.any() else float("nan")
        out["R_" + nom] = float(y[m_].mean()) if m_.any() else float("nan")
    return out


def courbe_conviction(policy, device):
    """(rangs, sens, multiplicateurs) : la conviction d'un modele sur une
    grille de rangs de l'expert (0.90 a 0.999), pour les deux sens."""
    r = np.concatenate([np.linspace(0.90, 0.99, 91), np.linspace(0.9905, 0.999, 18)])
    rr, ss = np.concatenate([r, r]), np.r_[np.zeros(len(r), np.int64), np.ones(len(r), np.int64)]
    with torch.no_grad():
        m = policy.conviction(torch.from_numpy(entrees_conviction(rr, ss)).to(device))
    return rr, ss, m.float().cpu().numpy().astype(np.float64)


def copie_conviction(policy, rr, ss, cible, device, pas: int = 1500) -> float:
    """La tete de conviction du MODELE FINAL apprend a reproduire `cible` (la
    mediane des dix folds sur la grille de `courbe_conviction`), en log ;
    rend l'ecart final (rapport moyen, en %).

    POURQUOI, 2026-10-03 (run m5_31) : l'expert du modele final est appris
    sur tout l'historique puis predit ce meme historique (`expert_deploiement`).
    Ses rangs y « voient » les resultats : reapprise dessus, la tete croirait
    qu'un rang de 0.99 gagne presque a coup sur, et miserait beaucoup trop en
    live, ou les rangs redeviennent honnetes. Les dix folds, eux, ont appris
    sur des rangs que leur expert n'avait pas vus."""
    c = torch.from_numpy(entrees_conviction(rr, ss)).to(device)
    y = torch.as_tensor(np.log(np.clip(cible, 1e-3, None)), dtype=torch.float32, device=device)
    params = list(policy.reseau_conviction.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    for _ in range(int(pas)):
        perte = ((torch.log(policy.conviction(c)) - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        perte.backward()
        opt.step()
    with torch.no_grad():
        ecart = (torch.log(policy.conviction(c)) - y).abs().mean()
    return float(100.0 * (math.exp(float(ecart)) - 1.0))


def index_conviction(cfg) -> int:
    """La place du multiplicateur de conviction dans un coup du jeu (le rang
    de l'expert suit)."""
    return index_budget(cfg) + 2


def mesure_conviction(coups, cfg) -> Dict[str, object]:
    """Sur des coups du jeu : multiplicateur de conviction moyen, profit
    factor et part des coups, par tranche du rang de l'expert (< 0.98,
    0.98 - 0.99, >= 0.99)."""
    ic = index_conviction(cfg)
    c = [q for q in coups if len(q) > ic + 1]
    if len(c) < 20:
        return {}
    c = np.asarray([q[:ic + 2] for q in c], np.float64)
    R = c[:, 10] / np.maximum(c[:, 8] * c[:, 15] * float(cfg.contrat), 1e-12)
    m, rang = c[:, ic], c[:, ic + 1]
    out = {"n": int(len(c)), "mult_moyen": float(m.mean())}
    for nom, a, z in (("bas", 0.0, 0.98), ("milieu", 0.98, 0.99), ("haut", 0.99, 1.01)):
        k = (rang >= a) & (rang < z)
        g, p_ = R[k][R[k] > 0].sum(), -R[k][R[k] < 0].sum()
        out[nom] = (float(m[k].mean()) if k.any() else float("nan"),
                    float(g / p_) if p_ > 0 else float("nan"), float(100 * k.mean()))
    return out


def multiplicateur_budget(cfg, k, u, z) -> float:
    """Le multiplicateur des tetes de lot, d'allocation et de confiance autour
    de la mise de base, borne a [0.5, 1.5]. Voir `budget_baisse`."""
    m = (float(cfg.multiplicateurs_lot[int(k)]) * float(cfg.multiplicateurs_allocation[int(u)])
         * float(cfg.multiplicateurs_confiance[int(z)]))
    return float(np.clip(m, 0.5, 1.5))


def index_budget(cfg) -> int:
    """La place de la mise de base dans un coup du jeu (le multiplicateur suit)."""
    return index_dejavu_meteo(cfg) + (2 if bool(getattr(cfg, "apprendre_dejavu_meteo", False)) else 0)


def mesure_budget_baisse(coups, cfg) -> Tuple[float, Dict[str, object]]:
    """La mise de base que supportent ces coups : la plus forte (en part du
    compte risquee au stop) pour laquelle, en rejouant les coups
    `budget_baisse_tirages` fois, journees dans le desordre, la pire baisse ne
    depasse `budget_baisse_pct` qu'une fois sur 10. Les multiplicateurs reels
    des tetes de mise sont rejoues. Au plus le plafond de la serie noire (et
    5 %) ; au moins 0.1 %, et 0.1 % si les coups perdent en moyenne. Voir
    `budget_baisse`."""
    if len(coups) < 30:
        return float(cfg.mise_budget), {}
    c = np.asarray(coups, dtype=np.float64)
    c = c[np.argsort(c[:, 1], kind="stable")]
    R = c[:, 10] / np.maximum(c[:, 8] * c[:, 15] * float(cfg.contrat), 1e-12)
    ib = index_budget(cfg)
    m = c[:, ib + 1] if c.shape[1] > ib + 1 else np.ones(len(c))
    if c.shape[1] > ib + 3:
        m = m * c[:, ib + 2]   # le multiplicateur de conviction
    parts = c[:, 0].astype(np.int64)
    u = np.unique(parts)
    idx = [np.flatnonzero(parts == p) for p in u]
    plaf = min(plafond_serie_noire(cfg), 0.05)
    grille = np.round(np.arange(0.001, plaf + 1e-9, 0.0005), 4)
    if not len(grille):
        grille = np.array([0.001])
    rng = np.random.default_rng(int(cfg.graine) + len(c))
    pires = []
    for _ in range(int(cfg.budget_baisse_tirages)):
        o = np.concatenate([idx[i] for i in rng.permutation(len(u))])
        le = np.cumsum(np.log(np.maximum(1.0 + grille[:, None] * (m[o] * R[o])[None, :], 1e-6)), 1)
        le = np.concatenate([np.zeros((len(grille), 1)), le], 1)
        pires.append(np.exp(-(np.maximum.accumulate(le, 1) - le).max(1)) - 1.0)
    q = np.quantile(np.asarray(pires), float(cfg.budget_baisse_quantile), axis=0)
    ok = q >= -float(cfg.budget_baisse_pct)
    # un signal qui perd en moyenne ne merite aucune mise, meme supportable
    ok &= float(np.mean(m * R)) > 0.0
    f = float(grille[ok].max()) if ok.any() else float(grille[0])
    return f, {"n": int(len(c)), "r_moyen": float(R.mean()), "mult_moyen": float(m.mean()),
               "plafond": float(plaf), "baisse_a_f": float(q[np.flatnonzero(grille == f)[0]]),
               "baisse_1pct": float(q[np.argmin(np.abs(grille - 0.01))])}


def plafond_serie_noire(cfg) -> float:
    """La part du compte qu'un coup peut risquer au plus. Voir `serie_noire`.
    1.0 (aucune limite) si la regle est coupee."""
    if not bool(getattr(cfg, "serie_noire", False)):
        return 1.0
    n = max(int(getattr(cfg, "serie_noire_n", 10)), 1)
    k = max(float(getattr(cfg, "serie_noire_k", 1.0)), 1.0)
    garde = 1.0 - float(getattr(cfg, "serie_noire_perte_max", 0.5))
    return float((1.0 - garde ** (1.0 / n)) / k)


def mesure_serie_noire(coups, cfg) -> Tuple[int, float]:
    """(pire serie de coups perdants d'affilee, perte reelle / perte au stop
    la plus forte) sur des coups du jeu version 2, dans l'ordre du temps."""
    if not len(coups):
        return 0, 1.0
    c = np.asarray([q[:16] for q in coups], np.float64)
    c = c[np.argsort(c[:, 1], kind="stable")]
    pnl = c[:, 10]
    serie = pire = 0
    for x in pnl:
        serie = serie + 1 if x < 0 else 0
        pire = max(pire, serie)
    prevue = c[:, 8] * float(cfg.contrat) * c[:, 15]
    perd = (pnl < 0) & (prevue > 0)
    k = float(np.max(-pnl[perd] / prevue[perd])) if perd.any() else 1.0
    return int(pire), max(1.0, k)


def tire_mois(jours: np.ndarray, cfg, rng) -> Tuple[np.ndarray, np.ndarray]:
    """(journees, numero du mois) : `mois_par_epoch` mois de `mois_jours`
    journees CONSECUTIVES, sans chevauchement, tires dans `jours` (les
    journees d'entrainement). Voir `note_mise_mois`."""
    n, L = len(jours), int(getattr(cfg, "mois_jours", 30))
    if n < L:
        return np.zeros((0, 2), np.int64), np.zeros(0, np.int64)
    tol = max(int(cfg.barres_par_partie) // 2, 1)
    trou = np.r_[False, (jours[1:, 0] - jours[:-1, 1]) > tol]
    debuts = [i for i in range(n - L + 1) if not trou[i + 1:i + L].any()]
    rng.shuffle(debuts)
    pris, occupe = [], np.zeros(n, bool)
    for i in debuts:
        if not occupe[i:i + L].any():
            occupe[i:i + L] = True
            pris.append(i)
        if len(pris) >= int(getattr(cfg, "mois_par_epoch", 12)):
            break
    if not pris:
        return np.zeros((0, 2), np.int64), np.zeros(0, np.int64)
    j = np.concatenate([jours[i:i + L] for i in pris])
    g = np.repeat(np.arange(len(pris)), L)
    return j, g


def avantage_mois(trans, suivi, cfg) -> np.ndarray:
    """La note du mois de chaque decision, alignee sur `avantages(trans)` :
    log(solde de fin de mois / solde avant le coup), plancher a log(0.001)
    pour la ruine, centree et reduite sur les decisions de mise ; 0 pour
    les attentes. Voir `note_mise_mois`."""
    finaux = suivi.get("final", np.zeros(0))
    chaine_de = {}
    for s_, ch in enumerate(suivi.get("chaines", [])):
        for g in ch:
            chaine_de[g] = s_
    notes = {}
    for g, k, s_, avant in suivi.get("decisions", []):
        fin = float(finaux[chaine_de.get(g, s_)])
        notes[(g, k)] = float(np.log(max(fin / max(avant, 1e-12), 1e-3)))
    out = []
    for g, tr in enumerate(trans or []):
        for k in range(len(tr)):
            out.append(notes.get((g, k), np.nan))
    out = np.asarray(out, np.float64)
    ok = np.isfinite(out)
    res = np.zeros(len(out), np.float32)
    if ok.sum() >= 2:
        res[ok] = ((out[ok] - out[ok].mean()) / (out[ok].std() + 1e-8)).astype(np.float32)
    return res


def maj_ppo(policy, optims, lignes, Xn, cfg: JeuConfig, device, rng,
            mode: str = "tout", adv_mise_externe=None):
    """Une mise a jour PPO, chaque groupe par son optimiseur.

    `mode` : "tout" (toutes les tetes), "signal" (entree, objectif, stop,
    valeur : les journees quand la mise apprend sur les mois) ou "mise" (les
    seules tetes de mise, notees par `adv_mise_externe`, la note du mois).

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
    if mode == "mise":
        # Les attentes ne choisissent aucune mise.
        if not coup.any():
            return {"kl": float("nan"), "clip": float("nan"), "H": float("nan"),
                    "Hb": float("nan"), "v": float("nan"), "n": 0, "n_coups": 0, "n_total": n}
        garde = coup.copy()
    elif n_att > place:
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
    avec_lot = bool(getattr(cfg, "apprendre_lot", False))
    avec_risque = bool(getattr(cfg, "apprendre_risque", False))
    avec_allocation = bool(getattr(cfg, "apprendre_allocation", False))
    avec_confiance = bool(getattr(cfg, "apprendre_confiance", False))
    klot = np.array([lignes[k][13] for k in sel]) if avec_lot else None
    lpk = np.array([lignes[k][14] for k in sel], np.float32) if avec_lot else None
    krisque = np.array([lignes[k][15] for k in sel]) if avec_risque else None
    lprisque = np.array([lignes[k][16] for k in sel], np.float32) if avec_risque else None
    kallocation = np.array([lignes[k][17] for k in sel]) if avec_allocation else None
    lpallocation = np.array([lignes[k][18] for k in sel], np.float32) if avec_allocation else None
    kconfiance = np.array([lignes[k][19] for k in sel]) if avec_confiance else None
    lpconfiance = np.array([lignes[k][20] for k in sel], np.float32) if avec_confiance else None
    # Cible auxiliaire : resultat immediat du coup. Elle n'entre jamais dans
    # la recompense PPO des signaux ; elle apprend seulement a l'arbitre a
    # reserver le budget fort aux contextes dont la queue de perte est faible.
    r_allocation = np.array([lignes[k][10] for k in sel], np.float32) if avec_allocation else None
    dec = (2 * int(avec_lot) + 2 * int(avec_risque) +
           2 * int(avec_allocation) + 2 * int(avec_confiance))
    adv = np.array([lignes[k][13 + dec] for k in sel], np.float32)
    ret = np.array([lignes[k][14 + dec] for k in sel], np.float32)
    w = poids[sel]
    mu = float(np.average(adv, weights=w))
    sd = float(np.sqrt(np.average((adv - mu) ** 2, weights=w))) + 1e-8
    advn = (adv - mu) / sd
    # LA NOTE DES TETES DE MISE, a part. Voir `note_mise_separee`.
    if adv_mise_externe is not None:
        adv_mise = np.asarray(adv_mise_externe, np.float32)[sel]
    elif bool(getattr(cfg, "note_mise_separee", False)):
        adv_mise = avantage_mise(np.array([lignes[k][10] for k in sel], np.float64),
                                 a_all[sel] != ATTENDRE, cfg)
    else:
        adv_mise = advn
    st = {"kl": [], "clip": [], "H": [], "Hb": [], "v": []}
    policy.eval()   # le dropout rendrait le rapport de PPO inexact
    T = lambda z, dt=torch.float32: torch.as_tensor(z, dtype=dt, device=device)
    for _ in range(cfg.ppo_epochs):
        perm = rng.permutation(len(sel))
        for d0 in range(0, len(perm), cfg.minibatch):
            b = perm[d0:d0 + cfg.minibatch]
            x = T(observations(Xn, tt[b], et[b], L))
            sortie = policy.jeu(x)
            le, v, ltp, lsl = sortie[:4]
            llo = sortie[4] if len(sortie) >= 5 else None
            lri = sortie[5] if len(sortie) >= 6 else None
            lal = sortie[6] if len(sortie) >= 7 else None
            lco = sortie[7] if len(sortie) >= 8 else None
            if int(getattr(cfg, "jeu_version", 1)) >= 2:
                # Rejouer exactement les stops permis lors de la collecte.
                ms = ((peut[b, None] >> (np.arange(len(cfg.sl_atr)) + 2)) & 1) > 0
                lsl = lsl.masked_fill(~T(ms[:, None, :], torch.bool), _NEG)
                if lri is not None:
                    # Le pourcentage de risque etait lui aussi masque selon
                    # le stop choisi, le solde, la marge et le lot minimum.
                    mr = ((peut[b, None] >> (np.arange(len(cfg.niveaux_risque_pct))
                                               + 2 + len(cfg.sl_atr))) & 1) > 0
                    lri = lri.masked_fill(~T(mr[:, None, :], torch.bool), _NEG)
            le = _masque_logits(le, T((peut[b] & 1) > 0, torch.bool),
                                T((peut[b] & 2) > 0, torch.bool))
            ab = T(a[b], torch.long)
            wb, Ab = T(w[b]), T(advn[b])
            logp = torch.log_softmax(le, -1)
            lp = logp.gather(1, ab[:, None]).squeeze(1)
            ratio = torch.exp(lp - T(lpe[b]))
            pe = -torch.min(ratio * Ab, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * Ab)
            ent = -(logp.exp() * logp).sum(-1)
            pv = (v - T(ret[b])) ** 2
            if mode == "mise":
                perte = torch.zeros((), device=device)
            else:
                perte = (pe * wb).sum() / wb.sum()
                perte = perte - cfg.entropie_entree * (ent * wb).sum() / wb.sum()
                perte = perte + cfg.vf_coef * (pv * wb).sum() / wb.sum()
            cb = ab != ATTENDRE
            hb = torch.zeros((), device=device)
            if bool(cb.any()):
                sidx = (ab[cb] == VENDRE).long()
                ar = torch.arange(int(cb.sum()), device=device)
                Am = T(adv_mise[b])[cb]
                eb = torch.zeros((), device=device)
                for lg, choisi, lp_vieux in (((ltp, i, lpi), (lsl, j, lpj))
                                             if mode != "mise" else ()):
                    lgc = torch.log_softmax(lg[cb][ar, sidx], -1)
                    ch = T(choisi[b], torch.long)[cb]
                    lpn = lgc.gather(1, ch[:, None]).squeeze(1)
                    rb = torch.exp(lpn - T(lp_vieux[b])[cb])
                    Ac = Ab[cb]
                    pb = -torch.min(rb * Ac, rb.clamp(1 - cfg.clip, 1 + cfg.clip) * Ac)
                    eb = -(lgc.exp() * lgc).sum(-1)
                    perte = perte + pb.mean() - cfg.entropie_barrieres * eb.mean()
                if mode == "signal":
                    lal = lco = llo = lri = None
                if lal is not None:
                    lgc = torch.log_softmax(lal[cb][ar, sidx], -1)
                    ch = T(kallocation[b], torch.long)[cb]
                    lpn = lgc.gather(1, ch[:, None]).squeeze(1)
                    rb = torch.exp(lpn - T(lpallocation[b])[cb])
                    pb = -torch.min(rb * Am, rb.clamp(1 - cfg.clip, 1 + cfg.clip) * Am)
                    perte = perte + pb.mean() - cfg.entropie_barrieres * (-(lgc.exp() * lgc).sum(-1)).mean()
                    # Classes ordonnees : perte nette -> budget 10 %, petit
                    # gain -> prudent, forte opportunite -> budget maximal.
                    cible = torch.bucketize(T(r_allocation[b])[cb],
                                             T(np.array([-0.25, 0.0, 0.20, 0.75], np.float32)))
                    perte = perte + float(cfg.allocation_aux_coef) * F.cross_entropy(lgc, cible)
                    hb = hb + eb.mean().detach() / 2
                if lco is not None:
                    # Le thermostat apprend par PPO, donc uniquement de la
                    # consequence economique de son propre niveau. Il ne
                    # modifie jamais la recompense des signaux d'entree.
                    lgc = torch.log_softmax(lco[cb][ar, sidx], -1)
                    ch = T(kconfiance[b], torch.long)[cb]
                    lpn = lgc.gather(1, ch[:, None]).squeeze(1)
                    rb = torch.exp(lpn - T(lpconfiance[b])[cb])
                    pb = -torch.min(rb * Am, rb.clamp(1 - cfg.clip, 1 + cfg.clip) * Am)
                    eb = -(lgc.exp() * lgc).sum(-1)
                    perte = perte + pb.mean() - cfg.entropie_barrieres * eb.mean()
                    hb = hb + eb.mean().detach() / 2
                # Un minibatch peut ne contenir que des HOLD. Dans ce cas il
                # n'existe ni sens, ni lot, ni niveau de risque a rejouer :
                # les trois tetes de coup doivent donc etre sautees ensemble.
                if llo is not None:
                    lgc = torch.log_softmax(llo[cb][ar, sidx], -1)
                    ch = T(klot[b], torch.long)[cb]
                    lpn = lgc.gather(1, ch[:, None]).squeeze(1)
                    rb = torch.exp(lpn - T(lpk[b])[cb])
                    pb = -torch.min(rb * Am, rb.clamp(1 - cfg.clip, 1 + cfg.clip) * Am)
                    eb = -(lgc.exp() * lgc).sum(-1)
                    perte = perte + pb.mean() - cfg.entropie_barrieres * eb.mean()
                if lri is not None:
                    lgc = torch.log_softmax(lri[cb][ar, sidx], -1)
                    ch = T(krisque[b], torch.long)[cb]
                    lpn = lgc.gather(1, ch[:, None]).squeeze(1)
                    rb = torch.exp(lpn - T(lprisque[b])[cb])
                    pb = -torch.min(rb * Am, rb.clamp(1 - cfg.clip, 1 + cfg.clip) * Am)
                    eb = -(lgc.exp() * lgc).sum(-1)
                    perte = perte + pb.mean() - cfg.entropie_barrieres * eb.mean()
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


def expert_deploiement(Xn, y, cfg: JeuConfig):
    """Expert du modele de deploiement, appris sur tout l'historique.

    Il ne sert jamais a mesurer un fold : les experts des folds conservent
    leurs predictions OOF. Cette version finale est donc libre d'apprendre
    toutes les periodes, comme la politique continue qui l'accompagne.
    """
    import lightgbm as lgb

    pred = np.full((len(Xn), 2), np.nan, np.float32)
    idx = np.arange(0, len(Xn), max(int(cfg.expert_pas_app), 1), dtype=np.int64)
    modeles = []
    for s_ in range(2):
        ok = idx[np.isfinite(y[idx, s_])]
        if len(ok) < 2:
            raise ValueError("Pas assez de cibles pour l'expert de deploiement")
        m = lgb.LGBMRegressor(
            n_estimators=cfg.expert_arbres, learning_rate=0.03,
            num_leaves=31, min_child_samples=400, subsample=0.7,
            subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
            n_jobs=4, verbose=-1)
        m.fit(Xn[ok], y[ok, s_])
        pred[:, s_] = m.predict(Xn).astype(np.float32)
        modeles.append(m)
    return pred, modeles


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


def cibles_expert(R: np.ndarray, cfg: Optional[JeuConfig] = None) -> np.ndarray:
    """Moyenne des R nets de TOUS les couples TP/SL de chaque sens.

    Un ancien manifeste portant explicitement un ratio positif conserve sa
    cible pour une relecture historique. Les nouveaux runs ne filtrent rien.
    """
    valeurs = R.reshape(R.shape[0], 2, -1)
    if cfg is not None and float(getattr(cfg, "expert_ratio_min", 0.0)) > 0:
        tp = np.asarray(cfg.tp_atr, np.float64)[:, None]
        sl = np.asarray(cfg.sl_atr, np.float64)[None, :]
        garde = tp / sl >= cfg.expert_ratio_min
        if not garde.any():
            raise ValueError("expert_ratio_min exclut toutes les barrieres")
        valeurs = R[:, :, garde]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y = np.nanmean(valeurs, axis=2)
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
    prep = prepare_imitation(pos, jours, fin_valide, cfg)
    for ep in range(cfg.expert_epochs):
        passe_imitation(policy, optims, prep, Xn, cfg, device, rng, ep)
    policy.eval()


def prepare_imitation(pos, jours, fin_valide, cfg, etiquette: str = ""):
    """Les minutes de l'imitation : les coups de l'expert et les attentes.
    Voir `imite_expert` ; `passe_imitation` en fait une passe."""
    suf = f"   [{etiquette}]" if etiquette else ""
    n_j = max(len(jours), 1)
    print(f"  expert  {len(pos):,} coups enseignes sur {len(jours)} journees "
          f"({len(pos)/n_j:.1f} par jour), choisis sur des minutes que "
          f"l'expert n'a pas apprises{suf}", flush=True)
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
          f"poids des coups x{cfg.expert_poids_pos:g}{suf}", flush=True)
    return {"tt": tt, "act": act, "w": w, "et": et, "etiquette": etiquette}


def passe_imitation(policy, optims, prep, Xn, cfg, device, rng, ep: int) -> None:
    """UNE passe d'imitation sur les minutes de `prepare_imitation`."""
    tt, act, w, et = prep["tt"], prep["act"], prep["w"], prep["et"]
    suf = f"   [{prep['etiquette']}]" if prep.get("etiquette") else ""
    L = int(cfg.lookback)
    T = lambda z, dt=torch.float32: torch.as_tensor(z, dtype=dt, device=device)
    policy.train()
    perm = rng.permutation(len(tt))
    pertes, justes_c, n_c = [], 0, 0
    for d0 in range(0, len(perm), 512):
        b = perm[d0:d0 + 512]
        x = T(observations(Xn, tt[b], et[b], L))
        le, _, ltp, lsl = policy.jeu(x)[:4]
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
          f"{100 * justes_c / max(n_c, 1):.1f}%{suf}", flush=True)


# ======================================================================
# LE BILAN D'UNE SERIE DE PARTIES
# ======================================================================
def bilan(scores, coups, close, atr, sp, cfg: JeuConfig,
          frac: float = 1.0, ordre=None, *, open_=None, temps=None) -> Dict[str, float]:
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
    lots = c[:, 8] if c.shape[1] > 8 else None
    risques_choisis = c[:, 9] if c.shape[1] > 9 else None
    # Depuis M5_11, le rollout conserve le PnL dollar reel de chaque lot.
    # Le multiplier par un R fixe de 10 $ faussait le solde des que l'equite
    # evoluait et pouvait faire apparaitre des drawdowns impossibles.
    pnl_reels = c[:, 10] if c.shape[1] > 10 else r * cfg.risque_dollars
    total_reel = float(pnl_reels.sum())
    compose = bool(getattr(cfg, "recompense_composee", False))
    net = r * ksl[j] * atr[t] / close[t] * 1e4
    # LE COUT REELLEMENT PAYE : pendant la rampe, une part seulement.
    cout = frac * (sp[t] + cfg.glissement_entree_bps
                   + np.where(so != 0, cfg.glissement_sortie_bps, 0.0))
    mesure_pf = pnl_reels if compose else r
    g, p = mesure_pf[mesure_pf > 0].sum(), -mesure_pf[mesure_pf < 0].sum()
    # LE DRAWDOWN — la pire baisse du compte, sommet a creux, en dollars et
    # en % du capital, les coups joues dans l'ordre du temps. Demande du
    # proprietaire, 2026-09-27, avec les comptes ci-dessous.
    # EN MULTI-MARCHES, l'indice d'une ligne ne suit pas le temps (les
    # marches sont empiles) : `ordre` donne l'instant de chaque ligne.
    # EN POSITIONS MULTIPLES, les coups se chevauchent : le compte bouge a
    # leur RESOLUTION, pas a leur ouverture.
    t_ref = (t + du.astype(np.int64) if int(getattr(cfg, "jeu_version", 1)) >= 2
             or int(getattr(cfg, "positions_max", 1)) > 1 else t)
    if ordre is not None:
        t_ref = np.minimum(t_ref, len(ordre) - 1)
    tri = np.argsort(t_ref if ordre is None else np.asarray(ordre)[t_ref], kind="stable")
    # Plusieurs positions peuvent se resoudre sur la meme bougie. Le jeu
    # realise alors leur PnL ensemble et ne laisse jamais le solde devenir
    # negatif. L'ancien cumsum additionnait chaque perte brute sans ce
    # plancher : il pouvait afficher un drawdown de -678 %, impossible dans
    # le moteur qui a produit les trades.
    sorties = t_ref[tri] if ordre is None else np.asarray(ordre)[t_ref[tri]]
    eq = [float(cfg.capital)]
    debut = 0
    while debut < len(tri):
        fin_groupe = debut + 1
        while fin_groupe < len(tri) and sorties[fin_groupe] == sorties[debut]:
            fin_groupe += 1
        eq.append(max(0.0, eq[-1] + float(pnl_reels[tri[debut:fin_groupe]].sum())))
        debut = fin_groupe
    eq = np.asarray(eq, dtype=np.float64)
    pic = np.maximum.accumulate(eq)
    dd_d = float((eq - pic).min())
    dd_p = float(np.clip(((eq - pic) / np.maximum(pic, 1e-12)).min(), -1.0, 0.0))
    # LA MISE REDUITE EN BAISSE, jouee dans le meme ordre. Voir `seuil_baisse`.
    ep = compte_prudent(r[tri], cfg)
    pic_p = np.maximum.accumulate(ep)
    # UN SEUL MARCHE : les lots, contrats et devises des indices different de
    # ceux du BTC, le compte « comme en live » ne vaut que pour le BTC seul.
    # Ce replay historique recalcule un lot a 1 % et ne connait pas encore
    # le lot reel choisi par la nouvelle tete. L'afficher ici produirait des
    # « refus par risque » fictifs ; le bilan PPO reste la source de verite.
    live = (compte_live(t, j, r, du, close, atr, cfg, sens=s, open_=open_,
                        spread=sp, sorties=so, temps=temps)
            if n and not tuple(getattr(cfg, "marches", ()))
            and not bool(getattr(cfg, "apprendre_lot", False)) else {})
    return b | live | {
        "prudent_total": float(ep[-1] - cfg.capital),
        "prudent_dd_dollars": float((ep - pic_p).min()),
        "prudent_dd_pct": float(np.clip(((ep - pic_p) / np.maximum(pic_p, 1e-12)).min(), -1.0, 0.0)),
        "gagnants": int((r > 0).sum()), "perdants": int((r < 0).sum()),
        "longs": int((s == 0).sum()), "shorts": int((s == 1).sum()),
        "win_rate": float((r > 0).mean()), "dd_dollars": dd_d, "dd_pct": dd_p,
        # Le total du bilan est le solde realisable. La somme brute des PnL
        # des positions ne peut pas faire passer un compte sous zero.
        "total_dollars": float(eq[-1] - cfg.capital),
        "net": float(net.mean()), "brut": float((net + cout).mean()),
        "cout": float(cout.mean()), "pf": float(g / p) if p > 0 else float("inf"),
        "achat": float(np.mean(s == 0)),
        **({"lot_min_choisi": float(lots.min()), "lot_moy_choisi": float(lots.mean()),
            "lot_max_choisi": float(lots.max()), "lot_min_pct": float(np.mean(lots == cfg.lot_min))}
           if lots is not None else {}),
        **({"risque_min_choisi": float(risques_choisis.min()),
            "risque_moy_choisi": float(risques_choisis.mean()),
            "risque_max_choisi": float(risques_choisis.max())}
           if risques_choisis is not None and np.isfinite(risques_choisis).any() else {}),
        "sorties": tuple(float(np.mean(so == k)) for k in range(5)),
        "tp": tuple(float(np.mean(i == k)) for k in range(len(cfg.tp_atr))),
        "sl": tuple(float(np.mean(j == k)) for k in range(len(cfg.sl_atr))),
        "duree": float(np.median(du)), "gain_R": float(r.mean()),
        "t_brut": float((net + cout).mean() / ((net + cout).std(ddof=1)
                        / np.sqrt(n) + 1e-12)) if n > 1 else float("nan"),
    }


def ecrit_trades_csv(path: str, coups, temps, cfg: JeuConfig, *, epoch: int, phase: str) -> None:
    """Ajoute les executions d'une epoch au CSV d'audit du jeu M5_11.

    Ce journal est volontairement en dollars reels : il permet de controler
    tout lot, notamment 100.00, contre l'equite et la marge qui l'ont rendu
    possible au moment de l'entree.
    """
    if not coups:
        return
    c = np.asarray(coups, dtype=float)
    if c.ndim != 2 or c.shape[1] < 17:
        return
    entree = c[:, 1].astype(np.int64)
    sortie = np.minimum(entree + c[:, 6].astype(np.int64), len(temps) - 1)
    dates = pd.Series(temps).reset_index(drop=True)
    libelle_sortie = np.array(["TP", "SL", "TEMPS", "TP1+ENTREE", "TP1+TEMPS"],
                              dtype=object)[c[:, 7].astype(np.int64)]
    frame = pd.DataFrame({
        "epoch": int(epoch), "phase": str(phase), "partie": c[:, 0].astype(np.int64),
        "entree_index": entree, "entree_time": dates.iloc[entree].astype(str).to_numpy(),
        "sortie_index": sortie, "sortie_time": dates.iloc[sortie].astype(str).to_numpy(),
        "sens": np.where(c[:, 2].astype(np.int64) == 0, "BUY", "SELL"),
        "tp_atr": np.asarray(cfg.tp_atr)[c[:, 3].astype(np.int64)],
        "tp1_atr": np.asarray([tp1_objectif(cfg, i) for i in range(len(cfg.tp_atr))])[
            c[:, 3].astype(np.int64)],
        "sl_atr": np.asarray(cfg.sl_atr)[c[:, 4].astype(np.int64)],
        "duree_bougies": c[:, 6].astype(np.int64), "sortie": libelle_sortie,
        "lot": c[:, 8], "risque_pct_equite": c[:, 9], "pnl_reel_usd": c[:, 10],
        "equite_avant_usd": c[:, 11], "marge_libre_avant_usd": c[:, 12],
        "marge_requise_usd": c[:, 13], "prix_entree": c[:, 14],
        "distance_stop": c[:, 15], "lot_plafond_selection": c[:, 16],
        "gouverneur_exposition": c[:, 17] if c.shape[1] > 17 else 1.0,
        "drawdown_entree": c[:, 18] if c.shape[1] > 18 else 0.0,
        "allocation_budget": c[:, 19] if c.shape[1] > 19 else 1.0,
        "thermostat_confiance": c[:, 20] if c.shape[1] > 20 else 1.0,
        "esperance_moyenne": c[:, 21] if c.shape[1] > 23 else np.nan,
        "esperance_ecart": c[:, 22] if c.shape[1] > 23 else np.nan,
        "mise_base_pct": 100 * c[:, 23] if c.shape[1] > 23 else np.nan,
        "dejavu_log_erreur": (c[:, index_dejavu_meteo(cfg)]
                              if getattr(cfg, "apprendre_dejavu_meteo", False)
                              and c.shape[1] > index_dejavu_meteo(cfg) + 1 else np.nan),
        "meteo_log_amplitude_bps": (c[:, index_dejavu_meteo(cfg) + 1]
                                    if getattr(cfg, "apprendre_dejavu_meteo", False)
                                    and c.shape[1] > index_dejavu_meteo(cfg) + 1 else np.nan),
        "mise_budget_pct": (100 * c[:, index_budget(cfg)]
                            if c.shape[1] > index_budget(cfg) + 1 else np.nan),
        "multiplicateur_mise": (c[:, index_budget(cfg) + 1]
                                if c.shape[1] > index_budget(cfg) + 1 else np.nan),
        "multiplicateur_conviction": (c[:, index_conviction(cfg)]
                                      if c.shape[1] > index_conviction(cfg) + 1 else np.nan),
        "rang_expert": (c[:, index_conviction(cfg) + 1]
                        if c.shape[1] > index_conviction(cfg) + 1 else np.nan),
        "equite_apres_usd": np.maximum(0.0, c[:, 11] + c[:, 10]),
        "score_R": c[:, 5],
    })
    # Un CSV historique peut ne pas posseder les deux colonnes du gouverneur.
    # On le preserve et on ecrit alors un journal homonyme separe, plutot que
    # d'ajouter des lignes decalees sous son ancien en-tete.
    cible = path
    if os.path.exists(cible):
        ancien = pd.read_csv(cible, nrows=0).columns
        if "gouverneur_exposition" not in ancien:
            racine, extension = os.path.splitext(cible)
            cible = racine + "_gouverneur" + extension
    frame.to_csv(cible, mode="a", index=False, header=not os.path.exists(cible), encoding="utf-8")


def compte_live(t, j, r, du, close, atr, cfg: JeuConfig, **kwargs) -> Dict[str, float]:
    if int(getattr(cfg, "jeu_version", 1)) < 2:
        return _compte_live_historique(t, j, r, du, close, atr, cfg)
    from jeu_compte import compte_live as compte_verifie
    return compte_verifie(t, j, r, du, close, atr, cfg, **kwargs)


def _compte_live_historique(t, j, r, du, close, atr, cfg: JeuConfig) -> Dict[str, float]:
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
            E = max(0.0, E + pnl)
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
        E = max(0.0, E + pnl)
        courbe.append(E)
    courbe = np.asarray(courbe)
    pic = np.maximum.accumulate(courbe)
    return {"live_total": float(E - cfg.capital),
            "live_dd_dollars": float((courbe - pic).min()),
            "live_dd_pct": float(np.clip(((courbe - pic) / np.maximum(pic, 1e-12)).min(), -1.0, 0.0)),
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
        eq.append(max(0.0, eq[-1] + float(r) * cfg.risque_dollars * m))
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
            + (f"  |  lots choisis {b['lot_min_choisi']:.2f}/{b['lot_moy_choisi']:.2f}/"
               f"{b['lot_max_choisi']:.2f} (min {100 * b['lot_min_pct']:.0f} %)"
               if 'lot_moy_choisi' in b else '')
            + (f"  |  risque choisi {b['risque_min_choisi']:.0f}/"
               f"{b['risque_moy_choisi']:.1f}/{b['risque_max_choisi']:.0f} % de l'equite"
               if 'risque_moy_choisi' in b else '')
            + (f"  |  COMME EN LIVE ({cfg.risque_pct:g} % du solde, lots de "
               f"{cfg.lot_min:g}, levier 1:{cfg.levier:.0f}) : total {b['live_total']:+.2f} $, "
               f"baisse du solde {100 * b['live_dd_pct']:+.1f} %, risque nominal moyen "
               f"{100 * b['live_risque_moy']:.2f} % (max {100 * b['live_risque_max']:.1f} %)"
               + (f", {b['live_marge']} refuses faute de marge" if b.get("live_marge") else "")
               + (f", {b['live_risque_refuses']} refuses par le risque" if b.get("live_risque_refuses") else "")
               + (f", {b['live_positions']} refuses par la capacite" if b.get("live_positions") else "")
               + (f", drawdown equity aux clotures {100 * b['live_equity_dd_pct']:+.1f} %"
                  if b.get("live_equity_disponible") else "")
               if "live_total" in b else ""))


def ligne_bilan(b, cfg: JeuConfig) -> str:
    if b["coups"] == 0:
        return (f"parties {b['parties']}  AUCUN COUP JOUE  score +0.000 R/partie")
    dollars_partie = b["total_dollars"] / max(b["parties"], 1)
    return (f"score {b['score']:+.3f} R/partie ({dollars_partie:+.2f}$)  "
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
def purge_barres(cfg):
    fixe = int(getattr(cfg, "purge_barres_fixes", 0))
    if fixe > 0:
        return fixe
    unite = cfg.barres_par_partie if int(getattr(cfg, "jeu_version", 1)) < 2 else 7 * cfg.barres_par_jour
    return int(cfg.horizon_max + cfg.lookback + cfg.purge_semaines * unite)


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


def _jours_par_regime(jours: np.ndarray, N: int, n_regimes: int):
    """Decoupe les parties entieres en regimes chronologiques non melanges."""
    bornes = np.linspace(0, N, n_regimes + 1).astype(np.int64)
    out = []
    for a, z in zip(bornes[:-1], bornes[1:]):
        g = jours[(jours[:, 0] >= a) & (jours[:, 1] <= z)]
        if len(g):
            out.append(g)
    if not out:
        raise ValueError("Aucune partie entiere pour le curriculum de deploiement")
    return out


def _rejoue_regimes(regimes, courant: int, n: int, rng):
    """Nouvelles parties du regime courant + rappel egal de son passe.

    Cela evite qu'un passage sur une phase recente efface les tendances,
    crises ou ranges deja appris. Aucun tirage ne vient d'un regime futur.
    """
    groupes = regimes[:courant + 1]
    par_groupe = max(1, int(np.ceil(n / len(groupes))))
    tires = []
    for g in groupes:
        tires.append(g[rng.choice(len(g), size=par_groupe,
                                  replace=len(g) < par_groupe)])
    out = np.concatenate(tires, axis=0)
    return out[rng.permutation(len(out))[:n]]


def distille_decisions_oof(policy, optims, transitions, Xn, cfg, device, rng):
    """Transfere les decisions des dix professeurs dans un seul eleve.

    `transitions` ne contient que les trajectoires de TEST des folds. Pour
    chaque date, le professeur qui donne le label ne l'a donc jamais utilisee
    pour apprendre. L'eleve peut assimiler tous les regimes sans invalider le
    resultat croise deja mesure. Les cinq tetes de decision sont copiees :
    entree, TP, SL, lot et risque.
    """
    lignes = [x for episode in transitions for x in episode]
    if not lignes:
        print("  OOF distillation ignoree : aucune decision de test", flush=True)
        return
    # Toutes les entrees sont conservees ; les HOLD sont echantillonnes au
    # meme pas que l'imitation de l'expert afin de ne pas noyer les trades.
    actions = np.asarray([x[3] for x in lignes], np.int64)
    garde = actions != ATTENDRE
    attente = np.flatnonzero(~garde)
    if len(attente):
        garde[attente[::max(int(cfg.expert_pas_neg), 1)]] = True
    lignes = [x for x, ok in zip(lignes, garde) if ok]
    tt = np.asarray([x[0] for x in lignes], np.int64)
    et = np.stack([x[1] for x in lignes]).astype(np.float32)
    act = np.asarray([x[3] for x in lignes], np.int64)
    tp = np.asarray([x[4] for x in lignes], np.int64)
    sl = np.asarray([x[5] for x in lignes], np.int64)
    lot = np.asarray([x[13] for x in lignes], np.int64)
    risque = np.asarray([x[15] for x in lignes], np.int64)
    confiance = np.asarray([x[19] for x in lignes], np.int64)
    poids = np.where(act == ATTENDRE, 1.0, float(cfg.expert_poids_pos)).astype(np.float32)
    T = lambda z, dt=torch.float32: torch.as_tensor(z, dtype=dt, device=device)
    print(f"  OOF distillation : {len(lignes):,} decisions de test croisees, "
          f"{int((act != ATTENDRE).sum()):,} trades, {cfg.distillation_oof_epochs} passe(s)",
          flush=True)
    policy.train()
    for ep in range(int(cfg.distillation_oof_epochs)):
        pertes = []
        ordre = rng.permutation(len(lignes))
        for d0 in range(0, len(lignes), 512):
            b = ordre[d0:d0 + 512]
            x = T(observations(Xn, tt[b], et[b], int(cfg.lookback)))
            sortie = policy.jeu(x)
            le, _v, ltp, lsl = sortie[:4]
            wb, ab = T(poids[b]), T(act[b], torch.long)
            loss = (F.cross_entropy(le, ab, reduction="none") * wb).sum() / wb.sum()
            cb = act[b] != ATTENDRE
            if bool(cb.any()):
                ar = torch.arange(int(cb.sum()), device=device)
                sens = (ab[cb] == VENDRE).long()
                for logits, cible in ((ltp, tp[b]), (lsl, sl[b])):
                    loss = loss + F.cross_entropy(logits[cb][ar, sens],
                                                   T(cible[cb], torch.long))
                if len(sortie) >= 5:
                    loss = loss + F.cross_entropy(sortie[4][cb][ar, sens],
                                                   T(lot[b][cb], torch.long))
                if len(sortie) >= 6:
                    loss = loss + F.cross_entropy(sortie[5][cb][ar, sens],
                                                   T(risque[b][cb], torch.long))
                if len(sortie) >= 8:
                    loss = loss + F.cross_entropy(sortie[7][cb][ar, sens],
                                                   T(confiance[b][cb], torch.long))
            _pas(policy, optims, loss, cfg)
            pertes.append(float(loss.detach()))
        print(f"  OOF distillation passe {ep + 1}/{cfg.distillation_oof_epochs} "
              f"perte {np.mean(pertes):.4f}", flush=True)
    policy.eval()


def entraine_deploiement_continu_ancien(cfg, d, X, y_ex, toutes, o, h, l, c, sp, atr,
                                 R0, D0, S0, R1, D1, S1, marge, candidats,
                                 transitions_oof, device, rng):
    """Ancienne recette mono-fold, desactivee.

    Les folds sont un banc de mesure, jamais une chaine de poids. Une fois
    cette mesure terminee, le champion de validation devient la graine d'un
    unique modele qui avance chronologiquement, sans remise a zero, tout en
    rejouant les regimes passes. C'est cette phase -- et non un dernier fit
    neuf -- qui transmet l'apprentissage des dix periodes au live.
    """
    raise RuntimeError("La recette DEPLOY mono-fold est desactivee : utiliser l'ensemble des 10 folds")
    if not candidats:
        print("  DEPLOY ignore : aucun checkpoint de fold disponible", flush=True)
        return None
    # Le test ne participe jamais a ce choix. Les candidats sont classes par
    # leur profit de VALIDATION, la zone reservee du fold.
    champion = max(candidats, key=lambda x: x["profit_jour"])
    print("\n" + "=" * 70)
    print(f"  DEPLOY CONTINU : graine fold {champion['bloc']} epoch {champion['epoch']} "
          f"({champion['profit_jour']:+.2f}$/jour en validation)", flush=True)
    print("  parcours chronologique des regimes, avec rappel equilibre du passe ; "
          "aucun reset entre regimes", flush=True)
    N = len(d)
    st_m = {"mean": X.astype(np.float64).mean(0).astype(np.float32),
            "std": X.astype(np.float64).std(0).astype(np.float32)}
    Xn = safe_normalize(X, st_m).astype(np.float32)
    t_ex = time.time()
    pred, modeles_expert = expert_deploiement(Xn, y_ex, cfg)
    FE = features_expert(pred, fenetre=10_000 // int(cfg.minutes_par_barre))
    rangs_ex = FE[:, 2:4].copy()
    st_e = {"mean": FE.astype(np.float64).mean(0), "std": FE.astype(np.float64).std(0)}
    FEn = np.clip((FE - st_e["mean"]) / (st_e["std"] + 1e-8), -5.0, 5.0).astype(np.float32)
    Xk = np.concatenate([Xn, FEn[:, :cfg.n_expert]], axis=1)
    print(f"  expert final sur tout l'historique : {time.time() - t_ex:.0f} s", flush=True)

    from jeu_artifacts import sauvegarde_pipeline_bloc, empreinte_fichier
    pipeline_path = f"pipeline_{cfg.prefixe}_deploy.json"
    sauvegarde_pipeline_bloc(
        pipeline_path, cfg=cfg, donnees=d, colonnes=colonnes_jeu(cfg),
        normalisation_marche=st_m, normalisation_expert=st_e,
        modeles_expert=modeles_expert, predictions_expert=pred,
        meta={"type": "deploiement_continu", "graine_fold": champion["bloc"],
              "graine_epoch": champion["epoch"],
              "selection": "profit_validation", "regimes": int(cfg.n_blocs),
              "rejeu": "egal_des_regimes_precedents"})
    pipeline_ref = {"fichier": pipeline_path, "sha256": empreinte_fichier(pipeline_path)}

    ck = torch.load(champion["path"], map_location=device, weights_only=False)
    policy = PolitiqueJeu(cfg).to(device)
    policy.load_state_dict(ck["modele"])
    optims = optimiseurs(policy, cfg)
    fin = np.full(N, N, np.int64)
    pos = coups_expert_predits(toutes, pred, D1, fin, cfg)
    imite_expert(policy, optims, pos, toutes, Xk, fin, cfg, device, rng)
    distille_decisions_oof(policy, optims, transitions_oof, Xk, cfg, device, rng)
    regimes = _jours_par_regime(toutes, N, int(cfg.n_blocs))
    epochs_candidats = [int(x["epoch"]) for x in candidats if x["epoch"] is not None]
    n_ep = int(cfg.epochs_deploiement) if int(cfg.epochs_deploiement) > 0 else \
        int(np.median(epochs_candidats)) if epochs_candidats else int(cfg.epochs)
    n_ep = max(len(regimes), n_ep, 1)
    marge_d = max(240 // int(cfg.minutes_par_barre), cfg.barres_par_partie // 6)
    for epoch in range(1, n_ep + 1):
        courant = min(len(regimes) - 1, (epoch - 1) * len(regimes) // n_ep)
        n_jours = min(cfg.parties_par_epoch, sum(len(x) for x in regimes[:courant + 1]))
        jours = _rejoue_regimes(regimes, courant, n_jours, rng)
        frac = min(1.0, max(0.0, (epoch - 1) / cfg.rampe_cout)) if cfg.rampe_cout > 0 else 1.0
        Rf, Df, Sf = table_a_cout(frac, (R0, D0, S0), (R1, D1, S1),
                                  lambda f: table_coups(o, h, l, sp, atr, cfg, f))
        _sc, _cp, tr = joue(policy, departs_tires(jours, rng, marge_d), Xk,
                             Rf, Df, Sf, fin, cfg, device, explore=True,
                             collecte=True, rangs=rangs_ex, marge=marge)
        stats = maj_ppo(policy, optims, avantages(tr, cfg), Xk, cfg, device, rng)
        del Rf, Df, Sf
        print(f"DEPLOY {epoch:03d}/{n_ep:03d}  regime {courant + 1}/{len(regimes)}  "
              f"cout {100 * frac:.0f}%  {len(jours)} parties  "
              f"PPO kl {stats['kl']:+.4f}  v {stats['v']:.4f}",
              flush=True)

    etat = {"modele": policy.state_dict(), "config": asdict(cfg), "epoch": n_ep,
            "type": "deploiement_continu", "pipeline": pipeline_ref,
            "graine": {"bloc": champion["bloc"], "epoch": champion["epoch"],
                       "profit_jour_validation": champion["profit_jour"]},
            "regimes": len(regimes), "rejeu": "egal_des_regimes_precedents",
            "distillation": "decisions_oof_des_10_folds"}
    path = f"deploy_{cfg.prefixe}.pth"
    torch.save(etat, path)
    with open(f"deploy_{cfg.prefixe}.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in etat.items() if k != "modele"}, fh,
                  indent=1, default=str)
    print(f"  DEPLOY pret : {path} + {pipeline_path}", flush=True)
    return path


def _compacte_decisions_enseignant(transitions, cfg, rng, maximum_attentes=20_000):
    """Conserve tous les coups et un echantillon equilibre des attentes.

    Un professeur rejoue tout le cache, soit plusieurs centaines de milliers
    de decisions. Garder chaque attente des dix professeurs consommerait de
    la memoire sans apporter dix fois plus de signal. Tous les trades restent
    presents ; les attentes sont tirees sur toute la periode et plafonnees.
    """
    lignes = [x for episode in (transitions or []) for x in episode]
    if not lignes:
        return None
    action = np.asarray([x[3] for x in lignes], dtype=np.int64)
    coups = np.flatnonzero(action != ATTENDRE)
    attentes = np.flatnonzero(action == ATTENDRE)
    n_att = min(len(attentes), max(int(maximum_attentes), len(coups)))
    if len(attentes) > n_att:
        attentes = rng.choice(attentes, size=n_att, replace=False)
    garde = np.sort(np.concatenate([coups, attentes]))
    lignes = [lignes[i] for i in garde]
    action = action[garde]
    return {
        "t": np.asarray([x[0] for x in lignes], dtype=np.int64),
        "etat": np.stack([x[1] for x in lignes]).astype(np.float32),
        "action": action,
        "tp": np.asarray([x[4] for x in lignes], dtype=np.int64),
        "sl": np.asarray([x[5] for x in lignes], dtype=np.int64),
        "lot": np.asarray([x[13] for x in lignes], dtype=np.int64),
        "risque": np.asarray([x[15] for x in lignes], dtype=np.int64),
        "allocation": np.asarray([x[17] for x in lignes], dtype=np.int64),
        "confiance": np.asarray([x[19] for x in lignes], dtype=np.int64),
    }


def distille_ensemble_folds(policy, optims, paquets, Xn, cfg, device, rng):
    """Imite a parts egales les dix meilleurs folds dans le modele final."""
    paquets = [p for p in paquets if p is not None and len(p["t"])]
    if not paquets:
        raise ValueError("Aucune decision des professeurs a distiller")
    noms = ("t", "etat", "action", "tp", "sl", "lot", "risque", "allocation", "confiance")
    data = {nom: np.concatenate([p[nom] for p in paquets], axis=0) for nom in noms}
    act = data["action"]
    poids = np.where(act == ATTENDRE, 1.0, float(cfg.expert_poids_pos)).astype(np.float32)
    T = lambda z, dt=torch.float32: torch.as_tensor(z, dtype=dt, device=device)
    n_passes = max(1, int(cfg.distillation_oof_epochs))
    print(f"  distillation ensemble : {len(act):,} decisions equilibrees de "
          f"{len(paquets)} meilleurs folds, {int((act != ATTENDRE).sum()):,} coups, "
          f"{n_passes} passe(s)", flush=True)
    policy.train()
    for ep in range(n_passes):
        pertes = []
        # Un ordre unique et melange : aucun fold final ne prend le dessus.
        ordre = rng.permutation(len(act))
        for d0 in range(0, len(ordre), 512):
            b = ordre[d0:d0 + 512]
            x = T(observations(Xn, data["t"][b], data["etat"][b], int(cfg.lookback)))
            sortie = policy.jeu(x)
            le, _v, ltp, lsl = sortie[:4]
            wb, ab = T(poids[b]), T(act[b], torch.long)
            loss = (F.cross_entropy(le, ab, reduction="none") * wb).sum() / wb.sum()
            cb = act[b] != ATTENDRE
            if bool(cb.any()):
                ar = torch.arange(int(cb.sum()), device=device)
                sens = (ab[cb] == VENDRE).long()
                for logits, cible in ((ltp, data["tp"][b]), (lsl, data["sl"][b])):
                    loss = loss + F.cross_entropy(logits[cb][ar, sens],
                                                   T(cible[cb], torch.long))
                loss = loss + F.cross_entropy(sortie[4][cb][ar, sens],
                                               T(data["lot"][b][cb], torch.long))
                loss = loss + F.cross_entropy(sortie[5][cb][ar, sens],
                                               T(data["risque"][b][cb], torch.long))
                if len(sortie) >= 7:
                    loss = loss + F.cross_entropy(sortie[6][cb][ar, sens],
                                                   T(data["allocation"][b][cb], torch.long))
                if len(sortie) >= 8:
                    loss = loss + F.cross_entropy(sortie[7][cb][ar, sens],
                                                   T(data["confiance"][b][cb], torch.long))
            _pas(policy, optims, loss, cfg)
            pertes.append(float(loss.detach()))
        print(f"  distillation ensemble passe {ep + 1}/{n_passes} "
              f"perte {np.mean(pertes):.4f}", flush=True)
    policy.eval()


def entraine_deploiement_continu(cfg, d, X, y_ex, toutes, o, h, l, c, sp, atr,
                                 R0, D0, S0, R1, D1, S1, marge, candidats,
                                 transitions_oof, device, rng):
    """Construit le deploy par ensemble des dix folds, jamais par champion.

    Les mesures de validation/test restent celles des folds independants. La
    phase qui suit ne les selectionne ni ne les remplace : elle utilise les
    dix meilleurs checkpoints comme professeurs, chacun uniquement sur SA
    validation -- la periode ou il a gagne sa place. L'eleve apprend donc les
    meilleurs coups valides, sans diluer ce signal sur des dates etrangeres.
    """
    del R0, D0, S0, transitions_oof
    if len(candidats) != int(cfg.n_blocs):
        raise ValueError(f"DEPLOY ensemble : {len(candidats)}/{cfg.n_blocs} checkpoints disponibles")
    candidats = sorted(candidats, key=lambda x: int(x["bloc"]))
    manquants = [x["path"] for x in candidats if not os.path.isfile(x["path"])]
    if manquants:
        raise FileNotFoundError("Checkpoint de fold absent : " + ", ".join(manquants))
    print("\n" + "=" * 70)
    print("  DEPLOY ENSEMBLE : les 10 meilleurs folds enseignent un eleve neuf", flush=True)
    print("  chaque professeur enseigne uniquement sa periode de validation ; "
          "les validations et tests deja ecrits ne sont pas modifies.", flush=True)
    N = len(d)
    st_m = {"mean": X.astype(np.float64).mean(0).astype(np.float32),
            "std": X.astype(np.float64).std(0).astype(np.float32)}
    Xn = safe_normalize(X, st_m).astype(np.float32)
    t_ex = time.time()
    pred, modeles_expert = expert_deploiement(Xn, cibles_expert(R1, cfg), cfg)
    FE = features_expert(pred, fenetre=10_000 // int(cfg.minutes_par_barre))
    rangs_ex = FE[:, 2:4].copy()
    st_e = {"mean": FE.astype(np.float64).mean(0), "std": FE.astype(np.float64).std(0)}
    FEn = np.clip((FE - st_e["mean"]) / (st_e["std"] + 1e-8), -5.0, 5.0).astype(np.float32)
    X_final = np.concatenate([Xn, FEn[:, :cfg.n_expert]], axis=1)
    print(f"  pipeline final appris sur tout l'historique : {time.time() - t_ex:.0f} s", flush=True)

    from jeu_artifacts import (charge_pipeline_bloc, empreinte_fichier,
                               predit_pipeline_bloc, sauvegarde_pipeline_bloc)
    pipeline_path = f"pipeline_{cfg.prefixe}_deploy_ensemble_validation_ddsafe.json"
    if os.path.exists(pipeline_path):
        raise FileExistsError(f"{pipeline_path} existe deja : il est volontairement conserve")
    sauvegarde_pipeline_bloc(
        pipeline_path, cfg=cfg, donnees=d, colonnes=colonnes_jeu(cfg),
        normalisation_marche=st_m, normalisation_expert=st_e,
        modeles_expert=modeles_expert, predictions_expert=pred,
        meta={"type": "deploiement_ensemble_10_folds", "folds": [int(x["bloc"]) for x in candidats],
              "professeurs": "meilleur_checkpoint_par_fold_sur_sa_validation",
              "selection": "aucune_selection_de_champion", "ppo": "une_passe_complete"})
    pipeline_ref = {"fichier": pipeline_path, "sha256": empreinte_fichier(pipeline_path)}

    fin = np.full(N, N, np.int64)
    paquets = []
    courbes_conviction = []
    for candidat in candidats:
        bloc = int(candidat["bloc"])
        chemin_pipe = f"pipeline_{cfg.prefixe}_bloc{bloc}.json"
        pipe = charge_pipeline_bloc(chemin_pipe, donnees=d, colonnes=colonnes_jeu(cfg))
        validation = pipe.get("meta", {}).get("validation")
        if not isinstance(validation, list) or len(validation) != 2:
            raise ValueError(f"Fenetre de validation absente du pipeline du fold {bloc}")
        va0, va1 = (int(validation[0]), int(validation[1]))
        jours_prof = np.asarray([j for j in toutes if int(j[0]) >= va0 and int(j[1]) <= va1],
                                dtype=np.int64).reshape(-1, 2)
        if not len(jours_prof):
            raise ValueError(f"Aucune partie complete dans la validation du fold {bloc}")
        X_prof, rangs_prof, _ = predit_pipeline_bloc(pipe, d)
        professeur = PolitiqueJeu(cfg).to(device)
        etat_prof = torch.load(candidat["path"], map_location=device, weights_only=False)
        professeur.load_state_dict(etat_prof["modele"])
        if getattr(professeur, "tete_conviction", False):
            rr_c, ss_c, m_c = courbe_conviction(professeur, device)
            courbes_conviction.append(m_c)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(cfg.graine) + bloc)
        fin_validation = np.full(N, va1, np.int64)
        _score, coups, transitions = joue(professeur, jours_prof, X_prof, R1, D1, S1, fin_validation,
                                           cfg, device, explore=False, gen=gen,
                                           collecte=True, rangs=rangs_prof, marge=marge)
        paquet = _compacte_decisions_enseignant(transitions, cfg, rng)
        paquets.append(paquet)
        print(f"  professeur fold {bloc:2d} : {len(coups):,} coups sur sa validation "
              f"({len(jours_prof)} parties), {len(paquet['t']):,} decisions retenues", flush=True)
        del professeur, transitions, X_prof, rangs_prof, fin_validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    policy = PolitiqueJeu(cfg).to(device)
    optims = optimiseurs(policy, cfg)
    pos = coups_expert_predits(toutes, pred, D1, fin, cfg)
    imite_expert(policy, optims, pos, toutes, X_final, fin, cfg, device, rng)
    # LES TETES DU DEJA-VU ET METEO de l'eleve apprennent avant qu'il n'imite
    # les mises des professeurs, puis a chaque passe PPO.
    M_final = cible_meteo(h, l, c, cfg)
    etats_prof = [[(int(t_), e_) for t_, e_ in zip(p["t"], p["etat"])] for p in paquets]
    entraine_dejavu_meteo(policy, optims, etats_prof, X_final, M_final, cfg, device, rng)
    distille_ensemble_folds(policy, optims, paquets, X_final, cfg, device, rng)
    # LA CONVICTION DU MODELE FINAL : la mediane des dix folds, copiee, puis
    # plus apprise. Voir `copie_conviction`.
    conviction_mediane = None
    if getattr(policy, "tete_conviction", False) and courbes_conviction:
        conviction_mediane = np.median(np.stack(courbes_conviction), 0)
        ecart = copie_conviction(policy, rr_c, ss_c, conviction_mediane, device)
        _tr = lambda a, z: float(np.mean(conviction_mediane[(rr_c >= a) & (rr_c < z)]))
        print(f"  conviction du modele final : mediane de {len(courbes_conviction)} folds, "
              f"x{_tr(0.0, 0.98):.2f} (rang < 0.98) / x{_tr(0.98, 0.99):.2f} (0.98-0.99) / "
              f"x{_tr(0.99, 1.01):.2f} (>= 0.99) ; copiee a {ecart:.1f} % pres", flush=True)

    # Une passe PPO qui couvre toutes les journees : contrairement a l'ancien
    # curriculum, aucun regime final ni fold n'a de privilege.
    marge_d = max(240 // int(cfg.minutes_par_barre), cfg.barres_par_partie // 6)
    fragments = np.array_split(toutes, max(1, int(np.ceil(len(toutes) / cfg.parties_par_epoch))))
    for numero, jours in enumerate(fragments, 1):
        _sc, _cp, tr = joue(policy, departs_tires(jours, rng, marge_d), X_final,
                             R1, D1, S1, fin, cfg, device, explore=True,
                             collecte=True, rangs=rangs_ex, marge=marge)
        stats = maj_ppo(policy, optims, avantages(tr, cfg), X_final, cfg, device, rng)
        entraine_esperance(policy, optims, tr, X_final, R1, cfg, device, rng)
        entraine_dejavu_meteo(policy, optims, tr, X_final, M_final, cfg, device, rng)
        print(f"DEPLOY ENSEMBLE PPO {numero:02d}/{len(fragments):02d}  "
              f"{len(jours)} parties  kl {stats['kl']:+.4f}  v {stats['v']:.4f}", flush=True)

    etat = {"modele": policy.state_dict(), "config": asdict(cfg), "epoch": len(fragments),
            "type": "deploiement_ensemble_10_folds", "pipeline": pipeline_ref,
            "folds": [{"bloc": int(x["bloc"]), "epoch": int(x["epoch"])} for x in candidats],
            "distillation": "10_professeurs_sur_leur_validation",
            "conviction_mediane_folds": (None if conviction_mediane is None
                                         else [float(x) for x in conviction_mediane]),
            "ppo": "une_passe_complete_toutes_les_parties"}
    path = f"deploy_{cfg.prefixe}_ensemble_validation_ddsafe.pth"
    torch.save(etat, path)
    with open(f"deploy_{cfg.prefixe}_ensemble_validation_ddsafe.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in etat.items() if k != "modele"}, fh, indent=1, default=str)
    print(f"  DEPLOY ENSEMBLE pret : {path} + {pipeline_path}", flush=True)
    return path


def config_serie_noire_folds(cfg, candidats):
    """La serie noire du modele final : la plus prudente des dix modeles de
    folds (la plus longue serie, la plus forte perte reelle)."""
    if not bool(getattr(cfg, "serie_noire", False)):
        return cfg
    ns, ks = [], []
    for x in candidats:
        if os.path.isfile(x["path"]):
            c = torch.load(x["path"], map_location="cpu", weights_only=False).get("config", {})
            ns.append(int(c.get("serie_noire_n", cfg.serie_noire_n)))
            ks.append(float(c.get("serie_noire_k", cfg.serie_noire_k)))
    if not ns:
        return cfg
    cfg = replace(cfg, serie_noire_n=max(ns), serie_noire_k=max(ks))
    print(f"  serie noire du modele final : {cfg.serie_noire_n} pertes d'affilee, "
          f"perte reelle x{cfg.serie_noire_k:.2f} -> risque max {100 * plafond_serie_noire(cfg):.1f} %",
          flush=True)
    if bool(getattr(cfg, "budget_baisse", False)):
        # LE BUDGET DE BAISSE du modele final : la mediane des dix folds.
        fs = [float(torch.load(x["path"], map_location="cpu", weights_only=False)
                    .get("config", {}).get("mise_budget", cfg.mise_budget))
              for x in candidats if os.path.isfile(x["path"])]
        if fs:
            cfg = replace(cfg, mise_budget=float(np.median(fs)))
            print(f"  budget de baisse du modele final : mise de base {100 * cfg.mise_budget:.2f} % "
                  f"(mediane des folds, de {100 * min(fs):.2f} a {100 * max(fs):.2f} %)", flush=True)
    return cfg


def reconstruit_deploy_ensemble(cfg: JeuConfig) -> int:
    """Relance seulement l'assemblage deploy apres un run de folds termine."""
    rng = np.random.default_rng(cfg.graine)
    torch.manual_seed(cfg.graine)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = pd.read_pickle(cfg.cache)
    cols = list(dict.fromkeys(["time", "open", "high", "low", "close", "atr_14", "spread_bar"] +
                              colonnes_jeu(cfg)))
    d = d[cols].reset_index(drop=True)
    N = len(d)
    o, h, l, c = (d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    sp = d["spread_bar"].to_numpy(np.float64)
    atr = atr_effectif(d["atr_14"].to_numpy(np.float64), c, cfg)
    X = d[colonnes_jeu(cfg)].to_numpy(np.float32)
    marge = fraction_marge(c, atr, cfg)
    R0, D0, S0 = table_coups(o, h, l, sp, atr, cfg, 0.0)
    R1, D1, S1 = table_coups(o, h, l, sp, atr, cfg, 1.0)
    toutes = parties(d["time"], 0, N, cfg)
    candidats = []
    for bloc in range(1, int(cfg.n_blocs) + 1):
        path = f"best_{cfg.prefixe}_bloc{bloc}.pth"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Meilleur checkpoint absent : {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        candidats.append({"bloc": bloc, "epoch": int(ck.get("epoch", 0)), "path": path})
    print(f"[DEPLOY ENSEMBLE] reconstruction a partir de {len(candidats)} meilleurs folds, "
          "sans relancer les validations croisees.", flush=True)
    cfg = config_serie_noire_folds(cfg, candidats)
    entraine_deploiement_continu(cfg, d, X, None, toutes, o, h, l, c, sp, atr,
                                 R0, D0, S0, R1, D1, S1, marge, candidats, [], device, rng)
    return 0


def main_blocs(cfg: JeuConfig) -> int:
    """La validation croisee purgee. Voir `validation`."""
    rng = np.random.default_rng(cfg.graine)
    torch.manual_seed(cfg.graine)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seul = int(getattr(cfg, "bloc_seul", 0))
    manifeste = f"run_{cfg.prefixe}" + (f"_bloc{seul}" if seul else "") + ".json"
    if os.path.exists(manifeste):
        raise FileExistsError(f"{manifeste} existe : nouveau prefixe, ou lancer.ps1")
    with open(manifeste, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=1, default=str)
    print("=" * 70)
    print(f"  KAIROS EN JEU — {getattr(cfg, 'marche_seul', '') or 'BTCUSD'} "
          f"M{cfg.minutes_par_barre} : une "
          f"{'semaine' if cfg.partie == 'semaine' else 'journee'} = une partie  |  "
          f"VALIDATION CROISEE PURGEE en {cfg.n_blocs} blocs")
    print("=" * 70)
    print(f"  regles : {cfg.jetons} coups par partie, {cfg.positions_max} position(s) a la fois, "
          f"fin de partie a -{cfg.vie_R:g} R, temps limite {cfg.horizon_max} bougies  |  un R = "
          f"{cfg.risque_dollars:.0f}$  |  {device}", flush=True)
    d = pd.read_pickle(cfg.cache)
    if getattr(cfg, "marche_seul", ""):
        d, cfg = donnees_marche_seul(d, cfg)
    cols = list(dict.fromkeys(["time", "open", "high", "low", "close", "atr_14",
                               "spread_bar"] + colonnes_jeu(cfg)))
    d = d[cols].reset_index(drop=True)
    N = len(d)
    o, h, l, c = (d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    sp = d["spread_bar"].to_numpy(np.float64)
    atr = atr_effectif(d["atr_14"].to_numpy(np.float64), c, cfg)
    if getattr(cfg, "marche_seul", ""):
        # LE PLANCHER DU MARCHE, comme au multi-marches : le cout reste sous
        # ~0.2 R. Le BTC garde ses 20 bps (5 x 3.5 = 17.5 < 20).
        cout_med = float(np.median(sp)) + cfg.glissement_entree_bps + cfg.glissement_sortie_bps
        atr = np.maximum(atr, cfg.plancher_couts * cout_med * 1e-4 * c)
        print(f"  {cfg.marche_seul} : cout med {cout_med:.2f} bps, ATR des barrieres >= "
              f"{max(float(cfg.atr_min_bps), cfg.plancher_couts * cout_med):.1f} bps, swap "
              f"{cfg.swap_achat_bps_jour:.2f} / {cfg.swap_vente_bps_jour:.2f} bps/jour, contrat "
              f"{cfg.contrat:g}, lot min {cfg.lot_min:g}", flush=True)
    t_ns = d["time"].values.astype("int64")
    X = d[colonnes_jeu(cfg)].to_numpy(np.float32)
    marge = fraction_marge(c, atr, cfg)
    R0, D0, S0 = table_coups(o, h, l, sp, atr, cfg, 0.0)
    R1, D1, S1 = table_coups(o, h, l, sp, atr, cfg, 1.0)
    M1 = cible_meteo(h, l, c, cfg)
    y_ex = cibles_expert(R1, cfg)
    toutes = parties(d["time"], 0, N, cfg)
    purge = purge_barres(cfg)
    _fmt = lambda i: pd.Timestamp(d["time"].iloc[min(i, N - 1)]).strftime("%Y-%m-%d")
    print(f"[CACHE] {N:,} bougies, {_fmt(0)} -> {_fmt(N - 1)} ; {len(toutes)} parties ; "
          f"zone tampon {purge} bougies autour du test et de la validation", flush=True)
    tests = []
    cfg_depart = cfg
    candidats_deploiement = []
    transitions_oof = []
    for k in range(cfg.n_blocs):
        if seul and k + 1 != seul:
            continue
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
        pred, modeles_expert = expert_multi(Xn, y_ex, t_ns, permis, fin_tr, cfg)
        FE = features_expert(pred, fenetre=10_000 // int(cfg.minutes_par_barre))
        rangs_ex = FE[:, 2:4].copy()
        m_e = FE[permis].astype(np.float64).mean(0)
        s_e = FE[permis].astype(np.float64).std(0)
        FEn = np.clip((FE - m_e) / (s_e + 1e-8), -5.0, 5.0).astype(np.float32)
        Xk = np.concatenate([Xn, FEn[:, :cfg.n_expert]], axis=1)
        from jeu_artifacts import sauvegarde_pipeline_bloc, empreinte_fichier
        pipeline_path = f"pipeline_{cfg.prefixe}_bloc{k + 1}.json"
        sauvegarde_pipeline_bloc(
            pipeline_path, cfg=cfg, donnees=d, colonnes=colonnes_jeu(cfg),
            normalisation_marche={"mean": st_m.astype(np.float32), "std": st_s.astype(np.float32)},
            normalisation_expert={"mean": m_e, "std": s_e},
            modeles_expert=modeles_expert, predictions_expert=pred,
            meta={"bloc": k + 1, "test": [te0, te1], "validation": [va0, va1],
                  "purge_barres": purge, "validation_mode": "croisee_purgee"})
        pipeline_ref = {"fichier": pipeline_path, "sha256": empreinte_fichier(pipeline_path)}
        print(f"  pipeline fige : {pipeline_path} (expert, normalisations, predictions)", flush=True)
        ics = []
        for s_ in range(2):
            ok = np.zeros(N, bool)
            ok[te0:te1] = True
            ok &= np.isfinite(y_ex[:, s_])
            ics.append(pd.Series(pred[ok, s_]).corr(pd.Series(y_ex[ok, s_]), method="spearman"))
        print(f"  expert  LightGBM appris sur les autres blocs, {time.time() - t_ex:.0f} s ; "
              f"correlation sur le bloc de test : achat {ics[0]:+.3f}, vente {ics[1]:+.3f}",
              flush=True)
        # LA SERIE NOIRE repart de sa valeur prudente a chaque bloc.
        cfg = replace(cfg, serie_noire_n=cfg_depart.serie_noire_n,
                      serie_noire_k=cfg_depart.serie_noire_k,
                      mise_budget=cfg_depart.mise_budget)
        policy = PolitiqueJeu(cfg).to(device)
        optims = optimiseurs(policy, cfg)
        pos = coups_expert_predits(j_tr, pred, D1, fin_tr, cfg)
        imite_expert(policy, optims, pos, j_tr, Xk, fin_tr, cfg, device, rng)
        suffixe = f"_bloc{k + 1}"
        # Le checkpoint principal est choisi sur le resultat reel du compte
        # (lots et marge inclus), sans changer la recompense PPO en R.
        best_path = f"best_{cfg.prefixe}{suffixe}.pth"
        best_r_path = f"bestR_{cfg.prefixe}{suffixe}.pth"
        record_profit, garde_profit = 0.0, None
        record_r, garde_r = 0.0, None
        for epoch in range(0, cfg.epochs + 1):
            t_ep = time.time()
            frac = min(1.0, max(0.0, (epoch - 1) / cfg.rampe_cout)) if cfg.rampe_cout > 0 else 1.0
            st = None
            dm, cvc = {}, {}
            if epoch >= 1:
                Rf, Df, Sf = table_a_cout(frac, (R0, D0, S0), (R1, D1, S1),
                                         lambda f: table_coups(o, h, l, sp, atr, cfg, f))
                choix = rng.choice(len(j_tr), size=min(cfg.parties_par_epoch, len(j_tr)),
                                   replace=False)
                marge_d = max(240 // int(cfg.minutes_par_barre), cfg.barres_par_partie // 6)
                sc, cp, tr = joue(policy, departs_tires(j_tr[choix], rng, marge_d), Xk,
                                  Rf, Df, Sf, fin_tr, cfg, device, explore=True,
                                  collecte=True, rangs=rangs_ex, marge=marge)
                mois_on = bool(getattr(cfg, "note_mise_mois", False))
                st = maj_ppo(policy, optims, avantages(tr, cfg), Xk, cfg, device, rng,
                             mode="signal" if mois_on else "tout")
                # LA TETE D'ESPERANCE apprend le vrai resultat des coups joues.
                es = entraine_esperance(policy, optims, tr, Xk, R1, cfg, device, rng)
                # LES TETES DU DEJA-VU ET METEO, sur les etats de ces journees.
                dm = entraine_dejavu_meteo(policy, optims, tr, Xk, M1, cfg, device, rng)
                # LA TETE DE CONVICTION, sur les coups de ces journees.
                cvc = entraine_conviction(policy, optims, tr, R1, rangs_ex, cfg, device, rng)
                if mois_on:
                    # LES MOIS : seules les tetes de mise apprennent. Voir
                    # `note_mise_mois`.
                    jm, gm = tire_mois(j_tr, cfg, rng)
                    if len(jm):
                        suivi = {}
                        _sm, cpm, trm = joue(policy, jm, Xk, Rf, Df, Sf, fin_tr, cfg, device,
                                             explore=not bool(getattr(cfg, "mois_signal_reel", False)),
                                             explore_mise=True,
                                             collecte=True, rangs=rangs_ex,
                                             marge=marge, chainer=True, groupes=gm, suivi=suivi)
                        am = avantage_mois(trm, suivi, cfg)
                        maj_ppo(policy, optims, avantages(trm, cfg), Xk, cfg, device, rng,
                                mode="mise", adv_mise_externe=am)
                        es = entraine_esperance(policy, optims, trm, Xk, R1, cfg, device, rng) or es
                        cvc = entraine_conviction(policy, optims, trm, R1, rangs_ex, cfg, device, rng) or cvc
                        fins = suivi.get("final", np.zeros(0))
                        print(f"  mois  {len(fins)} mois de {cfg.mois_jours} jours, {len(cpm)} coups : "
                              f"compte final median {np.median(fins):.0f} $, "
                              f"{int((fins < 0.5 * cfg.capital).sum())} sous 500 $, "
                              f"{int((fins <= 1e-6).sum())} vides", flush=True)
                del Rf, Df, Sf
            gen = torch.Generator(device=device)
            gen.manual_seed(cfg.graine)
            sv, cv, _ = joue(policy, j_va, Xk, R1, D1, S1, va1, cfg, device,
                             explore=False, gen=gen, rangs=rangs_ex, marge=marge)
            bv = bilan(sv, cv, c, atr, sp, cfg, open_=o, temps=d["time"])
            ecrit_trades_csv(f"trades_{cfg.prefixe}{suffixe}.csv", cv, d["time"], cfg,
                              epoch=epoch, phase="validation")
            nom = "EXPERT IMITE" if epoch == 0 else f"cout {100 * frac:.0f}%"
            print(f"\nEPOCH {epoch:03d}  {nom:>12}  VAL  {ligne_bilan(bv, cfg)}  "
                  f"{(time.time() - t_ep) / 60:.1f} min", flush=True)
            print(f"  bilan  {ligne_detail(bv, cfg)}", flush=True)
            if bool(getattr(cfg, "sortie_deux_temps", False)) and bv["coups"]:
                so = bv["sorties"]
                print(f"  sorties  objectif final {100 * so[0]:.0f} %  |  objectif proche puis "
                      f"entree {100 * so[3]:.0f} %  |  objectif proche puis temps {100 * so[4]:.0f} %  |  "
                      f"stop {100 * so[1]:.0f} %  |  temps {100 * so[2]:.0f} %", flush=True)
            etat = {"modele": policy.state_dict(), "config": asdict(cfg), "epoch": epoch,
                    "bloc": k + 1, "pipeline": pipeline_ref}
            torch.save(etat, f"last_{cfg.prefixe}{suffixe}.pth")
            if getattr(policy, "tete_conviction", False):
                mc_ = mesure_conviction(cv, cfg)
                if mc_:
                    _t = lambda q: f"x{q[0]:.2f} PF {q[1]:.2f} ({q[2]:.0f} %)"
                    print(f"  conviction  validation, rang de l'expert < 0.98 : {_t(mc_['bas'])}  |  "
                          f"0.98-0.99 : {_t(mc_['milieu'])}  |  >= 0.99 : {_t(mc_['haut'])}  |  "
                          f"multiplicateur moyen x{mc_['mult_moyen']:.2f}", flush=True)
                if cvc:
                    print(f"  conviction  apprise sur {cvc['n']} coups d'entrainement : choisit "
                          f"x{cvc['mult_bas']:.2f} / x{cvc['mult_milieu']:.2f} / x{cvc['mult_haut']:.2f} "
                          f"(R moyen {cvc['R_bas']:+.3f} / {cvc['R_milieu']:+.3f} / {cvc['R_haut']:+.3f})  |  "
                          f"prix du risque {cvc['lambda']:+.3f} R  |  moyenne x{cvc['mult_moyen']:.2f}, "
                          f"plus forte x{cvc['mult_max']:.2f}", flush=True)
            if getattr(policy, "apprendre_dejavu_meteo", False):
                md = mesure_dejavu_meteo(cv, cfg, M1)
                if md:
                    _f = lambda v: "  ".join("inf" if not np.isfinite(x) else f"{x:.2f}" for x in v)
                    appris = (f"erreur d'apprentissage {dm['perte_vu']:.3f}"
                              if epoch >= 1 and dm else "pas encore appris")
                    print(f"  deja-vu  PF par tranche, du plus familier au plus nouveau : "
                          f"{_f(md['dejavu']['pf'])}  |  mise {_f(md['dejavu']['mise'])} %  |  "
                          f"{appris}", flush=True)
                    appris = (f"R2 a l'entrainement {dm['r2_meteo']:+.2f}"
                              if epoch >= 1 and dm else "pas encore appris")
                    print(f"  meteo  PF par tranche, du plus calme au plus agite : "
                          f"{_f(md['meteo']['pf'])}  |  mise {_f(md['meteo']['mise'])} %  |  "
                          f"prevision / mouvement reel en validation {md['rho_meteo']:+.2f}  |  "
                          f"{appris}", flush=True)
            if bool(getattr(cfg, "apprendre_esperance", False)):
                rho, base = mesure_esperance(cv)
                print(f"  esperance  correlation prediction / resultat en validation {rho:+.3f}  |  "
                      f"mise de base moyenne {base:.2f} % du compte", flush=True)
            if bool(getattr(cfg, "serie_noire", False)) and len(cv):
                # LA SERIE NOIRE, remesuree sur cette validation pour la
                # suite ; le checkpoint garde celle avec laquelle il a joue.
                # Une validation sans coup ne mesure rien : la regle reste.
                n_obs, k_obs = mesure_serie_noire(cv, cfg)
                avant = plafond_serie_noire(cfg)
                cfg = replace(cfg, serie_noire_n=max(int(cfg.serie_noire_n_min), n_obs),
                              serie_noire_k=k_obs)
                print(f"  serie noire  pire serie {n_obs} pertes d'affilee, perte reelle "
                      f"jusqu'a {k_obs:.2f} fois le stop -> risque max par coup "
                      f"{100 * avant:.1f} % -> {100 * plafond_serie_noire(cfg):.1f} %", flush=True)
            if bool(getattr(cfg, "budget_baisse", False)) and len(cv):
                # LE BUDGET DE BAISSE, remesure sur cette validation pour la
                # suite ; le checkpoint garde celui avec lequel il a joue.
                f_b, mb = mesure_budget_baisse(cv, cfg)
                if mb:
                    print(f"  budget  validation rejouee {cfg.budget_baisse_tirages} fois, journees melangees : "
                          f"mise de base {100 * cfg.mise_budget:.2f} % -> {100 * f_b:.2f} % "
                          f"(pire baisse 1 fois sur 10 : {100 * mb['baisse_a_f']:+.0f} % a cette mise, "
                          f"{100 * mb['baisse_1pct']:+.0f} % a 1 %)  |  R moyen {mb['r_moyen']:+.4f}  |  "
                          f"multiplicateur des tetes {mb['mult_moyen']:.2f}  |  plafond {100 * mb['plafond']:.1f} %",
                          flush=True)
                    cfg = replace(cfg, mise_budget=f_b)
            # une validation sans coup n'a ni gain ni baisse (`bilan`)
            profit_jour = bv.get("total_dollars", 0.0) / max(bv["parties"], 1)
            assez = bv["coups"] >= cfg.min_coups_val
            # SANS RUINE : voir `dd_max_sauvegarde`.
            sans_ruine = bv.get("dd_pct", 0.0) >= -float(getattr(cfg, "dd_max_sauvegarde", 1.0))
            admissible = assez and sans_ruine
            sauve = admissible and profit_jour > record_profit
            sauve_r = assez and bv["score"] > record_r
            if sauve:
                record_profit, garde_profit = profit_jour, epoch
                torch.save(etat, best_path)
                print(f"  sauvegarde  MEILLEUR PROFIT : {profit_jour:+.2f}$/jour en "
                      f"validation, {bv['coups']} coups -> {best_path}", flush=True)
            else:
                raison = (f"{bv['coups']} coups, il en faut {cfg.min_coups_val}"
                          if not assez else
                          f"pire baisse {100 * bv.get('dd_pct', 0.0):+.1f} %, au-dela de la limite de "
                          f"-{100 * float(cfg.dd_max_sauvegarde):.0f} %"
                          if not sans_ruine else
                          f"{profit_jour:+.2f}$/jour ne bat pas {record_profit:+.2f}$")
                print(f"  garde  meilleur profit inchange : {raison}"
                      + (f" (meilleur : epoch {garde_profit})"
                         if garde_profit is not None else ""), flush=True)
            # Le meilleur score R est garde pour comparaison, mais ne pilote
            # plus le test final ni le checkpoint principal.
            if sauve_r:
                record_r, garde_r = bv["score"], epoch
                torch.save(etat, best_r_path)
                print(f"  repere  meilleur R : {record_r:+.3f} R/partie "
                      f"(epoch {garde_r}) -> {best_r_path}", flush=True)
            if getattr(cfg, "echanges", ""):
                ecrit_echange(cfg, f"val_bloc{k + 1:02d}_ep{epoch:03d}", cv, t_ns, c, atr,
                              len(j_va), sauve, (va0, va1), d["time"])
        src = best_path if os.path.exists(best_path) else f"last_{cfg.prefixe}{suffixe}.pth"
        etat_src = torch.load(src, map_location=device, weights_only=False)
        policy.load_state_dict(etat_src["modele"])
        # Le test joue avec la serie noire du modele garde (mesuree avant lui).
        cfg = replace(cfg, serie_noire_n=int(etat_src["config"].get("serie_noire_n", cfg.serie_noire_n)),
                      serie_noire_k=float(etat_src["config"].get("serie_noire_k", cfg.serie_noire_k)),
                      mise_budget=float(etat_src["config"].get("mise_budget", cfg.mise_budget)))
        gen = torch.Generator(device=device)
        gen.manual_seed(cfg.graine)
        s_t, c_t, tr_t = joue(policy, j_te, Xk, R1, D1, S1, te1, cfg, device,
                              explore=False, gen=gen, collecte=True,
                              rangs=rangs_ex, marge=marge)
        bt = bilan(s_t, c_t, c, atr, sp, cfg, open_=o, temps=d["time"])
        ecrit_trades_csv(f"trades_{cfg.prefixe}{suffixe}.csv", c_t, d["time"], cfg,
                          epoch=-1, phase="test")
        tests.append((k + 1, _fmt(te0), _fmt(te1 - 1), bt, c_t))
        # Ce sont les seules trajectoires qui alimentent l'eleve final :
        # chacune a ete generee par un modele qui ignorait ce bloc de test.
        transitions_oof.extend(tr_t or [])
        # Seule la validation, jamais le test, peut choisir une graine pour
        # l'apprentissage continu final.
        candidats_deploiement.append({
            "bloc": k + 1,
            "epoch": garde_profit if garde_profit is not None else cfg.epochs,
            "profit_jour": record_profit if garde_profit is not None else float("-inf"),
            "path": src,
        })
        print(f"\nTEST fold {k + 1} ({os.path.basename(src)})  {ligne_bilan(bt, cfg)}", flush=True)
        print(f"TEST fold {k + 1} bilan  {ligne_detail(bt, cfg)}", flush=True)
        with open(f"test_{cfg.prefixe}{suffixe}.json", "w", encoding="utf-8") as fh:
            json.dump({k_: v_ for k_, v_ in bt.items()}, fh, indent=1, default=str)
        if getattr(cfg, "echanges", ""):
            ecrit_echange(cfg, f"test_bloc{k + 1:02d}", c_t, t_ns, c, atr, len(j_te), True,
                          (te0, te1), d["time"])
    if seul:
        print(f"\nFIN du bloc {seul}", flush=True)
        return 0
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
    if bool(getattr(cfg, "deploiement_continu", True)):
        cfg = config_serie_noire_folds(cfg, candidats_deploiement)
        entraine_deploiement_continu(
            cfg, d, X, y_ex, toutes, o, h, l, c, sp, atr,
            R0, D0, S0, R1, D1, S1, marge, candidats_deploiement,
            transitions_oof, device, rng)
    print("\nFIN de la validation croisee purgee", flush=True)
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


MT5_CRYPTO = "cache_crypto_m5_mt5.pkl"


def spec_marche(m: str, cfg: JeuConfig) -> Dict[str, float]:
    """Contrat, lot minimum et pas de lot du symbole Vantage, lus dans MT5
    (`telecharge_crypto_mt5`) ; ceux de `cfg` si le symbole n'y est pas."""
    try:
        inf = pd.read_pickle(MT5_CRYPTO)["infos"][m]
        return {"contrat": float(inf["contrat"]), "lot_min": float(inf["lot_min"]),
                "pas_lot": float(inf["pas_lot"]), "lot_max": float(inf.get("lot_max", cfg.lot_max))}
    except (FileNotFoundError, KeyError):
        return {"contrat": float(cfg.contrat), "lot_min": float(cfg.lot_min),
                "pas_lot": float(cfg.pas_lot), "lot_max": float(cfg.lot_max)}


def config_marche(cfg: JeuConfig, m: str, bloc: int = 0) -> JeuConfig:
    """La configuration du modele d'UN marche (et d'un seul bloc si `bloc`).
    Voir `modeles_par_marche`."""
    sp = spec_marche(m, cfg)
    return replace(cfg, marches=(), marche_seul=m, modeles_par_marche=False,
                   marches_communs=tuple(cfg.marches), bloc_seul=int(bloc),
                   prefixe=f"{cfg.prefixe}_{m}", echanges=f"echanges_{cfg.prefixe}",
                   contrat=sp["contrat"], lot_min=sp["lot_min"], pas_lot=sp["pas_lot"],
                   lot_max=sp["lot_max"])


def donnees_marche_seul(d_all: pd.DataFrame, cfg: JeuConfig):
    """(bougies du marche, configuration a son swap) dans le cache multi :
    a partir du debut commun aux marches du compte (`marches_communs`, tous
    ceux du cache s'il est vide) si `debut_commun`."""
    mc = tuple(getattr(cfg, "marches_communs", ())) or tuple(d_all["marche"].unique())
    dc = d_all[d_all["marche"].isin(mc)]
    t0 = (dc.groupby("marche")["time"].min().max() if getattr(cfg, "debut_commun", False)
          else d_all["time"].min())
    d = d_all[(d_all["marche"] == cfg.marche_seul) & (d_all["time"] >= t0)].reset_index(drop=True)
    if "swap_achat_bps_jour" in d.columns:
        cfg = replace(cfg, swap_achat_bps_jour=float(d["swap_achat_bps_jour"].iloc[0]),
                      swap_vente_bps_jour=float(d["swap_vente_bps_jour"].iloc[0]))
    return d, cfg


def table_trades(coups, t_ns, close, atr, cfg: JeuConfig) -> pd.DataFrame:
    """Les coups d'un marche, ce qu'il faut pour les rejouer sur le compte
    commun : entree, sortie, R, sens, distance du stop en prix, prix."""
    cols = ["t_entree", "t_sortie", "r", "sens", "dist", "prix"]
    if not len(coups):
        return pd.DataFrame(columns=cols)
    a = np.array([q[:7] for q in coups], np.float64)
    t = a[:, 1].astype(np.int64)
    ts = np.minimum(t + a[:, 6].astype(np.int64), len(t_ns) - 1)
    j = a[:, 4].astype(np.int64)
    return pd.DataFrame({"t_entree": np.asarray(t_ns)[t], "t_sortie": np.asarray(t_ns)[ts],
                         "r": a[:, 5], "sens": a[:, 2].astype(np.int64),
                         "dist": np.asarray(cfg.sl_atr, np.float64)[j] * np.asarray(atr)[t],
                         "prix": np.asarray(close)[t]})


def ecrit_echange(cfg: JeuConfig, nom: str, coups, t_ns, close, atr, n_jours: int,
                  sauve: bool, bornes, temps: pd.Series) -> None:
    """Depose les coups d'une validation ou d'un test pour le compte commun.
    Ecrit a cote puis renomme : le lecteur ne voit jamais un fichier a moitie."""
    os.makedirs(cfg.echanges, exist_ok=True)
    f = os.path.join(cfg.echanges, f"{nom}__{cfg.marche_seul}.pkl")
    n = len(temps)
    pd.to_pickle({"trades": table_trades(coups, t_ns, close, atr, cfg), "n_jours": int(n_jours),
                  "sauve": bool(sauve), "marche": cfg.marche_seul,
                  "debut": temps.iloc[min(int(bornes[0]), n - 1)],
                  "fin": temps.iloc[min(int(bornes[1]) - 1, n - 1)]}, f + ".tmp")
    os.replace(f + ".tmp", f)


def bilan_commun(parts: Dict[str, pd.DataFrame], specs: Dict[str, Dict[str, float]],
                 n_jours: int, cfg: JeuConfig) -> Dict:
    """Les coups de TOUS les marches sur UN compte commun.

    A mise fixe (un R = `risque_dollars`) : trades, win rate, profit factor,
    longs et shorts, total, drawdown du compte, resultats portes a la sortie
    de chaque coup, dans l'ordre du temps.

    COMME EN LIVE : un seul solde de `capital`. Chaque coup, a son entree,
    mise `risque_pct` du solde REALISE du moment, en lots de SON symbole ; il
    n'est pris que si la marge libre du compte (solde moins la marge des
    coups encore ouverts, TOUS marches confondus) couvre la sienne.
    """
    import heapq
    vides = [p.assign(marche=m) for m, p in parts.items() if len(p)]
    if not vides:
        return {"coups": 0, "n_jours": n_jours, "par_marche": {}}
    tout = pd.concat(vides, ignore_index=True)
    tout = tout.sort_values(["t_entree"], kind="stable").reset_index(drop=True)
    r = tout["r"].to_numpy(np.float64)
    n = len(r)
    g, pe = float(r[r > 0].sum()), float(-r[r < 0].sum())
    b = {"coups": n, "n_jours": n_jours, "gagnants": int((r > 0).sum()),
         "perdants": int((r < 0).sum()), "win_rate": float((r > 0).mean()),
         "pf": g / pe if pe > 0 else float("inf"), "longs": int((tout["sens"] == 0).sum()),
         "shorts": int((tout["sens"] == 1).sum()), "total_dollars": float(r.sum() * cfg.risque_dollars),
         "gain_R": float(r.mean())}
    sorties_t = tout["t_sortie"].to_numpy()
    o = np.argsort(sorties_t, kind="stable")
    courbe = [float(cfg.capital)]
    debut = 0
    while debut < len(o):
        fin_groupe = debut + 1
        while fin_groupe < len(o) and sorties_t[o[fin_groupe]] == sorties_t[o[debut]]:
            fin_groupe += 1
        courbe.append(max(0.0, courbe[-1] + float((r[o[debut:fin_groupe]] * cfg.risque_dollars).sum())))
        debut = fin_groupe
    courbe = np.asarray(courbe, dtype=np.float64)
    pic = np.maximum.accumulate(courbe)
    b["dd_dollars"] = float((courbe - pic).min())
    b["dd_pct"] = float(np.clip(((courbe - pic) / np.maximum(pic, 1e-12)).min(), -1.0, 0.0))
    b["total_dollars"] = float(courbe[-1] - cfg.capital)
    # COMME EN LIVE, sur le solde commun
    # LA MISE SUIT LE SOLDE : plus le solde commun grandit, plus les lots de
    # chaque marche grossissent (et l'inverse en baisse), comme en live.
    E = float(cfg.capital)
    ouverts, courbe_l, pnl_m, pris, sautes, risques = [], [E], {}, 0, 0, []
    lot_max, lot_moy, gagnes_l = {}, {}, 0
    for k in range(n):
        te = int(tout["t_entree"].iat[k])
        while ouverts and ouverts[0][0] <= te:
            _, _, pnl, _m = heapq.heappop(ouverts)
            E = max(0.0, E + pnl)
            courbe_l.append(E)
        if E <= 0:
            break
        m = tout["marche"].iat[k]
        sp = specs[m]
        dist = float(tout["dist"].iat[k])
        cible = E * cfg.risque_pct / 100.0
        lots = np.floor(cible / (dist * sp["contrat"]) / sp["pas_lot"] + 1e-9) * sp["pas_lot"]
        lots = max(float(lots), float(sp["lot_min"]))
        marge = lots * sp["contrat"] * float(tout["prix"].iat[k]) / float(cfg.levier)
        if marge > E - sum(q[1] for q in ouverts):
            sautes += 1
            continue
        risque = lots * sp["contrat"] * dist
        risques.append(risque / E)
        pnl = float(r[k]) * risque
        pnl_m[m] = pnl_m.get(m, 0.0) + pnl
        lot_max[m] = max(lot_max.get(m, 0.0), lots)
        lot_moy.setdefault(m, []).append(lots)
        gagnes_l += int(r[k] > 0)
        heapq.heappush(ouverts, (int(tout["t_sortie"].iat[k]), marge, pnl, m))
        pris += 1
    while ouverts:
        _, _, pnl, _m = heapq.heappop(ouverts)
        E = max(0.0, E + pnl)
        courbe_l.append(E)
    cl = np.asarray(courbe_l)
    pl = np.maximum.accumulate(cl)
    b.update({"live_total": float(E - cfg.capital), "live_dd_dollars": float((cl - pl).min()),
              "live_dd_pct": float(np.clip(((cl - pl) / np.maximum(pl, 1e-12)).min(), -1.0, 0.0)), "live_pris": pris,
              "live_marge": sautes, "live_solde": float(E),
              "live_win_rate": gagnes_l / pris if pris else float("nan"),
              "live_risque_moy": float(np.mean(risques)) if risques else float("nan")})
    par = {}
    for m in parts:
        rm = tout.loc[tout["marche"] == m, "r"].to_numpy(np.float64)
        gm, pm = float(rm[rm > 0].sum()), float(-rm[rm < 0].sum())
        par[m] = {"coups": len(rm), "total_dollars": float(rm.sum() * cfg.risque_dollars),
                  "win_rate": float((rm > 0).mean()) if len(rm) else float("nan"),
                  "pf": gm / pm if pm > 0 else float("inf"), "live": pnl_m.get(m, 0.0),
                  "lot_max": lot_max.get(m, 0.0),
                  "lot_moy": float(np.mean(lot_moy[m])) if m in lot_moy else 0.0}
    b["par_marche"] = par
    return b


def ligne_commune(b: Dict, cfg: JeuConfig) -> str:
    if not b.get("coups"):
        return "aucun trade"
    j = max(int(b["n_jours"]), 1)
    return (f"{b['coups']} trades ({b['coups'] / j:.1f}/jour) : {b['gagnants']} gagnants, "
            f"{b['perdants']} perdants (win rate {100 * b['win_rate']:.1f} %)  |  {b['longs']} longs, "
            f"{b['shorts']} shorts  |  profit factor {b['pf']:.2f}  |  total {b['total_dollars']:+.2f} $ "
            f"sur {j} jours ({b['total_dollars'] / j:+.2f} $/jour)  |  drawdown max "
            f"{b['dd_dollars']:+.2f} $ ({100 * b['dd_pct']:+.1f} %)  |  COMME EN LIVE, un seul solde "
            f"de {cfg.capital:.0f} $, {cfg.risque_pct:g} % du solde du moment par coup : solde final "
            f"{b['live_solde']:.2f} $ ({b['live_total']:+.2f} $), pire baisse "
            f"{b['live_dd_dollars']:+.2f} $ ({100 * b['live_dd_pct']:+.1f} %), win rate "
            f"{100 * b['live_win_rate']:.1f} % sur {b['live_pris']} coups pris, risque reel moyen "
            f"{100 * b['live_risque_moy']:.2f} %, {b['live_marge']} coups sans marge")


def ligne_commune_marches(b: Dict) -> str:
    return "  |  ".join(f"{m} {v['coups']} trades {v['total_dollars']:+.1f} $ WR "
                        f"{100 * v['win_rate']:.0f} % PF {v['pf']:.2f} (live {v['live']:+.1f} $, "
                        f"lots {v['lot_moy']:.2f} en moyenne, {v['lot_max']:.2f} au plus)"
                        for m, v in b.get("par_marche", {}).items())


def main_modeles_par_marche(cfg: JeuConfig) -> int:
    """Un modele par marche, en parallele, sur un compte commun. Voir
    `modeles_par_marche`.

    Chaque modele tourne dans son propre processus (`python jeu_kairos.py
    <marche> <bloc>`, journal `training_<marche>.log`) ; ce processus-ci
    relaie ses lignes, suffixees du marche, et des que tous les marches ont
    joue la meme validation (ou le meme test), rejoue leurs coups sur le
    compte commun.

    UN MODELE APRES L'AUTRE, BLOC PAR BLOC — 2026-09-28 (run multi_m5_05),
    demande du proprietaire. En parallele (run multi_m5_04), les deux
    modeles se disputaient une carte graphique bridee par la chaleur (90 °C,
    210 MHz sur 2100) : la deuxieme passe d'imitation n'avait pas fini en
    dix minutes, contre moins d'une minute pour le BTC seul. Mesure du
    2026-09-28, lot d'imitation de 512 : 67 ms sur la carte, 1256 ms sur le
    processeur — la carte reste le bon endroit, un modele a la fois."""
    import re
    import shutil
    import subprocess
    manifeste = f"run_{cfg.prefixe}.json"
    if os.path.exists(manifeste):
        raise FileExistsError(f"{manifeste} existe : nouveau prefixe, ou lancer.ps1")
    with open(manifeste, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=1, default=str)
    marches = tuple(cfg.marches)
    specs = {m: spec_marche(m, cfg) for m in marches}
    print("=" * 70)
    print(f"  KAIROS EN JEU — {len(marches)} MODELES INDEPENDANTS M{cfg.minutes_par_barre}, "
          f"UN COMPTE COMMUN  |  VALIDATION CROISEE PURGEE en {cfg.n_blocs} blocs")
    print("=" * 70)
    print(f"  marches : {', '.join(marches)}  |  chacun son tronc, ses tetes, ses optimiseurs, "
          f"son expert, sa normalisation, SON meilleur modele", flush=True)
    print(f"  regles de chaque modele : {cfg.jetons} coups par jour, {cfg.positions_max} position(s) "
          f"a la fois, fin de partie a -{cfg.vie_R:g} R, porte {cfg.porte_rang_expert:g}, expert "
          f"{cfg.expert_k} coups  |  compte commun : {cfg.capital:.0f} $, {cfg.risque_pct:g} % du "
          f"solde commun par coup, marge partagee (levier 1:{cfg.levier:.0f})", flush=True)
    for m in marches:
        print(f"  {m} : contrat {specs[m]['contrat']:g}, lot min {specs[m]['lot_min']:g}, "
              f"pas {specs[m]['pas_lot']:g}", flush=True)
    dossier = f"echanges_{cfg.prefixe}"
    shutil.rmtree(dossier, ignore_errors=True)
    os.makedirs(dossier, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
               OMP_NUM_THREADS="5", MKL_NUM_THREADS="5")
    pos, reste = {}, {}
    for m in marches:
        open(f"training_{m}.log", "wb").close()
        pos[m], reste[m] = 0, b""
    bruit = re.compile(r"Warning|^\s+(o = |d\[|creux12)")

    def relaie(m):
        with open(f"training_{m}.log", "rb") as f:
            f.seek(pos[m])
            data = f.read()
        pos[m] += len(data)
        lignes = (reste[m] + data).split(b"\n")
        reste[m] = lignes.pop()
        for x in lignes:
            t = x.decode("utf-8", "replace").rstrip("\r")
            if not bruit.search(t):
                print(f"{t}   [{m}]" if t.strip() else "", flush=True)

    faits, tests = set(), {}

    def combine():
        noms = {}
        for f in os.listdir(dossier):
            if f.endswith(".pkl") and "__" in f:
                a, mm = f[:-4].split("__")
                noms.setdefault(a, set()).add(mm)
        def cle(a):
            ep = re.search(r"_ep(\d+)", a)
            return (int(re.search(r"bloc(\d+)", a).group(1)), a.startswith("test_"),
                    int(ep.group(1)) if ep else 0)
        for a in sorted(noms, key=cle):
            if a in faits or not set(marches) <= noms[a]:
                continue
            ech = {m: pd.read_pickle(os.path.join(dossier, f"{a}__{m}.pkl")) for m in marches}
            nj = max(e["n_jours"] for e in ech.values())
            b = bilan_commun({m: e["trades"] for m, e in ech.items()}, specs, nj, cfg)
            sauves = [m for m in marches if ech[m]["sauve"]]
            deb = min(e["debut"] for e in ech.values())
            fin = max(e["fin"] for e in ech.values())
            if a.startswith("val_"):
                bloc, _t, ep = cle(a)
                print(f"\nEPOCH {ep:03d}  COMPTE COMMUN  VAL bloc {bloc} ({deb:%Y-%m-%d} -> "
                      f"{fin:%Y-%m-%d})  |  modeles sauvegardes a cette epoch : "
                      f"{', '.join(sauves) if sauves else 'aucun'}", flush=True)
                print(f"  bilan  COMPTE COMMUN  {ligne_commune(b, cfg)}", flush=True)
                print(f"  bilan  COMPTE COMMUN par marche : {ligne_commune_marches(b)}", flush=True)
            else:
                bloc = cle(a)[0]
                tests[bloc] = (deb, fin, b, ech)
                print(f"\nTEST fold {bloc} bilan  COMPTE COMMUN ({deb:%Y-%m-%d} -> {fin:%Y-%m-%d}, "
                      f"chaque marche avec SON meilleur modele)  {ligne_commune(b, cfg)}", flush=True)
                print(f"TEST fold {bloc} bilan  COMPTE COMMUN par marche : "
                      f"{ligne_commune_marches(b)}", flush=True)
            faits.add(a)

    arret = False
    for k in range(1, cfg.n_blocs + 1):
        for m in marches:
            print(f"\n>>> bloc {k}/{cfg.n_blocs} : le modele {m} (un modele a la fois)", flush=True)
            with open(f"training_{m}.log", "ab") as fh:
                p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), m, str(k)],
                                     stdout=fh, stderr=subprocess.STDOUT, env=env)
                while p.poll() is None:
                    relaie(m)
                    combine()
                    time.sleep(5)
            relaie(m)
            combine()
            if p.returncode != 0:
                print(f"  ARRET du modele {m} au bloc {k} : code {p.returncode}, voir "
                      f"training_{m}.log", flush=True)
                arret = True
                break
        if arret:
            break
    print("\n" + "=" * 70)
    print(f"  RESUME DES {cfg.n_blocs} BLOCS DE TEST, COMPTE COMMUN (chacun jamais vu par ses modeles)")
    print("=" * 70)
    for k in sorted(tests):
        deb, fin, b, _e = tests[k]
        print(f"  bloc {k:2d}  {deb:%Y-%m-%d} -> {fin:%Y-%m-%d}  {b.get('coups', 0):5d} trades  WR "
              f"{100 * b.get('win_rate', 0):4.1f} %  PF {b.get('pf', float('nan')):5.2f}  total "
              f"{b.get('total_dollars', 0.0):+9.2f} $  DD {100 * b.get('dd_pct', 0.0):+.1f} %  |  "
              f"{ligne_commune_marches(b)}", flush=True)
    if tests:
        positifs = sum(1 for *_x, b, _e in tests.values() if b.get("total_dollars", 0.0) > 0)
        tous = {m: pd.concat([e[m]["trades"] for *_x, e in tests.values()], ignore_index=True)
                for m in marches}
        nj = sum(b.get("n_jours", 0) for *_x, b, _e in tests.values())
        bt = bilan_commun(tous, specs, nj, cfg)
        print(f"  blocs gagnants : {positifs}/{len(tests)}  |  total {bt.get('total_dollars', 0.0):+.2f} $  |  "
              f"{bt.get('coups', 0)} trades  |  profit factor global {bt.get('pf', float('nan')):.2f}",
              flush=True)
        print(f"  tous les blocs a la suite, COMPTE COMMUN : {ligne_commune(bt, cfg)}", flush=True)
        print(f"  tous les blocs, par marche : {ligne_commune_marches(bt)}", flush=True)
    print("\nFIN du walk-forward", flush=True)
    return 0


def main_pas_a_pas(cfg: JeuConfig) -> int:
    """Les modeles de chaque marche, entraines PAS A PAS dans UN processus.

    2026-09-28 (run multi_m5_06), demande du proprietaire : « la passe un du
    premier expert, puis la passe un du deuxieme, puis la passe deux du
    premier... ; les modeles ensuite en meme temps, mais en sequentiel ».
    En parallele (run multi_m5_04), deux processus se disputaient la carte
    graphique bridee par la chaleur ; un marche apres l'autre (multi_m5_05),
    l'ETH attendait la fin du BTC. Ici, pour chaque bloc :

      1. chaque marche prepare ses donnees et son expert LightGBM, l'un
         apres l'autre ;
      2. l'imitation alterne : passe 1 du BTC, passe 1 de l'ETH, passe 2 du
         BTC... ;
      3. chaque epoch entraine et valide le BTC, puis l'ETH, puis rejoue
         leurs coups de validation sur le COMPTE COMMUN (`bilan_commun`) ;
      4. le test : chaque marche avec SON meilleur modele, puis le compte
         commun.

    Chaque modele reste independant (`modeles_par_marche`) : son tronc, ses
    tetes, ses optimiseurs, sa normalisation, son expert, son hasard, et il
    ne se sauvegarde que s'il bat SON record."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifeste = f"run_{cfg.prefixe}.json"
    if os.path.exists(manifeste):
        raise FileExistsError(f"{manifeste} existe : nouveau prefixe, ou lancer.ps1")
    with open(manifeste, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=1, default=str)
    marches = tuple(cfg.marches)
    specs = {m: spec_marche(m, cfg) for m in marches}
    print("=" * 70)
    print(f"  KAIROS EN JEU — {len(marches)} MODELES INDEPENDANTS M{cfg.minutes_par_barre}, "
          f"PAS A PAS, UN COMPTE COMMUN  |  VALIDATION CROISEE PURGEE en {cfg.n_blocs} blocs")
    print("=" * 70)
    print(f"  marches : {', '.join(marches)}  |  chacun son tronc, ses tetes, ses optimiseurs, "
          f"son expert, sa normalisation, SON meilleur modele  |  {device}", flush=True)
    print(f"  regles de chaque modele : {cfg.jetons} coups par jour, {cfg.positions_max} position(s) "
          f"a la fois, fin de partie a -{cfg.vie_R:g} R, porte {cfg.porte_rang_expert:g}, expert "
          f"{cfg.expert_k} coups  |  compte commun : {cfg.capital:.0f} $, {cfg.risque_pct:g} % du "
          f"solde commun du moment par coup, marge partagee (levier 1:{cfg.levier:.0f})", flush=True)
    torch.manual_seed(cfg.graine)
    d_all = pd.read_pickle(cfg.cache)
    M = {}
    for m in marches:
        cm = config_marche(cfg, m)
        d, cm = donnees_marche_seul(d_all, cm)
        cols = list(dict.fromkeys(["time", "open", "high", "low", "close", "atr_14",
                                   "spread_bar"] + colonnes_jeu(cm)))
        d = d[cols].reset_index(drop=True)
        N = len(d)
        o, h, l, c = (d[x].to_numpy(np.float64) for x in ("open", "high", "low", "close"))
        sp = d["spread_bar"].to_numpy(np.float64)
        atr = atr_effectif(d["atr_14"].to_numpy(np.float64), c, cm)
        cout_med = float(np.median(sp)) + cm.glissement_entree_bps + cm.glissement_sortie_bps
        atr = np.maximum(atr, cm.plancher_couts * cout_med * 1e-4 * c)
        R0, D0, S0 = table_coups(o, h, l, sp, atr, cm, 0.0)
        R1, D1, S1 = table_coups(o, h, l, sp, atr, cm, 1.0)
        M[m] = {"cfg": cm, "temps": d["time"], "N": N, "c": c, "atr": atr, "sp": sp,
                "o": o, "h": h, "l": l,
                "t_ns": d["time"].values.astype("int64"),
                "X": d[colonnes_jeu(cm)].to_numpy(np.float32),
                "marge": fraction_marge(c, atr, cm), "R0": R0, "D0": D0, "R1": R1, "D1": D1,
                "S0": S0, "S1": S1, "y_ex": cibles_expert(R1, cm),
                "toutes": parties(d["time"], 0, N, cm),
                "rng": np.random.default_rng(cfg.graine)}
        print(f"  {m} : {N:,} bougies {d['time'].iloc[0]:%Y-%m-%d} -> {d['time'].iloc[-1]:%Y-%m-%d}, "
              f"cout med {cout_med:.2f} bps, ATR des barrieres >= "
              f"{max(float(cm.atr_min_bps), cm.plancher_couts * cout_med):.1f} bps, swap "
              f"{cm.swap_achat_bps_jour:.2f} / {cm.swap_vente_bps_jour:.2f} bps/jour, contrat "
              f"{cm.contrat:g}, lot min {cm.lot_min:g}", flush=True)
    del d_all
    purge = purge_barres(cfg)
    tests = {}
    for k in range(cfg.n_blocs):
        B = {}
        for m in marches:
            S, cm = M[m], M[m]["cfg"]
            N, t_ns = S["N"], S["t_ns"]
            permis, (te0, te1), (va0, va1) = masque_blocs(N, cfg.n_blocs, k, purge)
            fin_tr = prochain_exclu(permis)
            dedans = lambda a0, a1: np.array([w for w in S["toutes"] if w[0] >= a0 and w[1] <= a1],
                                             np.int64).reshape(-1, 2)
            j_tr = np.array([w for w in S["toutes"] if permis[w[0]:w[1]].all()],
                            np.int64).reshape(-1, 2)
            j_va, j_te = dedans(va0, va1), dedans(te0, te1)
            fm = lambda i: pd.Timestamp(S["temps"].iloc[min(i, N - 1)])
            print(f"\n--- Fold {k + 1} : test {fm(te0):%Y-%m-%d} -> {fm(te1 - 1):%Y-%m-%d} "
                  f"({len(j_te)} parties)  validation {fm(va0):%Y-%m-%d} -> {fm(va1 - 1):%Y-%m-%d} "
                  f"({len(j_va)} parties)  entrainement sur tout le reste ({len(j_tr)} parties) "
                  f"---   [{m}]", flush=True)
            X = S["X"]
            st_m = X[permis].astype(np.float64).mean(0)
            st_s = X[permis].astype(np.float64).std(0)
            Xn = safe_normalize(X, {"mean": st_m.astype(np.float32),
                                    "std": st_s.astype(np.float32)}).astype(np.float32)
            t_ex = time.time()
            pred, _ = expert_multi(Xn, S["y_ex"], t_ns, permis, fin_tr, cm)
            FE = features_expert(pred, fenetre=10_000 // int(cm.minutes_par_barre))
            rangs = FE[:, 2:4].copy()
            m_e = FE[permis].astype(np.float64).mean(0)
            s_e = FE[permis].astype(np.float64).std(0)
            FEn = np.clip((FE - m_e) / (s_e + 1e-8), -5.0, 5.0).astype(np.float32)
            Xk = np.concatenate([Xn, FEn[:, :cm.n_expert]], axis=1)
            ics = []
            for s_ in range(2):
                ok = np.zeros(N, bool)
                ok[te0:te1] = True
                ok &= np.isfinite(S["y_ex"][:, s_])
                ics.append(pd.Series(pred[ok, s_]).corr(pd.Series(S["y_ex"][ok, s_]),
                                                        method="spearman"))
            print(f"  expert  LightGBM appris sur les autres blocs, {time.time() - t_ex:.0f} s ; "
                  f"correlation sur le bloc de test : achat {ics[0]:+.3f}, vente {ics[1]:+.3f}"
                  f"   [{m}]", flush=True)
            policy = PolitiqueJeu(cm).to(device)
            optims = optimiseurs(policy, cm)
            pos = coups_expert_predits(j_tr, pred, S["D1"], fin_tr, cm)
            prep = prepare_imitation(pos, j_tr, fin_tr, cm, etiquette=m)
            B[m] = {"te": (te0, te1), "va": (va0, va1), "fin_tr": fin_tr, "j_tr": j_tr,
                    "j_va": j_va, "j_te": j_te, "Xk": Xk, "rangs": rangs, "policy": policy,
                    "optims": optims, "prep": prep, "record": 0.0, "garde": None,
                    "best": f"best_{cm.prefixe}_bloc{k + 1}.pth",
                    "last": f"last_{cm.prefixe}_bloc{k + 1}.pth"}
            del Xn, FE, FEn, pred
        # L'IMITATION, EN ALTERNANCE : passe 1 de chacun, puis passe 2...
        for ep in range(cfg.expert_epochs):
            for m in marches:
                b = B[m]
                passe_imitation(b["policy"], b["optims"], b["prep"], b["Xk"], M[m]["cfg"],
                                device, M[m]["rng"], ep)
        for m in marches:
            B[m]["policy"].eval()
            del B[m]["prep"]
        ref = marches[0]
        fr = lambda i: pd.Timestamp(M[ref]["temps"].iloc[min(i, M[ref]["N"] - 1)])
        for epoch in range(0, cfg.epochs + 1):
            frac = min(1.0, max(0.0, (epoch - 1) / cfg.rampe_cout)) if cfg.rampe_cout > 0 else 1.0
            parts, sauves = {}, []
            for m in marches:
                S, b, cm = M[m], B[m], M[m]["cfg"]
                t_ep = time.time()
                if epoch >= 1:
                    Rf, Df, Sf = table_a_cout(
                        frac, (S["R0"], S["D0"], S["S0"]), (S["R1"], S["D1"], S["S1"]),
                        lambda f: table_coups(S["o"], S["h"], S["l"], S["sp"], S["atr"], cm, f))
                    choix = S["rng"].choice(len(b["j_tr"]),
                                            size=min(cfg.parties_par_epoch, len(b["j_tr"])),
                                            replace=False)
                    marge_d = max(240 // int(cm.minutes_par_barre), cm.barres_par_partie // 6)
                    sc, cp, tr = joue(b["policy"], departs_tires(b["j_tr"][choix], S["rng"], marge_d),
                                      b["Xk"], Rf, Df, Sf, b["fin_tr"], cm, device,
                                      explore=True, collecte=True, rangs=b["rangs"], marge=S["marge"])
                    maj_ppo(b["policy"], b["optims"], avantages(tr, cm), b["Xk"], cm, device, S["rng"])
                    del Rf, Df, Sf, tr
                gen = torch.Generator(device=device)
                gen.manual_seed(cfg.graine)
                sv, cv, _ = joue(b["policy"], b["j_va"], b["Xk"], S["R1"], S["D1"], S["S1"],
                                 b["va"][1], cm, device, explore=False, gen=gen, rangs=b["rangs"],
                                 marge=S["marge"])
                bv = bilan(sv, cv, S["c"], S["atr"], S["sp"], cm)
                nom = "EXPERT IMITE" if epoch == 0 else f"cout {100 * frac:.0f}%"
                print(f"\nEPOCH {epoch:03d}  {nom:>12}  VAL  {ligne_bilan(bv, cm)}  "
                      f"{(time.time() - t_ep) / 60:.1f} min   [{m}]", flush=True)
                print(f"  bilan  {ligne_detail(bv, cm)}   [{m}]", flush=True)
                etat = {"modele": b["policy"].state_dict(), "config": asdict(cm), "epoch": epoch,
                        "bloc": k + 1}
                torch.save(etat, b["last"])
                if bv["coups"] >= cm.min_coups_val and bv["score"] > b["record"]:
                    b["record"], b["garde"] = bv["score"], epoch
                    torch.save(etat, b["best"])
                    sauves.append(m)
                    print(f"  sauvegarde  NOUVEAU MEILLEUR : {bv['score']:+.3f} R/partie en "
                          f"validation, {bv['coups']} coups -> {b['best']}   [{m}]", flush=True)
                else:
                    print(f"  garde  rien de sauvegarde : {bv['score']:+.3f} R/partie ne bat pas "
                          f"{b['record']:+.3f}" + (f" (meilleur : epoch {b['garde']})"
                                                   if b["garde"] is not None else "")
                          + f"   [{m}]", flush=True)
                parts[m] = table_trades(cv, S["t_ns"], S["c"], S["atr"], cm)
            nj = max(len(B[m]["j_va"]) for m in marches)
            bc = bilan_commun(parts, specs, nj, cfg)
            va0, va1 = B[ref]["va"]
            print(f"\nEPOCH {epoch:03d}  COMPTE COMMUN  VAL bloc {k + 1} ({fr(va0):%Y-%m-%d} -> "
                  f"{fr(va1 - 1):%Y-%m-%d})  |  modeles sauvegardes a cette epoch : "
                  f"{', '.join(sauves) if sauves else 'aucun'}", flush=True)
            print(f"  bilan  COMPTE COMMUN  {ligne_commune(bc, cfg)}", flush=True)
            print(f"  bilan  COMPTE COMMUN par marche : {ligne_commune_marches(bc)}", flush=True)
        parts_t = {}
        for m in marches:
            S, b, cm = M[m], B[m], M[m]["cfg"]
            src = b["best"] if os.path.exists(b["best"]) else b["last"]
            b["policy"].load_state_dict(torch.load(src, map_location=device,
                                                   weights_only=False)["modele"])
            gen = torch.Generator(device=device)
            gen.manual_seed(cfg.graine)
            s_t, c_t, _ = joue(b["policy"], b["j_te"], b["Xk"], S["R1"], S["D1"], S["S1"],
                               b["te"][1], cm, device, explore=False, gen=gen, rangs=b["rangs"],
                               marge=S["marge"])
            bt = bilan(s_t, c_t, S["c"], S["atr"], S["sp"], cm)
            print(f"\nTEST fold {k + 1} ({os.path.basename(src)})  {ligne_bilan(bt, cm)}   [{m}]",
                  flush=True)
            print(f"TEST fold {k + 1} bilan  {ligne_detail(bt, cm)}   [{m}]", flush=True)
            with open(f"test_{cm.prefixe}_bloc{k + 1}.json", "w", encoding="utf-8") as fh:
                json.dump({a: v for a, v in bt.items()}, fh, indent=1, default=str)
            parts_t[m] = table_trades(c_t, S["t_ns"], S["c"], S["atr"], cm)
        nj = max(len(B[m]["j_te"]) for m in marches)
        bt = bilan_commun(parts_t, specs, nj, cfg)
        te0, te1 = B[ref]["te"]
        tests[k + 1] = (fr(te0), fr(te1 - 1), bt, parts_t)
        print(f"\nTEST fold {k + 1} bilan  COMPTE COMMUN ({fr(te0):%Y-%m-%d} -> {fr(te1 - 1):%Y-%m-%d}, "
              f"chaque marche avec SON meilleur modele)  {ligne_commune(bt, cfg)}", flush=True)
        print(f"TEST fold {k + 1} bilan  COMPTE COMMUN par marche : {ligne_commune_marches(bt)}",
              flush=True)
        with open(f"test_{cfg.prefixe}_bloc{k + 1}.json", "w", encoding="utf-8") as fh:
            json.dump({a: v for a, v in bt.items()}, fh, indent=1, default=str)
        del B
    print("\n" + "=" * 70)
    print(f"  RESUME DES {cfg.n_blocs} BLOCS DE TEST, COMPTE COMMUN (chacun jamais vu par ses modeles)")
    print("=" * 70)
    for k in sorted(tests):
        a, z, b, _p = tests[k]
        print(f"  bloc {k:2d}  {a:%Y-%m-%d} -> {z:%Y-%m-%d}  {b.get('coups', 0):5d} trades  WR "
              f"{100 * b.get('win_rate', 0):4.1f} %  PF {b.get('pf', float('nan')):5.2f}  total "
              f"{b.get('total_dollars', 0.0):+9.2f} $  DD {100 * b.get('dd_pct', 0.0):+.1f} %  |  "
              f"{ligne_commune_marches(b)}", flush=True)
    positifs = sum(1 for *_x, b, _p in tests.values() if b.get("total_dollars", 0.0) > 0)
    tous = {m: pd.concat([p[m] for *_x, p in tests.values()], ignore_index=True) for m in marches}
    nj = sum(b.get("n_jours", 0) for *_x, b, _p in tests.values())
    bt = bilan_commun(tous, specs, nj, cfg)
    print(f"  blocs gagnants : {positifs}/{len(tests)}  |  total {bt.get('total_dollars', 0.0):+.2f} $  |  "
          f"{bt.get('coups', 0)} trades  |  profit factor global {bt.get('pf', float('nan')):.2f}",
          flush=True)
    print(f"  tous les blocs a la suite, COMPTE COMMUN : {ligne_commune(bt, cfg)}", flush=True)
    print(f"  tous les blocs, par marche : {ligne_commune_marches(bt)}", flush=True)
    print("\nFIN du walk-forward", flush=True)
    return 0


def coupe_debut_commun(d: pd.DataFrame) -> pd.DataFrame:
    """Les lignes a partir de la premiere bougie du marche le plus recent :
    tous les marches commencent au meme instant. Voir `debut_commun`."""
    t0 = d.groupby("marche")["time"].min().max()
    return d[d["time"] >= t0]


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
    if getattr(cfg, "debut_commun", False):
        d = coupe_debut_commun(d).copy()
        print(f"  debut commun : {d['time'].min():%Y-%m-%d %H:%M} UTC, la premiere bougie du "
              f"marche le plus recent ; tous les marches s'entrainent et se testent sur la "
              f"meme periode", flush=True)
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
    def _calcule_table(frac):
        parts = [table_coups(o[a:b], h[a:b], l[a:b], sp[a:b], atr[a:b], _cfg_m(a), frac)
                 for _, a, b in blocs]
        return tuple(np.concatenate([p_[k] for p_ in parts]) for k in range(3))

    t_tab = time.time()
    R0, D0, S0 = _calcule_table(0.0)
    R1, D1, S1 = _calcule_table(1.0)
    y_ex = cibles_expert(R1, cfg)
    print(f"  table des coups : {N:,} bougies x {R1[0].size} coups, marche par marche, "
          f"{time.time() - t_tab:.0f} s", flush=True)
    toutes = journees_multi(d["time"], blocs, int(t_ns.min()), int(t_ns.max()) + 1,
                            int(0.4 * cfg.barres_par_partie), cfg)
    purge = purge_barres(cfg)
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
                Rf, Df, Sf = table_a_cout(frac, (R0, D0, S0), (R1, D1, S1), _calcule_table)
                choix = rng.choice(len(j_tr), size=min(cfg.parties_par_epoch, len(j_tr)),
                                   replace=False)
                marge_d = max(240 // int(cfg.minutes_par_barre), cfg.barres_par_partie // 6)
                sc, cp, tr = joue(policy, departs_tires(j_tr[choix], rng, marge_d), Xk,
                                  Rf, Df, Sf, fin_tr, cfg, device, explore=True,
                                  collecte=True, rangs=rangs_ex, prix=c, atr=atr)
                maj_ppo(policy, optims, avantages(tr, cfg), Xk, cfg, device, rng)
                del Rf, Df, Sf
            gen = torch.Generator(device=device)
            gen.manual_seed(cfg.graine)
            sv, cv, _ = joue(policy, j_va, Xk, R1, D1, S1, fin_va, cfg, device,
                             explore=False, gen=gen, rangs=rangs_ex, prix=c, atr=atr)
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
                           explore=False, gen=gen, rangs=rangs_ex, prix=c, atr=atr)
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
    if "--rebuild-deploy-ensemble" in sys.argv:
        return reconstruit_deploy_ensemble(cfg)
    # LE MODELE D'UN MARCHE, lance par `main_modeles_par_marche`.
    if len(sys.argv) > 1 and tuple(cfg.marches):
        return main_blocs(config_marche(cfg, sys.argv[1],
                                        int(sys.argv[2]) if len(sys.argv) > 2 else 0))
    if tuple(cfg.marches) and getattr(cfg, "modeles_par_marche", False):
        return main_pas_a_pas(cfg)
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
    R0, D0, S0 = table_coups(o, h, l, sp, atr, cfg, 0.0)
    R1, D1, S1 = table_coups(o, h, l, sp, atr, cfg, 1.0)
    print(f"  table des coups : {N:,} bougies x {R1[0].size} coups, sans cout et "
          f"au cout reel, {time.time() - t0:.0f} s", flush=True)

    # L'EXPERT, APPRIS SUR LE TRAIN DU FOLD 1 SEULEMENT. Voir l'en-tete.
    t_ex = time.time()
    y_ex = cibles_expert(R1, cfg)
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
                Rf, Df, Sf = table_a_cout(frac, (R0, D0, S0), (R1, D1, S1),
                                         lambda f: table_coups(o, h, l, sp, atr, cfg, f))
                choix = rng.choice(len(j_tr), size=min(cfg.parties_par_epoch, len(j_tr)),
                                   replace=False)
                marge_d = max(240 // int(cfg.minutes_par_barre), cfg.barres_par_partie // 6)
                sc, cp, tr = joue(policy, departs_tires(j_tr[choix], rng, marge_d), Xn,
                                  Rf, Df, Sf, a_va, cfg, device, explore=True,
                                  collecte=True, rangs=rangs_ex, marge=marge)
                b_tr = bilan(sc, cp, c, atr, sp, cfg, frac=frac)
                st = maj_ppo(policy, optims, avantages(tr, cfg), Xn, cfg, device, rng)
                del Rf, Df, Sf
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

    # LE SWAP DE CHAQUE MARCHE, lu dans le cache (voir `prepare_multi_h1`).
    def _cfg_m(a):
        if "swap_achat_bps_jour" not in d.columns:
            return cfg
        return replace(cfg, swap_achat_bps_jour=float(d["swap_achat_bps_jour"].iloc[a]),
                       swap_vente_bps_jour=float(d["swap_vente_bps_jour"].iloc[a]))

    def _calcule_table(frac):
        parts = [table_coups(o[a:b], h[a:b], l[a:b], sp[a:b], atr[a:b], _cfg_m(a), frac)
                 for _, a, b in blocs]
        return tuple(np.concatenate([p_[k] for p_ in parts]) for k in range(3))

    t_tab = time.time()
    R0, D0, S0 = _calcule_table(0.0)
    R1, D1, S1 = _calcule_table(1.0)
    print(f"  table des coups : {N:,} bougies x {R1[0].size} coups, marche par "
          f"marche, {time.time() - t_tab:.0f} s", flush=True)

    t_ex = time.time()
    fin_tr1 = fin_segment(t_ns, blocs, bornes[0][1])
    y_ex = cibles_expert(R1, cfg)
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
                Rf, Df, Sf = table_a_cout(frac, (R0, D0, S0), (R1, D1, S1), _calcule_table)
                choix = rng.choice(len(j_tr), size=min(cfg.parties_par_epoch, len(j_tr)),
                                   replace=False)
                marge_d = max(240 // int(cfg.minutes_par_barre), cfg.barres_par_partie // 6)
                sc, cp, tr = joue(policy, departs_tires(j_tr[choix], rng, marge_d),
                                  Xn, Rf, Df, Sf, fin_tr, cfg, device, explore=True,
                                  collecte=True, rangs=rangs_ex, prix=c, atr=atr)
                b_tr = bilan(sc, cp, c, atr, sp, cfg, frac=frac, ordre=t_ns)
                st = maj_ppo(policy, optims, avantages(tr, cfg), Xn, cfg, device, rng)
                del Rf, Df, Sf
            gen = torch.Generator(device=device)
            gen.manual_seed(cfg.graine)
            sv, cv, _ = joue(policy, j_va, Xn, R1, D1, S1, fin_va, cfg, device,
                             explore=False, gen=gen, rangs=rangs_ex, prix=c, atr=atr)
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
                           explore=False, gen=gen, rangs=rangs_ex, prix=c, atr=atr)
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
