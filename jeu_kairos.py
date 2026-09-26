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
    prefixe: str = "kairos_jeu_btc01"
    cache: str = "data_cache_BTCUSD_M1.pkl"
    # --- le modele : celui du run PPO, a l'identique ---
    lookback: int = 4
    d_model: int = 8
    num_blocks: int = 2
    heads: int = 1
    n_freq: int = 2
    mlp_dim: int = 32
    # --- les regles ---
    jetons: int = 6
    vie_R: float = 3.0
    # En ATR de la barre de decision. L'ATR M1 du BTC vaut ~7 bps, un
    # mouvement de 15 minutes ~2.5 ATR : les coups vont de la demi-heure a
    # quelques heures.
    tp_atr: Tuple[float, ...] = (2.0, 4.0, 8.0)
    sl_atr: Tuple[float, ...] = (2.0, 4.0, 8.0)
    horizon_max: int = 120
    # Glissements ESPERES (la moitie des bornes de `training.PPOConfig`) :
    # entree toujours, sortie au stop et au temps, jamais a l'objectif.
    glissement_entree_bps: float = 0.5
    glissement_sortie_bps: float = 1.0
    capital: float = 1000.0
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
    expert_k: int = 12
    expert_R_min: float = 1.0
    expert_pas_neg: int = 10
    expert_poids_pos: float = 3.0
    expert_epochs: int = 3
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
    graine: int = 7

    @property
    def risque_dollars(self) -> float:
        return self.capital * self.risque_pct / 100.0


# ======================================================================
# LE MODELE — le SAINT du run, deux tetes de barrieres en plus
# ======================================================================
class PolitiqueJeu(SAINTPolicySingleHead):
    """Le tronc et les tetes d'achat et de vente du run PPO ; les tetes de
    coupure des gains et des pertes choisissent desormais l'objectif et le
    stop, AU MOMENT DU COUP, en lisant le tronc et le sens."""

    def __init__(self, cfg: JeuConfig):
        super().__init__(
            n_features=OBS_N_FEATURES, d_model=cfg.d_model,
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
def journees(temps: pd.Series, debut: int, fin: int) -> np.ndarray:
    """Les journees ENTIERES de [debut, fin) : (n, 2) indices [a, b)."""
    jour = temps.dt.normalize().to_numpy()
    idx = np.arange(len(jour))
    df = pd.DataFrame({"j": jour, "i": idx})
    g = df.groupby("j")["i"].agg(["min", "max"])
    g = g[(g["min"] >= debut) & (g["max"] < fin)]
    # UNE JOURNEE TROP COURTE (trou de donnees) n'est pas une partie.
    g = g[(g["max"] - g["min"]) >= 600]
    return np.stack([g["min"].to_numpy(), g["max"].to_numpy() + 1], 1)


def etat_jeu(jetons, score, reste_min, cfg: JeuConfig) -> np.ndarray:
    """Les cinq colonnes du bloc de position, qui portent l'etat de la partie."""
    n = len(jetons)
    return np.stack([
        jetons / float(cfg.jetons),
        np.clip(score / cfg.vie_R, -1.0, 3.0),
        np.clip(reste_min / 1440.0, 0.0, 1.0),
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


def _masque_logits(le: torch.Tensor, peut: torch.Tensor) -> torch.Tensor:
    m = torch.zeros_like(le)
    m[:, ACHETER] = torch.where(peut, 0.0, _NEG)
    m[:, VENDRE] = torch.where(peut, 0.0, _NEG)
    return le + m


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


def joue(policy, jours: np.ndarray, Xn, R, D, S, fin_valide: int,
         cfg: JeuConfig, device, explore: bool, gen=None,
         collecte: bool = False):
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
        peut_np = (jet[g] > 0) & (tt + 1 + H < fin_valide)
        with torch.no_grad():
            x = torch.from_numpy(ob).to(device)
            peut = torch.from_numpy(peut_np).to(device)
            le, v, ltp, lsl = policy.jeu(x)
            le = _masque_logits(le, peut)
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
                trans[g[q]].append((int(tt[q]), et[q], bool(peut_np[q]),
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
            le = _masque_logits(le, T(peut[b], torch.bool))
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
        ts = ts[ts + 1 + H < fin_valide]
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


def imite_expert(policy, optims, jours, R, D, Xn, fin_valide, cfg, device, rng):
    """L'apprentissage par imitation : l'ENTREE de l'expert, et elle seule.

    Les tetes de barrieres n'y apprennent rien : le choix a posteriori de
    l'expert est biaise vers le coup le plus risque. Voir l'en-tete.
    """
    pos = coups_expert(jours, R, D, fin_valide, cfg)
    n_j = max(len(jours), 1)
    print(f"  expert  {len(pos):,} coups sur {len(jours)} journees "
          f"({len(pos)/n_j:.1f} par jour), {np.mean([p[4] for p in pos]):+.2f} R "
          f"en moyenne a posteriori — c'est l'avenir qu'il lit, pas un "
          f"objectif atteignable", flush=True)
    t_pos = {p[0] for p in pos}
    H, L = int(cfg.horizon_max), int(cfg.lookback)
    neg = []
    for a0, b0 in jours:
        for t0 in range(max(a0, L - 1), b0, cfg.expert_pas_neg):
            if t0 not in t_pos and t0 + 1 + H < fin_valide:
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
          frac: float = 1.0) -> Dict[str, float]:
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
    return b | {
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
            f"temps {100 * so[2]:.0f}%  duree {b['duree']:.0f} min  |  objectif "
            + "/".join(f"{100 * x:.0f}" for x in b["tp"])
            + f"% sur {'/'.join(f'{x:g}' for x in cfg.tp_atr)} ATR  stop "
            + "/".join(f"{100 * x:.0f}" for x in b["sl"])
            + f"% sur {'/'.join(f'{x:g}' for x in cfg.sl_atr)} ATR")


# ======================================================================
# LE WALK-FORWARD
# ======================================================================
def main() -> int:
    cfg = JeuConfig()
    rng = np.random.default_rng(cfg.graine)
    torch.manual_seed(cfg.graine)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifeste = f"run_{cfg.prefixe}.json"
    if os.path.exists(manifeste):
        raise FileExistsError(f"{manifeste} existe : nouveau prefixe, ou lancer.ps1")
    with open(manifeste, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=1, default=str)

    print("=" * 70)
    print("  KAIROS EN JEU — BTCUSD M1 : une journee = une partie")
    print("=" * 70)
    print(f"  regles : {cfg.jetons} coups par partie, fin de partie a "
          f"-{cfg.vie_R:g} R, coup = sens + objectif {cfg.tp_atr} ATR + stop "
          f"{cfg.sl_atr} ATR, temps limite {cfg.horizon_max} min")
    print(f"  un R = {cfg.risque_dollars:.0f}$ ({cfg.risque_pct:g}% de "
          f"{cfg.capital:.0f}$)  |  quatre tetes : achat, vente, gain (objectif), "
          f"perte (stop), chacune son optimiseur  |  {device}", flush=True)

    d = pd.read_pickle(cfg.cache)
    cols = list(dict.fromkeys(["time", "open", "high", "low", "close", "atr_14",
                               "spread_bar"] + list(FEATURE_COLS)))
    d = d[cols].reset_index(drop=True)
    N = len(d)
    o, h, l, c = (d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    atr = d["atr_14"].to_numpy(np.float64)
    sp = d["spread_bar"].to_numpy(np.float64)
    n_tr, n_va, n_te = (int(N * x) for x in (cfg.part_train, cfg.part_val, cfg.part_test))
    pas_wf = n_te
    print(f"[CACHE] {N:,} minutes, {len(FEATURE_COLS)} features, "
          f"{d['time'].iloc[0]} -> {d['time'].iloc[-1]}", flush=True)

    X = d[list(FEATURE_COLS)].to_numpy(np.float32)
    mean = X[:n_tr].astype(np.float64).mean(0)
    std = X[:n_tr].astype(np.float64).std(0)
    stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    Xn = safe_normalize(X, stats).astype(np.float32)
    del X
    np.savez(f"{cfg.prefixe}_norm.npz", mean=stats["mean"], std=stats["std"],
             features=np.array(list(FEATURE_COLS)))
    print(f"  normalisation figee sur le train du fold 1 [0 : {n_tr:,})", flush=True)

    t0 = time.time()
    R0, D0, _ = table_coups(o, h, l, sp, atr, cfg, 0.0)
    R1, D1, S1 = table_coups(o, h, l, sp, atr, cfg, 1.0)
    print(f"  table des coups : {N:,} minutes x 18 coups, sans cout et au cout "
          f"reel, {time.time() - t0:.0f} s", flush=True)

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
        j_tr = journees(d["time"], a_tr, a_va)
        j_va = journees(d["time"], a_va, a_te)
        j_te = journees(d["time"], a_te, f_te)
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
            imite_expert(policy, optims, j_tr, R1, D1, Xn, a_va, cfg, device, rng)
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
                sc, cp, tr = joue(policy, departs_tires(j_tr[choix], rng), Xn,
                                  Rf, Df, S1, a_va, cfg, device, explore=True,
                                  collecte=True)
                b_tr = bilan(sc, cp, c, atr, sp, cfg, frac=frac)
                st = maj_ppo(policy, optims, avantages(tr, cfg), Xn, cfg, device, rng)
                del Rf
            gen = torch.Generator(device=device)
            gen.manual_seed(cfg.graine)
            sv, cv, _ = joue(policy, j_va, Xn, R1, D1, S1, a_te, cfg, device,
                             explore=False, gen=gen)
            bv = bilan(sv, cv, c, atr, sp, cfg)
            nom = "EXPERT IMITE" if epoch == 0 else f"cout {100 * frac:.0f}%"
            print(f"\nEPOCH {epoch:03d}  {nom:>12}  VAL  {ligne_bilan(bv, cfg)}  "
                  f"{(time.time() - t_ep) / 60:.1f} min", flush=True)
            print(f"  jeu  validation  {ligne_style(bv, cfg)}", flush=True)
            if st is not None:
                print(f"  jeu  entrainement  {ligne_bilan(b_tr, cfg)}", flush=True)
                print(f"  jeu  PPO  {st['n']:,} decisions ({st['n_coups']:,} coups, "
                      f"sur {st['n_total']:,})  H entree {st['H']:.3f}/1.099  "
                      f"H barrieres {st['Hb']:.3f}/1.099  KL {st['kl']:+.4f}  "
                      f"clip {100 * st['clip']:.0f}%  critique {st['v']:.4f}", flush=True)
            etat = {"modele": policy.state_dict(), "config": asdict(cfg),
                    "features": list(FEATURE_COLS), "mean": stats["mean"],
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
                           explore=False, gen=gen)
        bt = bilan(s_t, c_t, c, atr, sp, cfg)
        print(f"\nTEST fold {fold + 1} ({os.path.basename(src)})  {ligne_bilan(bt, cfg)}",
              flush=True)
        print(f"TEST fold {fold + 1}  {ligne_style(bt, cfg)}", flush=True)
        with open(f"test_{cfg.prefixe}{suffixe}.json", "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in bt.items()}, fh, indent=1, default=str)
        precedent = src
    print("\nFIN du walk-forward", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
