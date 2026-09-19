"""Veille : analyse chaque epoch d'entrainement des qu'elle apparait.

STRICTEMENT EN LECTURE SEULE. Ce script ouvre le journal, rien d'autre. Il
n'interroge aucun processus, n'en signale aucun, n'en arrete aucun. Cette
insistance n'est pas de la prudence gratuite : un veilleur precedent, cense
etre en lecture seule, a supprime un entrainement a l'epoch 7 parce que
os.kill(pid, 0) — le test de vie POSIX — TUE le processus sous Windows, le
signal devenant le code de sortie. Le fermer, ou l'interrompre par Ctrl+C,
n'a aucun effet sur l'entrainement.

    python veille_epochs.py

Ce qu'il ajoute a un simple `tail` : il calcule ce que le journal n'affiche
pas — profit par trade, winrate d'equilibre recalcule sur les gains et pertes
REELS de VALIDATION, ecart au point mort, erreur-type de cet ecart, PnL cumule du
run, ecart entre entrainement et validation — et il lit les diagnostics a voix
haute plutot que de les laisser sous forme de colonnes.

Il ecrit aussi un rapport cumulatif dans analyse_epochs.md.

QUATRE DEFAUTS DE FORMAT, tous payes une fois.

  1. `trades=(\\d+)` supposait qu'il n'y ait pas d'espace apres le signe egal.
     Le champ est cadre a droite sur quatre caracteres : `trades=1482` en M1,
     mais `trades= 374` en H1. La veille etait muette sans dire pourquoi.

  2. `[+-]?` manquait devant les nombres signes. Le journal ecrit
     `PNL  -147.55$` mais aussi `PNL  +667.73$`. Un motif qui n'accepte que le
     moins ne rate pas des epochs au hasard : il rate PRECISEMENT LES
     GAGNANTES, et le run parait uniformement perdant.

  3. LES EPOCHS ETAIENT CONFONDUES ENTRE FOLDS. Le walk-forward repart a
     l'epoch 1 a chaque fold ; en ne retenant que le NUMERO, la veille tenait
     les epochs du fold 2 pour deja vues. On retient (fold, epoch).

  4. LA LIGNE DU HASARD ETAIT PERIMEE, et le resterait toujours si on la
     recopiait : 26.7 % mesure sur M1 a 240 barres de detention, 40.9 % en H1
     aux memes barrieres. La reference est desormais LUE DANS LE RUN — pendant
     le warmup du critic l'actor est gele et la politique uniforme (H = ln 3,
     clipfrac 0), donc ces epochs SONT le tirage au hasard, sur la meme
     fenetre, le meme moteur, la meme friction et le meme calibrage. Rien a
     tenir a jour, et elle suit tout changement d'echelle ou de geometrie.
"""

import os
import re
import sys
import time
import math
import datetime as dt

JOURNAL = "training_btc.log"
RAPPORT = "analyse_epochs.md"
PAS_SONDAGE = 20             # secondes

# Deplacement typique du seuil calibre d'une epoch a l'autre, mesure sur les
# epochs a actor gele (meme reseau, donc tout ecart vient du calibrage seul) :
# amplitude 0.0010 sur les cinq premieres epochs d'exec12.
TREMBLEMENT_SEUIL = 0.0010

ANSI = re.compile(r"\x1b\[[0-9;]*m")
NB = r"[+-]?[\d.]+"

# Les trois lignes que le training ecrit par epoch. Chaque motif est ancre sur
# le fold pour que deux folds ne se confondent pas.
RE_VAL = re.compile(
    r"\[(?:BOTH|LONG|SHORT)_(wf\d+)\]\s+EPOCH (\d+)\s+VAL\s+PNL\s+(" + NB + r")\$\s+"
    r"trades=\s*(\d+)\s+WR\s+([\d.]+)%\s+PF\s+([\d.]+)\s+DD\s+([\d.]+)%"
    r".*?L\((\d+)W/(\d+)L\)\s+(" + NB + r")\$"
    r"\s+S\((\d+)W/(\d+)L\)\s+(" + NB + r")\$")
RE_TRAIN = re.compile(
    r"\[(?:BOTH|LONG|SHORT)_(wf\d+)\]\s+EPOCH (\d+)\s+TRAIN\s+PNL\s+(" + NB + r")\$\s+"
    r"trades=\s*(\d+)\s+WR\s+([\d.]+)%\s+PF\s+([\d.]+)\s+DD\s+([\d.]+)%")
RE_META = re.compile(
    # `rho` s'intercale entre META et Sortino depuis le 2026-09-16. Le groupe
    # est NON CAPTURANT : la suite du fichier lit m[2] a m[22] par position, et
    # un groupe de plus les decalerait tous en silence. Il est OPTIONNEL pour
    # que la veille continue de lire les journaux des runs anterieurs.
    r"\[(?:BOTH|LONG|SHORT)_(wf\d+)\]\s+EPOCH (\d+)\s+META\s+"
    # ON N'ENUMERE PLUS LES CHAMPS QUI PRECEDENT SORTINO. Ils ont ete quatre
    # a s'inserer la — `rho`, `rhoAux`, `[val epN]`, `sommet` — et un
    # CINQUIEME, `table`, a casse ce motif le 2026-09-19. A chaque fois le
    # symptome est le meme : la veille se tait, sans rien signaler, et on
    # cherche ailleurs.
    #
    # `.*?` tolere n'importe quel champ futur. Il ne peut pas deraper : le
    # motif est ancre sur `[COTE_wfN] EPOCH n META` en tete, il est NON
    # GOURMAND donc il s'arrete au PREMIER `Sortino` suivi d'une espace, et
    # `Sortino30` n'en est pas un — le `3` suit immediatement.
    #
    # Les champs qu'on veut LIRE se lisent par des motifs separes, comme
    # `RE_RHO` et `RE_SOMMET` : c'est la seule facon d'en ajouter sans
    # decaler les indices que `analyse` lit par position.
    r".*?Sortino\s+(" + NB + r").*?"
    r"AvgW\s+(" + NB + r")\$\s+AvgL\s+(" + NB + r")\$.*?H ([\d.]+).*?"
    r"sel\[train\s+([\d.]+)% val\s+([\d.]+)%\].*?"
    r"etendue\[tr ([\d.-]+) val ([\d.-]+)\].*?"
    r"KL\s+(" + NB + r").*?gnorm\s+([\d.]+)\s+"
    r"g\[actor ([\d.e+-]+).*?clipfrac ([\d.]+)%\s+"
    r"temps\[collecte (\d+)s maj PPO (\d+)s calibration (\d+)s "
    r"validation (\d+)s\].*?gpu\[(\d+)C (\d+)/"
    r".*?ENV \[B\s+([\d.]+)% S\s+([\d.]+)% H\s+([\d.]+)%\]")

# Le critere de SELECTION depuis le 2026-09-19 : ce que rapportent, en
# unites de risque, les occasions que le checkpoint mettrait en position, et
# ce que rapporte une occasion au hasard. Lu par un motif separe pour la
# meme raison que rho — ne pas decaler les indices de `analyse`.
RE_SOMMET = re.compile(r"sommet\s+(" + NB + r")R/(" + NB + r")R")

# Combien de scores sont venus de la table groupee, et combien ont du
# repasser par un forward. Le second doit rester petit : il compte les
# barres ou l'etat de l'environnement n'etait pas celui que la table
# suppose, et chacune coute un appel reseau complet.
RE_TABLE = re.compile(r"table\s+(\d+)/(\d+)")

# LE MARQUEUR DE REPRISE. La validation ne tourne qu'une epoch sur
# `validation_tous_les` ; les autres, l'entrainement REAFFICHE la derniere
# mesure et le signale par `[val epN]`. Sans lire ce marqueur, la veille
# presente trois fois le meme resultat comme trois mesures — et pire, elle
# l'ADDITIONNE trois fois au cumul du run.
RE_REPRISE = re.compile(r"\[val ep(\d+)\]")

RE_RHO = re.compile(r"META\s+rho\s+(" + NB + r")")
RE_RHO_AUX = re.compile(r"rhoAux\s+(" + NB + r")")

# Seuil sous lequel le gradient de l'acteur est considere comme NUL, donc la
# politique reellement gelee. Mesure : pendant le warmup du critic il vaut
# 1e-5 a 1e-7, et des la premiere epoch entrainee il saute a 3e-1. Deux ordres
# de grandeur separent les deux regimes, le seuil n'a donc rien de delicat.
GRADIENT_NUL = 1e-3

# LE PLAFOND D'ENTROPIE N'EST PLUS ln 3, corrige le 2026-09-16.
#
# CE QUI A CHANGE. Le veto de TabM masque une direction avant le softmax :
# quand il interdit l'achat, la politique choisit entre VENDRE et ATTENDRE,
# donc son entropie maximale vaut ln 2 et non ln 3. Sur exec29 l'entropie de
# warmup affiche 0.573 la ou elle valait 1.099 — et lue contre ln 3 elle
# ressemblerait a une politique DEJA fortement differenciee, alors qu'elle
# decide encore au hasard.
#
# C'est la meme faute que le point mort calcule sur la mauvaise fenetre :
# un chiffre juste, compare au mauvais repere.
#
# LE PLAFOND ATTENDU, le veto laissant passer une part p de chaque direction
# independamment :
#
#     les deux permises   p^2          -> ln 3
#     une seule           2 p (1-p)    -> ln 2
#     aucune              (1-p)^2      -> 0, il ne reste qu'ATTENDRE
#
# A p = 0.50 cela fait 0.621. C'est une esperance, pas une borne : une epoch
# peut la depasser si le veto a moins mordu que prevu. Elle sert de repere,
# comme 1.099 servait de repere avant.
def _plafond_entropie(cote: str = "both") -> float:
    """L'entropie maximale que la politique PEUT atteindre, a ce cote.

    LE PLAFOND DEPEND DU NOMBRE D'ACTIONS PERMISES, corrige le 2026-09-19 —
    et l'oubli rendait le diagnostic aveugle.

    A plat, un run BILATERAL choisit entre ACHETER, VENDRE et ATTENDRE :
    trois actions, plafond ln 3 = 1.099. Un run LONG-ONLY n'en a que deux,
    ACHETER et ATTENDRE : plafond ln 2 = 0.693.

    CE QUE L'ERREUR COUTAIT. or_exec06 affiche H = 0.693 aux deux premieres
    epochs. Compare a ln 3, cela fait 63 % du plafond — une politique deja
    bien differenciee. Compare a ln 2, cela fait 100 % : elle choisit entre
    ACHETER et ATTENDRE A PILE OU FACE, elle n'a rien differencie du tout.
    Et le diagnostic « APPREND MAIS RESTE PLAT », declenche au-dessus de
    0.99 x plafond, ne pouvait structurellement jamais tirer.

    C'est la meme faute que le point mort calcule sur la mauvaise fenetre,
    et elle est notee dans ce fichier depuis le 2026-09-16 : un chiffre
    juste, compare au mauvais repere.
    """
    n_actions = 3 if cote in ("both", "duel") else 2
    try:
        from training import PPOConfig as _C
        if not getattr(_C(), "votant_tabm", False):
            return math.log(n_actions)
        from tabm_votant import PART_LAISSEE as p
    except Exception:
        return math.log(n_actions)
    if n_actions == 2:
        # Le veto ne peut interdire que le seul cote ouvert ; il ne reste
        # alors qu'ATTENDRE, d'entropie nulle.
        return p * math.log(2)
    return (p * p * math.log(3) + 2 * p * (1 - p) * math.log(2))


# Valeur par defaut, bilaterale. `analyse` recalcule le plafond avec le
# COTE lu dans le journal : c'est lui qui decide du nombre d'actions.
H_MAX = _plafond_entropie()

# LE WARMUP SE LIT SUR LE NUMERO D'EPOCH, PAS SUR UN SEUIL, corrige le
# 2026-09-16. C'est la CONFIGURATION qui decide quelles epochs gelent l'actor,
# donc c'est elle qu'il faut lire — pas une empreinte indirecte.
#
# CE QUI A RENDU LE SEUIL INTENABLE. Le gradient d'acteur pendant le warmup
# valait 1e-5 a 1e-6 avec un reseau unique ; avec l'ensemble d'exec28 il monte
# a 1.2e-4 puis 5.0e-4 sur les deux premieres epochs — le bonus d'entropie
# n'est pas annule pendant le warmup, et il porte maintenant sur deux reseaux.
# Encore un facteur deux et le seuil de 1e-3 est franchi : des epochs GELEES
# seraient comptees comme entrainees, et la reference du hasard — celle contre
# laquelle tout le reste se juge — serait polluee par des epochs qui ne
# mesurent rien.
#
# Le numero d'epoch, lui, est exact par construction : `training.py` gele
# l'actor tant que `epoch <= critic_warmup_epochs`. Le gradient reste affiche,
# et sert desormais de CONTRE-VERIFICATION : si les deux se contredisent, la
# veille le dit au lieu de choisir silencieusement.
try:
    from training import PPOConfig as _Cfg
    WARMUP = int(_Cfg().critic_warmup_epochs)
except Exception:
    WARMUP = 5


class Couleur:
    VERT, ROUGE, JAUNE, CYAN, GRIS, GRAS, FIN = (
        "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[90m",
        "\033[1m", "\033[0m")


C = Couleur


def erreur_type(wr, avg_w, avg_l, trades):
    """Erreur-type du gain par trade, en dollars — un PLANCHER de bruit.

    Avec un winrate p, un gain moyen w et une perte moyenne l, la variance par
    trade vaut p(1-p)(w+l)^2 si l'on reduit gains et pertes a leurs moyennes.
    C'est une SOUS-ESTIMATION : les gains varient entre eux, les pertes aussi.
    Un ecart inferieur a ce chiffre n'est surement pas separable de zero ; un
    ecart superieur ne l'est pas forcement.

    Ce calcul remplace un plancher constant de 2.2 $/trade releve sur trois
    epochs 1 en M1. Un plancher fige vieillit exactement comme la ligne du
    hasard : il reste tandis que l'echelle, les barrieres et le nombre de
    trades changent sous lui.
    """
    p = max(min(wr / 100.0, 1.0), 0.0)
    if trades <= 1 or (avg_w + avg_l) <= 0:
        return float("nan")
    return (avg_w + avg_l) * math.sqrt(p * (1 - p) / trades)


def _point_mort(avg_w, avg_l):
    return (100.0 * avg_l / (avg_w + avg_l)) if (avg_w + avg_l) > 0 else float("nan")


def analyse(v, m, tr, precedent, reference, cumul, moyenne=False,
            ent_prec=None, rho=None, cote="both", herite=False,
            sommet=None, reprise=None):
    """Rend (lignes colorees, lignes brutes, gain par trade, ecart, gele)."""
    ep = int(v[1])
    pnl, trades, wr, pf, dd = (float(v[2]), int(v[3]), float(v[4]),
                               float(v[5]), float(v[6]))
    lw, ll, lpnl = int(v[7]), int(v[8]), float(v[9])
    sw, sl_, spnl = int(v[10]), int(v[11]), float(v[12])

    sortino, avg_w, avg_l = float(m[2]), float(m[3]), abs(float(m[4]))
    H = float(m[5])
    sel_tr, sel_val = float(m[6]), float(m[7])
    et_tr, et_val = float(m[8]), float(m[9])
    kl, gnorm, g_actor = float(m[10]), float(m[11]), float(m[12])
    clipfrac = float(m[13])
    t_col, t_ppo, t_cal, t_val = (int(m[14]), int(m[15]), int(m[16]), int(m[17]))
    temp, mhz = int(m[18]), int(m[19])
    b_pct, s_pct, h_pct = float(m[20]), float(m[21]), float(m[22])

    par_trade = pnl / max(trades, 1)
    # LE POINT MORT SE CALCULE SUR LA FENETRE QU'IL JUGE, corrige le 2026-09-16.
    #
    # AvgW et AvgL de la ligne META sont ceux de l'ENTRAINEMENT. La preuve tient
    # en une identite : sur exec23 epoch 7, PF_train x pertes/gains vaut 1.708 et
    # AvgW/AvgL de META vaut 1.717, tandis que le meme rapport cote validation
    # vaut 1.542. Les comparer au winrate de VALIDATION melangeait donc deux
    # distributions, et le point mort sortait systematiquement trop bas.
    #
    # Cout reel de l'erreur : exec23 affichait +0.3 pt a l'epoch 7 avec un PF de
    # 0.91. Or le signe de l'ecart au point mort EST celui de PF - 1 : un ecart
    # positif sous PF 1 est arithmetiquement impossible. Trois epochs de suite
    # avaient ete lues comme "revenues au point mort" alors qu'elles perdaient.
    #
    # Tout se deduit de la seule ligne VAL, ou les trois chiffres sont coherents
    # entre eux :   AvgW/AvgL = PF x nL / nW   puis   point mort = 1/(1 + ce
    # rapport). L'assertion plus bas verifie l'identite a chaque epoch, pour que
    # le jour ou une source change, ce soit le test qui le dise.
    n_gagnants = round(trades * wr / 100.0)
    n_perdants = trades - n_gagnants
    rapport = pf * n_perdants / max(n_gagnants, 1)      # AvgW/AvgL, cote VAL
    equilibre = 100.0 / (1.0 + rapport) if rapport > 0 else float("nan")
    ecart = wr - equilibre
    # Gains et pertes de VALIDATION, retrouves a partir du PnL et du rapport :
    # l'erreur-type doit se calculer sur la meme fenetre que l'ecart.
    denom = n_gagnants * rapport - n_perdants
    perte_v = (pnl / denom) if abs(denom) > 1e-9 else avg_l
    gain_v = perte_v * rapport
    err = erreur_type(wr, abs(gain_v), abs(perte_v), trades)
    som = abs(gain_v) + abs(perte_v)
    err_pt = 100.0 * err / som if som > 0 else float("nan")
    duree = (t_col + t_ppo + t_cal + t_val) / 60.0
    # GELE SE LIT SUR LE GRADIENT, PAS SUR L'ENTROPIE. Le critere precedent —
    # entropie au-dessus de 1.09 et clipfrac sous 0.5 % — confondait deux
    # regimes tres differents :
    #
    #   l'actor est GELE            gradient ~1e-5, aucune mise a jour
    #   l'actor APPREND mais a plat gradient ~3e-1, la politique bouge peu
    #
    # Sur exec21 en M5, les epochs 6 a 11 etaient annoncees "l'actor est GELE"
    # alors que le warmup du critic ne dure que cinq epochs : elles
    # apprenaient, lentement. Confondre les deux fait passer pour du tirage au
    # sort ce qui est en realite un apprentissage trop lent — deux problemes
    # qui n'appellent pas du tout la meme correction.
    # LE PLAFOND SE CALCULE POUR CE COTE, pas une fois pour toutes.
    h_max = _plafond_entropie(cote)
    gele = ep <= WARMUP
    # Contre-verification : le gradient doit s'effondrer pendant le warmup et
    # passer ensuite. Un desaccord signale que la configuration lue ici n'est
    # plus celle du run — un run relance avec un autre warmup, par exemple.
    desaccord = (gele != (g_actor < GRADIENT_NUL))

    L = []
    L.append((("MODELE DEPLOYE (moyenne des poids)  " if moyenne else "")
              + f"EPOCH {ep:03d}   {par_trade:+.2f}$/trade   {trades} trades   "
                f"WR {wr:.1f}%   PF {pf:.2f}   {duree:.0f} min"))
    # CE QUI EST MESURE, ET CE QUI EST REPRIS. Une epoch sans validation
    # reaffiche les chiffres de la derniere mesure reelle. Les presenter
    # comme neufs laisse croire que le modele a ete evalue alors qu'il ne
    # l'a pas ete — et c'est la meme ligne qui portait le cumul faux.
    if reprise is not None:
        L.append(f"  PnL      {pnl:+10.2f}$   {C.JAUNE}REPRIS de l'epoch "
                 f"{reprise}, pas mesure ici{C.FIN}   cumul run "
                 f"{cumul:+11.2f}$   DD {dd:.1f}%")
    else:
        L.append(f"  PnL      {pnl:+10.2f}$   cumul run {cumul:+11.2f}$   "
                 f"DD {dd:.1f}%   Sortino {sortino:+.3f}")
    L.append(f"  point mort {equilibre:.1f}%  ->  ecart {ecart:+.1f} pt "
             f"(+/- {err_pt:.1f} au mieux)")

    # LE CLASSEMENT, et pourquoi il vaut mieux que la ligne au-dessus. Le PnL
    # porte sur ~150 trades : son erreur-type vaut 0.11 R, donc il ne separe
    # pas un modele qui ajoute 0.10 R d'un modele qui n'ajoute rien. Le rho
    # porte sur toutes les decisions de la fenetre, et son incertitude est
    # environ 0.024 — mesuree par bootstrap par blocs sur exec32.
    aux = None
    if isinstance(rho, tuple):
        rho, aux = rho
    if rho is not None:
        if rho > 0.05:
            coul, quoi = C.VERT, "classe nettement"
        elif rho > 0.02:
            coul, quoi = C.VERT, "classe"
        elif rho > -0.02:
            coul, quoi = C.GRIS, "n'ordonne rien"
        else:
            coul, quoi = C.ROUGE, "classe A L'ENVERS"
        sup = ""
        if aux is not None:
            # La tete auxiliaire est entrainee a PREDIRE le rendement, la
            # politique a AGIR. Si la seconde colonne monte pendant que la
            # premiere reste plate, c'est la tete qui doit trier.
            ca = C.VERT if aux > 0.02 else (C.ROUGE if aux < -0.02 else C.GRIS)
            sup = f"   tete auxiliaire {ca}{aux:+.4f}{C.FIN}"
        L.append(f"  classement rho {coul}{rho:+.4f}{C.FIN} "
                 f"(+/- 0.024 env.)  -> {quoi}{sup}")

    # CE QUE RAPPORTE LE SOMMET, et c'est sur lui que le checkpoint est
    # retenu depuis le 2024-09-19. Le rho dit si l'ORDRE est bon ; celui-ci
    # dit si les occasions effectivement prises PAIENT. Les lire separement
    # evite la confusion qui a coute le run precedent : un tri juste dont le
    # sommet ne rapporte rien n'est pas une strategie.
    if sommet is not None:
        g_top, g_hasard = sommet
        ecart = g_top - g_hasard
        cs = (C.VERT if ecart > 0.05 else
              (C.ROUGE if ecart < -0.05 else C.GRIS))
        L.append(f"  sommet retenu  {cs}{g_top:+.3f} R{C.FIN} par occasion  "
                 f"contre {g_hasard:+.3f} au hasard  "
                 f"-> {cs}{ecart:+.3f} R{C.FIN} de mieux"
                 f"   [c'est ce qui selectionne le checkpoint]")

    if reference is not None:
        n_ref, moy_ref = reference
        quoi_ref = "politique heritee" if herite else "politique gelee"
        L.append(f"  vs {quoi_ref} du meme run : {ecart - moy_ref:+.1f} pt"
                 f"   (reference {moy_ref:+.1f} pt sur {n_ref} epochs)")
    if precedent is not None:
        L.append(f"  vs epoch precedente : {par_trade - precedent:+.2f}$/trade")

    # LES DEUX SENS SEPAREMENT. Un modele qui ne gagne que d'un cote n'a pas
    # d'avantage, il a un biais directionnel — c'est ainsi que le premier
    # resultat de TabM s'est effondre, tout son gain venant d'un bloc haussier.
    def _wr(w, l):
        return 100.0 * w / max(w + l, 1)
    # DEUX LIGNES, ET LE TOTAL EN CLAIR. L'ancien format ecrivait
    # `LONG  16W/12  L  57.1%` : le cadrage a gauche du nombre de perdants
    # detachait son `L`, et la paire se lisait comme une fraction — "16
    # gagnants sur 12 trades". Elle n'a jamais voulu dire cela : 16 gagnants
    # ET 12 perdants, soit 28 trades. Le total est desormais affiche, et
    # verifie contre la ligne du haut.
    n_l, n_s = lw + ll, sw + sl_
    L.append(f"  sens    LONG  {n_l:>4d} trades  {lw:>4d} gagnants "
             f"{ll:>4d} perdants  {_wr(lw, ll):5.1f}%  {lpnl:+10.2f}$")
    L.append(f"          SHORT {n_s:>4d} trades  {sw:>4d} gagnants "
             f"{sl_:>4d} perdants  {_wr(sw, sl_):5.1f}%  {spnl:+10.2f}$")
    L.append(f"  actions B {b_pct:.1f}%  S {s_pct:.1f}%  H {h_pct:.1f}%"
             f"   |   selectivite  train {sel_tr:.1f}%  val {sel_val:.1f}%")
    # L'ETAT DE LA POLITIQUE, sur sa propre ligne. L'entropie et l'etendue
    # decident de ce que les autres chiffres VEULENT DIRE : une politique quasi
    # uniforme choisit ses trades presque au hasard, et son resultat mesure le
    # tirage. Les enterrer dans une note conditionnelle les rendait invisibles
    # exactement quand elles importaient le plus.
    d_ent = "" if ent_prec is None else f" ({H - ent_prec:+.3f})"
    L.append(f"  politique  H {H:.3f}/{h_max:.3f}{d_ent}"
             f" ({100*H/max(h_max,1e-9):.0f} % du plafond a {cote})"
             f"   etendue {et_val:.4f} "
             f"({et_val/TREMBLEMENT_SEUIL:.0f}x le tremblement)   "
             f"KL {kl:+.4f}   clipfrac {clipfrac:.1f}%   "
             f"g_actor {g_actor:.1e}")

    if tr is not None:
        t_wr, t_pf = float(tr[4]), float(tr[5])
        t_pnl, t_n = float(tr[2]), int(tr[3])
        L.append(f"  train   PnL {t_pnl:+9.2f}$  {t_n} trades  WR {t_wr:.1f}%  "
                 f"PF {t_pf:.2f}   ->  ecart train-val {t_wr - wr:+.1f} pt")

    # ---- lecture des diagnostics ----
    notes = []
    if desaccord:
        notes.append(
            f"DESACCORD SUR L'ETAT DE L'ACTOR : l'epoch {ep} est "
            f"{'dans' if gele else 'hors'} le warmup ({WARMUP} epochs) mais "
            f"son gradient vaut {g_actor:.1e}. La configuration lue par la "
            f"veille n'est peut-etre pas celle du run.")
    if gele and herite:
        # UN FOLD CHAINE NE PART PAS DU HASARD. Ses epochs a actor gele
        # mesurent la politique HERITEE du fold precedent, appliquee a une
        # fenetre qu'elle n'a pas encore vue. C'est une reference utile — le
        # niveau avant tout nouvel apprentissage — mais l'appeler "hasard"
        # ferait lire un transfert reussi comme un coup de chance.
        notes.append(f"ACTOR GELE, POIDS HERITES : warmup du critic, gradient "
                     f"{g_actor:.1e}. Cette epoch mesure la politique du fold "
                     f"PRECEDENT sur cette fenetre — pas le hasard. C'est le "
                     f"niveau de depart que la suite doit battre.")
    elif gele:
        notes.append(f"ACTOR GELE : warmup du critic, gradient {g_actor:.1e}. "
                     f"Cette epoch ne mesure aucun apprentissage — elle sert de "
                     f"reference au hasard pour les suivantes.")
    elif H > 0.99 * h_max:
        notes.append(f"APPREND MAIS RESTE PLAT : le gradient passe "
                     f"({g_actor:.1e}) mais l'entropie tient a {H:.3f} sur "
                     f"{h_max:.3f}. La politique se differencie trop lentement pour "
                     f"que le filtre ait un sens — c'est un probleme de pas "
                     f"d'apprentissage ou d'echelle, pas de tirage.")
    # Mesure du 2026-09-15 sur les 5 epochs gelees d'exec12, qui partagent le
    # meme reseau : le seuil calibre s'est deplace de 0.0010 alors que
    # l'etendue totale des convictions valait 0.0002. Les 5 % retenus
    # changeaient donc presque entierement d'une epoch a l'autre — 136 trades
    # d'ecart et jusqu'a 10 points de resultat, a reseau identique. C'est la
    # fabrique des faux champions.
    if et_val < 10 * TREMBLEMENT_SEUIL:
        notes.append(f"SELECTION TIREE AU SORT : etendue val {et_val:.4f} "
                     f"contre un tremblement de seuil de ~{TREMBLEMENT_SEUIL:.4f}. "
                     f"Les {sel_val:.0f} % retenus changent presque entierement "
                     f"d'une epoch a l'autre — ce chiffre mesure le tirage.")
    else:
        notes.append(f"etendue val {et_val:.4f}, soit "
                     f"{et_val/TREMBLEMENT_SEUIL:.0f}x le tremblement du seuil : "
                     f"la selection est portee par le modele.")
    if n_l + n_s != trades:
        notes.append(f"LES DEUX SENS NE FONT PAS LE TOTAL : {n_l} achats + "
                     f"{n_s} ventes = {n_l + n_s}, or la ligne VAL en annonce "
                     f"{trades}. Un trade manque a la ventilation.")
    # LE COTE INTERDIT DOIT ETRE VIDE. or_exec02 a tourne 273 epochs en
    # long-only avec 31 a 68 % de ventes en validation, et rien ne l'a dit :
    # la veille affichait les deux sens sans jamais les confronter au cote
    # declare dans le tag du journal.
    if cote == "long" and n_s:
        notes.append(f"VENTES DANS UN RUN LONG-ONLY : {n_s} ventes pour "
                     f"{spnl:+.2f}$ alors que le run est declare LONG. La "
                     f"regle de decision ne respecte pas le masque d'actions "
                     f"— le resultat net ne mesure pas la strategie entrainee.")
    if cote == "short" and n_l:
        notes.append(f"ACHATS DANS UN RUN SHORT-ONLY : {n_l} achats pour "
                     f"{lpnl:+.2f}$ alors que le run est declare SHORT.")
    if (sw + sl_ > 0) and ((lpnl > 0) != (spnl > 0)):
        gagnant = "LONG" if lpnl > 0 else "SHORT"
        notes.append(f"GAIN UNILATERAL : seul le {gagnant} rapporte. Un modele "
                     f"qui ne gagne que d'un cote a un biais directionnel, pas "
                     f"un avantage — verifier que l'autre sens suit avant de "
                     f"conclure quoi que ce soit.")
    if not math.isnan(err) and abs(par_trade) < err:
        notes.append(f"gain par trade ({par_trade:+.2f}$) sous l'erreur-type de "
                     f"cette epoch ({err:.2f}$) — non separable de zero.")
    if mhz < 500:
        notes.append(f"GPU BRIDE : {mhz} MHz a {temp} C. Les durees ne sont pas "
                     f"comparables entre epochs si la temperature derive.")
    if clipfrac > 30:
        notes.append(f"clipfrac {clipfrac:.1f}% — la mise a jour tape souvent la "
                     f"borne de clipping, les pas sont tronques.")

    for n in notes:
        L.append(f"  . {n}")
    L.append(f"  temps : collecte {t_col}s  PPO {t_ppo}s  calib {t_cal}s  "
             f"validation {t_val}s")

    # La couleur juge par rapport a la REFERENCE DU RUN, pas a un seuil fige.
    if reference is None:
        couleur = C.GRIS
    else:
        gain = ecart - reference[1]
        couleur = (C.VERT if gain > err_pt
                   else (C.JAUNE if gain > -err_pt else C.ROUGE))
    console = [f"{couleur}{C.GRAS}{L[0]}{C.FIN}"] + \
              [f"{couleur}{x}{C.FIN}" for x in L[1:3]] + \
              [f"{C.GRIS}{x}{C.FIN}" for x in L[3:]]
    return console, L, par_trade, ecart, gele


RE_COTE = re.compile(r"\[(BOTH|LONG|SHORT)_(wf\d+)\]")


class Veilleur:
    """L'analyse des epochs, alimentee par le TEXTE du journal.

    POURQUOI UN OBJET. Cette analyse n'existait que dans une seconde fenetre :
    sans `veille_epochs.py` lance a cote, le terminal d'entrainement ne
    montrait que ses propres colonnes — ni le point mort, ni l'ecart a la
    politique gelee du run, ni aucun des diagnostics. L'etat necessaire tient
    en quelques dictionnaires par fold ; les sortir d'une boucle `main` suffit
    a ce que `training.py` fasse defiler le meme texte au moment ou il l'ecrit.

    IL N'Y A PAS DE SECONDE MISE EN FORME, et c'est la seule raison d'etre de
    ce decoupage. Recopier la presentation cote entrainement aurait cree deux
    descriptions du meme resultat, qui doivent s'accorder par convention :
    la faute que ce depot passe son temps a payer.

    `avale` rend une liste de (lignes colorees, lignes brutes). Les lignes
    brutes valent None pour les annonces qui ne sont pas des epochs.
    """

    def __init__(self):
        self.vus = set()
        self.vals, self.metas, self.trains, self.rhos = {}, {}, {}, {}
        self.cotes = {}         # par fold : long / short / both, lu du tag
        self.herites = set()    # folds dont les poids viennent du precedent
        self.sommets = {}       # par (fold, epoch) : (gain du sommet, au hasard)
        self.reprises = {}      # par (fold, epoch) : epoch dont le PnL est repris
        self.precedent = {}     # par fold : gain par trade de l'epoch d'avant
        self.geles = {}         # par fold : ecarts des epochs a actor gele
        self.cumul = {}         # par fold : PnL de validation cumule
        self.entropie = {}      # par fold : entropie de l'epoch precedente
        self.attend_moyenne = set()
        self.fold_courant = None

    def oublie_tout(self):
        """Journal tronque ou run relance : on repart de zero."""
        self.vus.clear()
        for d in (self.vals, self.metas, self.trains, self.rhos, self.cotes,
                  self.precedent, self.geles, self.cumul, self.entropie,
                  self.sommets, self.reprises):
            d.clear()
        self.herites.clear()
        self.attend_moyenne.clear()
        self.fold_courant = None

    def avale(self, texte):
        blocs = []
        for ligne in ANSI.sub("", texte).split("\n"):
            mc = RE_COTE.search(ligne)
            if mc:
                self.cotes[mc.group(2)] = mc.group(1).lower()
            if "POIDS HERITES" in ligne:
                if mc:
                    self.herites.add(mc.group(2))
                blocs.append(([f"{C.CYAN}{C.GRAS}  {ligne.strip()}{C.FIN}"],
                              None))
            if "[MEMOIRE]" in ligne:
                blocs.append(([f"{C.CYAN}  {ligne.strip()}{C.FIN}"], None))
            if "MOYENNE DES POIDS sur les" in ligne:
                # Annonce emise AVANT l'epoch concernee : les poids qui
                # suivent sont la moyenne des dernieres epochs, donc le
                # modele qui part au test.
                blocs.append((
                    [f"{C.CYAN}{C.GRAS}  {ligne.strip()}{C.FIN}"], None))
                self.attend_moyenne.add(
                    ligne.split("]")[0].strip("[").split("_")[-1])
            if "[TEST]" in ligne:
                blocs.append(([f"{C.CYAN}{C.GRAS}  {ligne.strip()}{C.FIN}"],
                              None))
            if "NEW BEST" in ligne:
                # Le checkpoint retenu, et sur quoi il l'a ete. Depuis que la
                # selection se fait sur le CLASSEMENT et non sur le Sortino,
                # c'est la ligne qui dit quel modele partira au fold suivant.
                blocs.append(([f"{C.VERT}{C.GRAS}  {ligne.strip()}{C.FIN}"],
                              None))
            mv, mm = RE_VAL.search(ligne), RE_META.search(ligne)
            mt = RE_TRAIN.search(ligne)
            if mv:
                self.vals[(mv.group(1), int(mv.group(2)))] = mv.groups()
            if mm:
                self.metas[(mm.group(1), int(mm.group(2)))] = mm.groups()
                # Lus par des motifs separes : les inclure dans RE_META
                # decalerait les indices que `analyse` lit par position.
                mr = RE_RHO.search(ligne)
                ma = RE_RHO_AUX.search(ligne)
                self.rhos[(mm.group(1), int(mm.group(2)))] = (
                    float(mr.group(1)) if mr else None,
                    float(ma.group(1)) if ma else None)
                ms = RE_SOMMET.search(ligne)
                if ms:
                    self.sommets[(mm.group(1), int(mm.group(2)))] = (
                        float(ms.group(1)), float(ms.group(2)))
                mrp = RE_REPRISE.search(ligne)
                if mrp:
                    self.reprises[(mm.group(1), int(mm.group(2)))] = int(
                        mrp.group(1))
            if mt:
                self.trains[(mt.group(1), int(mt.group(2)))] = mt.groups()

        for cle in sorted(set(self.vals) & set(self.metas) - self.vus):
            self.vus.add(cle)
            fold, ep = cle
            entete = []
            if fold != self.fold_courant:
                entete = [f"{C.CYAN}{C.GRAS}  ===  {fold.upper()}  ==={C.FIN}",
                          ""]
                self.fold_courant = fold
            ref = None
            if len(self.geles.get(fold, [])) >= 2:
                g = self.geles[fold]
                ref = (len(g), sum(g) / len(g))
            # ON N'ADDITIONNE QUE LES MESURES REELLES. Le cumul comptait
            # chaque epoch, reprises comprises : a une validation sur trois,
            # il annoncait donc environ TROIS FOIS le PnL reellement
            # realise. Personne ne l'avait vu parce que le chiffre est
            # plausible — il monte, il a le bon signe, il a juste le mauvais
            # facteur.
            _reprise = self.reprises.get(cle)
            if _reprise is None:
                self.cumul[fold] = (self.cumul.get(fold, 0.0)
                                    + float(self.vals[cle][2]))
            est_moyenne = fold in self.attend_moyenne
            self.attend_moyenne.discard(fold)
            console, brut, par_trade, ecart, gele = analyse(
                self.vals[cle], self.metas[cle], self.trains.get(cle),
                self.precedent.get(fold), ref, self.cumul[fold], est_moyenne,
                self.entropie.get(fold), self.rhos.get(cle),
                self.cotes.get(fold, "both"), fold in self.herites,
                self.sommets.get(cle), self.reprises.get(cle))
            self.entropie[fold] = float(self.metas[cle][5])
            self.precedent[fold] = par_trade
            if gele:
                self.geles.setdefault(fold, []).append(ecart)
            horo = dt.datetime.now().strftime("%H:%M:%S")
            blocs.append((entete + [f"{C.GRIS}[{horo}]{C.FIN}"] + console,
                          brut))
        return blocs


def ecrit_rapport(brut, fold, chemin=RAPPORT):
    horo = dt.datetime.now().strftime("%H:%M:%S")
    with open(chemin, "a", encoding="utf-8") as r:
        r.write(f"\n## {horo} — {fold} — {brut[0]}\n\n")
        for l in brut[1:]:
            r.write(l.strip() + "\n")


def main() -> int:
    print(f"\n{C.CYAN}{C.GRAS}  KAIROS — veille d'analyse par epoch{C.FIN}")
    print(f"{C.GRIS}  {'-' * 66}")
    print(f"  journal : {JOURNAL}")
    print(f"  rapport : {RAPPORT}")
    print(f"  reference : les epochs a actor gele DU RUN EN COURS")
    print(f"  LECTURE SEULE — fermer cette fenetre n'arrete pas "
          f"l'entrainement")
    print(f"  {'-' * 66}{C.FIN}\n")

    v = Veilleur()
    position = 0
    utf16 = False

    while True:
        try:
            taille = os.path.getsize(JOURNAL)
            if taille < position:
                print(f"\n{C.JAUNE}  journal tronque ou run relance — "
                      f"relecture depuis le debut{C.FIN}\n")
                position = 0
                utf16 = False
                v.oublie_tout()
            # Lecture en BYTES puis decodage selon le BOM : le journal peut
            # etre relance sous PowerShell (> fichier.log), qui ecrit en
            # UTF-16LE avec BOM au lieu d'UTF-8. Lu en utf-8, chaque caractere
            # est suivi d'un \x00, aucun motif ne matche et la veille reste
            # muette sans rien dire. Les offsets restent des octets, donc la
            # logique de reprise de position est inchangee.
            with open(JOURNAL, "rb") as f:
                f.seek(position)
                nouveau = f.read()
                position = f.tell()
            if (nouveau.startswith(b"\xff\xfe")
                    or nouveau.startswith(b"\xfe\xff")):
                nouveau = nouveau.decode("utf-16")
                utf16 = True
            elif utf16:
                nouveau = nouveau.decode("utf-16")
            else:
                nouveau = nouveau.decode("utf-8", errors="replace")
        except FileNotFoundError:
            time.sleep(PAS_SONDAGE)
            continue

        for console, brut in v.avale(nouveau):
            for l in console:
                print(l)
            print()
            if brut is not None:
                ecrit_rapport(brut, v.fold_courant or "?")

        time.sleep(PAS_SONDAGE)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nveille arretee — l'entrainement, lui, continue.")
