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
# `nan` EST UNE VALEUR, PAS UNE PANNE — et l'oublier a rendu la veille
# muette une neuvieme fois, le 2026-09-21.
#
# CE QUI S'EST PASSE. `ppo_actif` est passe a False : plus aucune mise a
# jour, donc `epoch_kl` et `epoch_grad_norm` restent vides, donc
# `np.mean([])` rend nan et le journal ecrit `KL   +nan`. Le motif exigeait
# des chiffres, il n'a plus mordu, et la veille a cesse d'afficher les
# epochs — avec, cette fois, le message qui nomme la ligne fautive, sans
# lequel on aurait encore cherche du cote de l'entrainement.
#
# LA LECON, ET ELLE VAUT POUR TOUT CHAMP FUTUR : un indicateur qu'on cesse
# de mesurer ne disparait pas du journal, il s'y ecrit `nan`. Le motif doit
# donc accepter `nan` et `inf` PARTOUT ou il accepte un nombre, sinon
# chaque grandeur qu'on eteint casse la lecture de toutes les autres.
#
# `float("+nan")` rend bien nan en Python : les consommateurs en aval n'ont
# rien a changer, ils traitaient deja des nan venus d'epochs sans grille.
NB = r"[+-]?(?:nan|inf|[\d.]+)"

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
    r"AvgW\s+(" + NB + r")\$\s+AvgL\s+(" + NB + r")\$.*?H (" + NB + r").*?"
    r"sel\[train\s+(" + NB + r")% val\s+(" + NB + r")%\].*?"
    r"etendue\[tr (" + NB + r") val (" + NB + r")\].*?"
    r"KL\s+(" + NB + r").*?gnorm\s+(" + NB + r")\s+"
    r"g\[actor ([\d.e+-]+|nan).*?clipfrac (" + NB + r")%\s+"
    # `.*?` ENTRE CHAQUE PHASE, et ce n'est pas de la coquetterie. Le chrono
    # s'ecrit en parcourant un dictionnaire : toute phase ajoutee a
    # l'entrainement s'insere dans la ligne, et un motif rigide cesse alors de
    # mordre — la veille se tait, sans rien signaler. C'est arrive quand la
    # phase `rang` est apparue entre `collecte` et `maj PPO`, et ca se
    # reproduira a la prochaine phase. Le motif ne nomme donc plus que les
    # phases qu'il LIT, et tolere tout ce qui se glisse entre elles.
    r"temps\[collecte (\d+)s.*?maj PPO (\d+)s.*?calibration (\d+)s.*?"
    # LE BLOC GPU EST FACULTATIF. `nvidia-smi` echoue parfois — carte
    # occupee, delai de 3 s depasse — et `training` ecrit alors `gpu[?]`. Le
    # motif exigeait `gpu[<temp>C <freq>/`, donc TOUTE la ligne META cessait
    # de se lire et plus une seule epoch ne s'affichait. C'est arrive le
    # 2026-09-20 sur exec32, et le symptome est toujours le meme : la veille
    # se tait, sans rien signaler, et on cherche du cote de l'entrainement.
    # Les deux groupes restent captures pour ne pas decaler les indices.
    r"validation (\d+)s\](?:.*?gpu\[(\d+)C (\d+)/)?"
    # LE BLOC ENV ETAIT LE DERNIER MOTIF RIGIDE DU FICHIER, et il a casse
    # exactement comme les quatre precedents. Le 2026-09-21, CLOTURER est
    # devenue une action a part entiere et l'entrainement ecrit desormais
    # `ENV [B .. S .. H .. C ..]`. Le motif exigeait `]` juste apres H : il
    # a cesse de mordre, et toute la ligne META avec lui.
    #
    # LA DIFFERENCE AVEC LES FOIS PRECEDENTES, c'est qu'on l'a SU. Le garde
    # « ligne META illisible, motif a reajuster » a parle des la premiere
    # epoch, au lieu des heures de silence qu'ont coutees `rho`, `table`,
    # la phase `rang` et le bloc `gpu[?]`.
    #
    # `C` EST OPTIONNEL ET EN DERNIER : optionnel pour continuer de lire les
    # journaux anterieurs, en dernier pour ne decaler aucun des indices que
    # `analyse` lit par position. Et `[^\]]*` avale tout champ futur avant
    # le crochet, pour que le SIXIEME ajout ne casse plus rien.
    r".*?ENV \[B\s+(" + NB + r")% S\s+(" + NB + r")% H\s+(" + NB + r")%"
    r"(?:\s+C\s+(" + NB + r")%)?[^\]]*\]")

# Le critere de SELECTION depuis le 2026-09-19 : ce que rapportent, en
# unites de risque, les occasions que le checkpoint mettrait en position, et
# ce que rapporte une occasion au hasard. Lu par un motif separe pour la
# meme raison que rho — ne pas decaler les indices de `analyse`.
RE_SOMMET = re.compile(r"sommet\s+(" + NB + r")R/(" + NB + r")R")

# LA TENUE PAR ISSUE : mediane/moyenne des gagnants, puis des perdants.
#
# Elle repond a une question posee quatre fois et resolue quatre fois par
# simulation hors ligne, alors que la boucle de validation avait le
# chiffre. Ce qu'elle montre au plafond de 480 : les gagnants tiennent
# TOUS jusqu'au bout, mediane ET moyenne egales au plafond. La regle de
# sortie n'a aucune prise de profit — elle ne sait que couper les pertes.
# LE SEUIL QUE LE SCORE NET DOIT BATTRE POUR QUE LE MODELE SOIT SAUVE.
# Motif SEPARE, comme ce fichier l'exige pour tout champ ajoute : c'est la
# seule facon de ne pas decaler les indices que `analyse` lit par position.
#
# UN JOURNAL ANTERIEUR AU 2026-09-25 NE L'ECRIT PAS. La veille le reconstruit
# alors elle-meme : zero tant que rien n'est retenu dans le fold, puis le
# score net de la derniere ligne NEW BEST. C'est exactement la regle de
# `retient_checkpoint`, et c'est exact tant que le journal est lu depuis
# son debut — ce que la veille fait toujours.
RE_BATTRE = re.compile(r"a_battre\s+(" + NB + r")R")

RE_TENUE = re.compile(
    r"tenue\[G (" + NB + r")/(" + NB + r") P (" + NB + r")/(" + NB + r") "
    r"x(" + NB + r") max (" + NB + r")\]")

# LE CRITERE DE SELECTION DEPUIS LE 2026-09-20, et la raison du changement.
#
# `sommet` note la QUALITE DU TRI, occasion par occasion. Il ignore l'ORDRE
# dans lequel elles arrivent — donc le creux — et la TAILLE misee sur chacune
# — donc l'abstention. Or la tete de budget est la seule que PPO entraine
# encore, et elle a un palier 0 % : un modele qui apprendrait a ne rien miser
# aux mauvais moments rendait exactement le meme `sommet` qu'un modele misant
# pareil partout.
#
# `net` = gain - baisse, PAR OCCASION RETENUE, en R d'une POSITION MINIMALE :
# ce qu'une occasion rapporte misee comme le modele la mise, moins la taille
# typique de ses pertes. `bud` est le nombre moyen de positions posees, `abst`
# la part des occasions ou le modele ne mise rien — les deux disent POURQUOI
# `net` a bouge.
#
# L'UNITE EST CELLE DE L'ACTION DU MODELE et ne depend ni du capital ni de la
# volatilite. Ce n'est pas un rendement de fenetre : il y a ~160 occasions, les
# lire comme un total serait une erreur d'un facteur 160 — et une vraie courbe
# d'equite serait fausse, les occasions se chevauchant cinq fois.
#
# MOTIF SEPARE, comme rho et sommet : l'inclure dans RE_META decalerait les
# indices que `analyse` lit par position, en silence.
RE_NET = re.compile(
    r"net\s+(" + NB + r")R\s+\(gain\s+(" + NB + r")R baisse\s+([\d.]+)R\)")

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

# ============================================================
# LA RETENUE DU MEILLEUR MODELE : ce qui a ete sauve, et sur quoi.
#
# POURQUOI DES MOTIFS PLUTOT QU'UN ECHO. La veille recopiait la ligne
# `NEW BEST` telle quelle, en vert. On y lisait un `sommet` — et depuis le
# 2026-09-20 ce n'est plus lui qui decide. On croyait donc savoir sur quoi le
# modele avait ete retenu, et c'etait faux. En lisant les champs un par un, la
# veille dit le CRITERE, ce qu'il a battu, et le fichier ecrit.
#
# CHAQUE MORCEAU EST OPTIONNEL A LA LECTURE : si un seul ne mord pas, on
# reaffiche la ligne brute au lieu de la perdre. Une ligne de retenue avalee
# en silence serait la pire des pannes de ce fichier — c'est celle qui dit
# quel modele sera deploye.
RE_BEST = re.compile(
    r"NEW BEST\s+retenu sur le SCORE NET\s+(" + NB + r") R par occasion\s+"
    r"\(gain\s+(" + NB + r") R - baisse\s+([\d.]+) R\)\s+"
    r"bat (premier retenu du fold|[+-][\d.]+ R)\s+"
    r"\[portillons : sommet (" + NB + r")R > hasard (" + NB + r")R, "
    r"(\d+) trades, compte intact\]\s+-> (\S+)")

# Le refus ordinaire — quatre-vingts epochs sur quatre-vingt-dix. Il passait
# en silence : on regardait defiler un run entier en croyant qu'un meilleur
# modele etait garde alors que rien ne l'etait.
RE_GARDE = re.compile(r"garde\s+le modele en place reste le meilleur — (.+)$")

# Le refus GRAVE : le modele etait meilleur au score, et c'est le
# portefeuille qui l'a disqualifie.
RE_REFUS = re.compile(r"REFUSE\s+score net\s+(" + NB + r") R par occasion, "
                      r"meilleur que\s+(" + NB + r") R, mais (.+?) —")

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
    # PPO PEUT ETRE COUPE — et alors un gradient d'acteur nul est NORMAL.
    # Sans cette lecture, la veille signale un « desaccord sur l'etat de la
    # politique » a chaque epoch du run : un avertissement rouge, repete
    # quatre-vingt-dix fois, pour un comportement voulu. Un indicateur qui
    # crie au loup a chaque epoch cesse d'etre lu — et le jour ou il aurait
    # raison, personne ne le verra.
    PPO_ACTIF = bool(getattr(_Cfg(), "ppo_actif", True))
except Exception:
    WARMUP = 5
    PPO_ACTIF = True


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
            sommet=None, reprise=None, net=None, tenue=None,
            a_battre=None, historique=None):
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
    # ABSENTS QUAND `nvidia-smi` ECHOUE. Une carte muette n'est pas une
    # raison de ne rien afficher de l'epoch.
    temp = int(m[18]) if m[18] is not None else None
    mhz = int(m[19]) if m[19] is not None else None
    b_pct, s_pct, h_pct = float(m[20]), float(m[21]), float(m[22])
    # LA PART DE CLOTURES, absente des journaux d'avant le 2026-09-21.
    c_pct = float(m[23]) if m[23] is not None else None

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

    # `rho` A DISPARU DU BANDEAU LE 2026-09-21, AVEC LE BUDGET.
    #
    # Il correlait la PART DEPLOYEE au rendement : « le modele mise-t-il
    # gros quand l'occasion paie ? ». Sans paliers il n'y a plus de part —
    # toute occasion retenue est prise a la meme taille, celle du lot
    # minimum, mesuree a 1.0000 unite (p5 0.9999, p95 1.0001) sur 7 363
    # trades.
    #
    # LE BLOC ETAIT CONDITIONNE A `rho`, donc `rhoAux` serait parti avec
    # lui — et `rhoAux` est celui qui compte : il correle le SCORE de la
    # tete de rang au rendement, c'est-a-dire la seule question qui reste,
    # « classe-t-elle les occasions ? ». C'est sur son sommet que le
    # checkpoint est retenu. On affiche donc celui qui existe.
    _rho_lu = aux if aux is not None else rho
    if _rho_lu is not None:
        if _rho_lu > 0.05:
            coul, quoi = C.VERT, "classe nettement"
        elif _rho_lu > 0.02:
            coul, quoi = C.VERT, "classe"
        elif _rho_lu > -0.02:
            coul, quoi = C.GRIS, "n'ordonne rien"
        else:
            coul, quoi = C.ROUGE, "classe A L'ENVERS"
        # L'INCERTITUDE EST CELLE DU RHO, pas une convention : environ
        # 0.024, mesuree par bootstrap par blocs sur exec32. Sans elle un
        # +0.019 se lirait comme un resultat.
        L.append(f"  classement de la tete de rang {coul}{_rho_lu:+.4f}"
                 f"{C.FIN} (+/- 0.024 env.)  -> {quoi}")

    # CE QUE RAPPORTE LE SOMMET. Le rho dit si l'ORDRE est bon ; celui-ci
    # dit si les occasions effectivement prises PAIENT. Les lire separement
    # evite la confusion qui a coute un run : un tri juste dont le sommet ne
    # rapporte rien n'est pas une strategie.
    #
    # IL NE CHOISIT PLUS LE CHECKPOINT depuis le 2026-09-20 : il note la
    # qualite du tri et rien d'autre, ni l'ordre des occasions donc pas le
    # creux, ni la taille misee donc pas l'abstention. C'est `net`, ci-
    # dessous, qui decide. `sommet` garde le portillon du hasard.
    # LA TENUE PAR ISSUE — ce que AvgW et AvgL ne peuvent pas montrer.
    #
    # Un rapport de gain de 26 pour 1 se lit deja dans AvgW/AvgL. Ce qu'il
    # ne dit pas, c'est SI la tete a decide de sortir avec un gain ou si le
    # trade a simplement survecu au chronometre. Quand la mediane ET la
    # moyenne des gagnants valent le plafond, la reponse est la seconde :
    # la regle de sortie n'a pas de prise de profit, elle ne sait que
    # couper les pertes.
    if tenue is not None:
        _mg, _yg, _mp, _yp, _rap, _mx = tenue
        _ct = (C.VERT if _rap >= 3.0 else
               (C.ROUGE if _rap < 1.0 else C.GRIS))
        # SANS PLAFOND, LE SIGNAL A GUETTER A CHANGE. Ce n'est plus
        # « tous les gagnants sortent au meme instant » mais « une
        # position ne sort jamais ».
        _fin = ("   les gagnants sortent TOUS au meme instant — regle figee"
                if (_mg > 0 and abs(_mg - _yg) < 1.0) else "")
        L.append(f"  tenue          gagnant {_mg:.0f} min (mediane) / "
                 f"{_yg:.0f} (moyenne)   perdant {_mp:.0f} / {_yp:.0f}   "
                 f"{_ct}x{_rap:.1f}{C.FIN}   max {_mx:.0f} min{_fin}")

    if sommet is not None:
        g_top, g_hasard = sommet
        ecart = g_top - g_hasard
        cs = (C.VERT if ecart > 0.05 else
              (C.ROUGE if ecart < -0.05 else C.GRIS))
        L.append(f"  sommet du tri  {cs}{g_top:+.3f} R{C.FIN} par occasion  "
                 f"contre {g_hasard:+.3f} au hasard  "
                 f"-> {cs}{ecart:+.3f} R{C.FIN} de mieux"
                 # CE LIBELLE DISAIT « c'est ce qui selectionne le
                 # checkpoint ». Le commentaire trente lignes plus haut dit
                 # le contraire depuis le 2026-09-20 — « il ne choisit plus
                 # le checkpoint, c'est `net` qui decide » — et le libelle
                 # n'avait jamais suivi. On regardait le mauvais chiffre.
                 f"   [portillon : necessaire, ne suffit pas]")

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
    # LA PART DE CLOTURES EST LE TEMOIN DE LA SORTIE.
    #
    # A zero, la tete de cloture ne ferme jamais : sans stop ni objectif ni
    # trailing, la position vit alors jusqu'au plafond de detention, et un
    # episode de 5 760 barres ne fait plus qu'un seul trade. C'est
    # exactement ce qui s'est passe avant que la tete soit branchee, et ce
    # chiffre l'aurait montre d'un coup d'oeil.
    _c = "" if c_pct is None else f"  C {c_pct:.1f}%"
    L.append(f"  actions B {b_pct:.1f}%  S {s_pct:.1f}%  H {h_pct:.1f}%{_c}"
             f"   |   selectivite  train {sel_tr:.1f}%  val {sel_val:.1f}%")
    if c_pct is not None and c_pct <= 0.0:
        L.append("          C a 0 % : la tete de cloture ne ferme JAMAIS. "
                 "Sans stop ni objectif, la position court jusqu'au plafond "
                 "de detention et l'episode ne fait qu'un trade.")
    # L'ETAT DE LA POLITIQUE, sur sa propre ligne. L'entropie et l'etendue
    # decident de ce que les autres chiffres VEULENT DIRE : une politique quasi
    # uniforme choisit ses trades presque au hasard, et son resultat mesure le
    # tirage. Les enterrer dans une note conditionnelle les rendait invisibles
    # exactement quand elles importaient le plus.
    # `H` EST L'ENTROPIE DE LA REGLE D'ENTREE, plus celle d'un acteur.
    # La tete de direction a ete supprimee : `H` porte desormais la
    # repartition des decisions reellement prises — acheter, vendre,
    # attendre. A 0 la regle s'est figee, au plafond elle est a pile ou face.
    # `g_actor` est le gradient de la tete de BUDGET, seule politique que PPO
    # entraine encore.
    d_ent = "" if ent_prec is None else f" ({H - ent_prec:+.3f})"
    L.append(f"  entrees  H {H:.3f}/{h_max:.3f}{d_ent}"
             f" ({100*H/max(h_max,1e-9):.0f} % du plafond a {cote})"
             f"   etendue {et_val:.4f} "
             f"({et_val/TREMBLEMENT_SEUIL:.0f}x le tremblement)   "
             f"KL {kl:+.4f}   clipfrac {clipfrac:.1f}%   "
             f"g_budget {g_actor:.1e}")

    # LE CRITERE QUI CHOISIT LE CHECKPOINT, et ce qui le compose.
    #
    # `net` = gain - baisse, PAR OCCASION retenue, en R d'une position
    # minimale.
    # Un modele qui s'abstient au bon moment est paye DEUX FOIS : il retire un
    # rendement negatif, ce qui monte la moyenne ET vide la queue gauche.
    #
    # LA `baisse` EST UN DEMI-ECART-TYPE SOUS ZERO, pas un creux de courbe.
    # Une courbe d'equite sur cette grille serait fausse : 160 occasions
    # retenues sur une fenetre qui n'en tient que ~34 sans chevauchement, donc
    # le meme capital compte cinq fois — elle affichait un creux de 164 %. Une
    # moyenne et une demi-variance, elles, supportent le chevauchement : il les
    # rend moins PRECISES, il ne les fausse pas.
    #
    # `bud` et `abst` disent POURQUOI il a bouge. `net` qui monte avec `abst`
    # qui monte : c'est l'abstention qui paie. `net` qui monte avec `bud` qui
    # monte : c'est le levier, et le creux dira bientot ce qu'il coute. Sans
    # ces deux nombres, les deux histoires sont indiscernables.
    if net is not None:
        v_net, v_gain, v_creux = net
        cn = C.VERT if v_net > 0 else (C.ROUGE if v_net < 0 else C.GRIS)
        # Le creux se lit RELATIVEMENT au gain : 8 % de creux pour 40 % de
        # gain n'est pas 8 % de creux pour 3 % de gain.
        _rap = (v_gain / v_creux) if v_creux > 1e-9 else float("inf")
        _lr = ("aucune perte" if not math.isfinite(_rap)
               else f"{_rap:.2f}x la baisse")
        L.append(f"  critere net  {cn}{v_net:+6.3f} R{C.FIN} par occasion  "
                 f"= gain {v_gain:+.3f} R - baisse {v_creux:.3f} R  ({_lr})"
                 f"   [C'EST LUI QUI DECIDE LA SAUVEGARDE]")
        # CE QU'IL DOIT BATTRE, et ce qu'il lui manque. Sans le seuil, on
        # voyait le score bouger sans savoir s'il s'approchait de quoi que ce
        # soit.
        if a_battre is not None:
            _manque = a_battre - v_net
            if _manque < 0:
                _etat = f"{C.VERT}{C.GRAS}FRANCHI de {-_manque:.3f} R{C.FIN}"
            else:
                _etat = f"{C.ROUGE}il manque {_manque:.3f} R{C.FIN}"
            _quoi = ("le record du fold" if a_battre > 0
                     else "rien de retenu dans ce fold : il suffit d'etre "
                          "positif")
            L.append(f"  sauvegarde   si net > {a_battre:+.3f} R "
                     f"({_quoi})  ->  {_etat}")
        # SON EVOLUTION : les dernieres mesures du fold, puis celle-ci.
        if historique:
            _prec = "  ".join(f"{x:+.3f}" for x in historique[-6:])
            _plus = "... " if len(historique) > 6 else ""
            _d = v_net - historique[-1]
            _cd = C.VERT if _d > 0 else (C.ROUGE if _d < 0 else C.GRIS)
            L.append(f"  evolution    {_plus}{_prec}  ->  {cn}{v_net:+.3f}"
                     f"{C.FIN}   ({_cd}{_d:+.3f}{C.FIN} sur l'epoch "
                     f"precedente, meilleur du fold "
                     f"{max(historique + [v_net]):+.3f})")
        # LA LIGNE `dimension` A ETE RETIREE LE 2026-09-21, avec le budget.
        # Elle disait la mise moyenne et la part d'abstention ; les deux
        # valaient 60 % et 0 % a chaque epoch — deux constantes affichees
        # comme des mesures. La selectivite, ligne `actions`, porte
        # desormais seule la question « combien d'occasions sont prises ».

    if tr is not None:
        t_wr, t_pf = float(tr[4]), float(tr[5])
        t_pnl, t_n = float(tr[2]), int(tr[3])
        L.append(f"  train   PnL {t_pnl:+9.2f}$  {t_n} trades  WR {t_wr:.1f}%  "
                 f"PF {t_pf:.2f}   ->  ecart train-val {t_wr - wr:+.1f} pt")

    # ---- lecture des diagnostics ----
    notes = []
    if not PPO_ACTIF:
        # RIEN A SIGNALER : `ppo_actif = False`, donc aucune mise a jour, donc
        # un gradient nul est la seule valeur possible. On le rappelle une
        # fois par epoch en gris plutot que de crier au desaccord.
        notes.append(
            "PPO COUPE : aucune mise a jour de politique ni de critic. Le "
            "gradient d'acteur nul est voulu. Le tronc n'apprend que par la "
            "tete de rang, et le budget se deduit du rang du score.")
    elif desaccord:
        notes.append(
            f"DESACCORD SUR L'ETAT DE LA POLITIQUE : l'epoch {ep} est "
            f"{'dans' if gele else 'hors'} le warmup ({WARMUP} epochs) mais "
            f"son gradient vaut {g_actor:.1e}. La configuration lue par la "
            f"veille n'est peut-etre pas celle du run.")
    if gele and herite:
        # UN FOLD CHAINE NE PART PAS DU HASARD. Ses epochs a actor gele
        # mesurent la politique HERITEE du fold precedent, appliquee a une
        # fenetre qu'elle n'a pas encore vue. C'est une reference utile — le
        # niveau avant tout nouvel apprentissage — mais l'appeler "hasard"
        # ferait lire un transfert reussi comme un coup de chance.
        notes.append(f"POLITIQUE GELEE, POIDS HERITES : warmup du critic, gradient "
                     f"{g_actor:.1e}. Cette epoch mesure la politique du fold "
                     f"PRECEDENT sur cette fenetre — pas le hasard. C'est le "
                     f"niveau de depart que la suite doit battre.")
    elif gele:
        notes.append(f"POLITIQUE GELEE : warmup du critic, gradient {g_actor:.1e}. "
                     f"Cette epoch ne mesure aucun apprentissage — elle sert de "
                     f"reference au hasard pour les suivantes.")
    elif H > 0.99 * h_max:
        # LES DEUX CHIFFRES PORTENT SUR DEUX TETES, et c'est le
        # rapprochement qui est informatif. Le gradient est celui du BUDGET,
        # que PPO entraine ; l'entropie est celle de la REGLE D'ENTREE, que
        # la tete de rang produit. PPO apprend donc a dimensionner pendant
        # que la selection entre au hasard — et c'est la selection, pas le
        # dimensionnement, qui fait le resultat.
        notes.append(f"LE BUDGET APPREND, L'ENTREE TIRE A PILE OU FACE : le "
                     f"gradient de budget passe ({g_actor:.1e}) mais "
                     f"l'entropie des entrees tient a {H:.3f} sur "
                     f"{h_max:.3f}. La tete de rang ne differencie pas encore "
                     f"les occasions — regarder `rhoAux` et l'etendue, pas le "
                     f"PnL, qui ne mesure ici que le tirage.")
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
    if mhz is not None and mhz < 500:
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
        self._meta_muettes = set()
        self.vus = set()
        self.vals, self.metas, self.trains, self.rhos = {}, {}, {}, {}
        self.nets = {}          # (net, gain, creux)
        self.cotes = {}         # par fold : long / short / both, lu du tag
        self.herites = set()    # folds dont les poids viennent du precedent
        self.sommets = {}       # par (fold, epoch) : (gain du sommet, au hasard)
        self.tenues = {}        # (med G, moy G, med P, moy P, rapport)
        self.a_battre = {}      # par (fold, epoch) : seuil de sauvegarde
        self.records = {}       # par fold : score net du dernier retenu
        self.hist_nets = {}     # par fold : score net des epochs mesurees
        self._fold_lu = None    # fold de la derniere ligne META LUE
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
                  self.sommets, self.reprises, self.nets, self.tenues,
                  self.a_battre, self.records, self.hist_nets):
            d.clear()
        self._fold_lu = None
        self.herites.clear()
        self.attend_moyenne.clear()
        self.fold_courant = None

    def _bloc_retenue(self, ligne: str):
        """Ce qui vient d'etre sauve, et sur quel critere.

        SI LE MOTIF NE MORD PAS, ON REAFFICHE LA LIGNE BRUTE. Une ligne de
        retenue avalee en silence serait la pire panne de ce fichier : c'est
        elle qui dit quel modele sera deploye.
        """
        m = RE_BEST.search(ligne)
        if m is None:
            return [f"{C.VERT}{C.GRAS}  {ligne.strip()}{C.FIN}",
                    f"{C.ROUGE}  VEILLE : ligne NEW BEST illisible, motif a "
                    f"reajuster{C.FIN}"]
        (net, gain, baisse, bat,
         sommet, hasard, trades, fichier) = m.groups()
        net, gain, baisse = float(net), float(gain), float(baisse)
        sommet, hasard = float(sommet), float(hasard)
        ou = (f" — {self.fold_courant.upper()}" if self.fold_courant else "")
        mieux = ""
        if bat != "premier retenu du fold":
            try:
                mieux = f"   -> {net - float(bat.rstrip(' R')):+.3f} R de mieux"
            except ValueError:
                mieux = ""
        return [
            "",
            f"{C.VERT}{C.GRAS}  ★ MODELE RETENU{ou}{C.FIN}",
            f"{C.VERT}    critere     SCORE NET {C.GRAS}{net:+.3f} R{C.FIN}"
            f"{C.VERT} par occasion  "
            f"= gain {gain:+.3f} R - baisse {baisse:.3f} R{C.FIN}",
            f"{C.GRIS}                en R d'une POSITION MINIMALE : ce qu'une "
            f"occasion rapporte, misee comme le modele la mise,{C.FIN}",
            f"{C.GRIS}                moins la taille typique de ses "
            f"pertes{C.FIN}",
            f"    a battu     " + ("le premier retenu de ce fold"
                                   if bat == "premier retenu du fold"
                                   else f"le record precedent, {bat}{mieux}"),
            f"{C.GRIS}    portillons  sommet {sommet:+.3f} R > {hasard:+.3f} "
            f"au hasard   {trades} trades   compte intact{C.FIN}",
            f"{C.GRIS}    fichier     {fichier}{C.FIN}",
            f"{C.VERT}    C'est CE modele qui part au fold suivant, et c'est "
            f"lui qu'on deploierait.{C.FIN}",
            "",
        ]

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
            if "NEW BEST PROFIT" in ligne:
                # LE TEMOIN, PAS LE CRITERE. Deux etoiles se suivaient dans le
                # journal sans que rien ne dise laquelle comptait. Celle-ci
                # suit le PnL par trade — ~55 trades, erreur-type 2.81 $ pour
                # un gain de 5.30 $ — et selectionner la-dessus coute -3.3
                # points mesures ici.
                blocs.append(([f"{C.GRIS}  {ligne.strip()}{C.FIN}"], None))
            elif "NEW BEST" in ligne:
                blocs.append((self._bloc_retenue(ligne), None))
                _mb = RE_BEST.search(ligne)
                if _mb and self._fold_lu is not None:
                    self.records[self._fold_lu] = float(_mb.group(1))
            mg = RE_GARDE.search(ligne)
            if mg:
                blocs.append(([f"{C.GRIS}  non retenu — {mg.group(1)}"
                               f"{C.FIN}"], None))
            mrf = RE_REFUS.search(ligne)
            if mrf:
                blocs.append(([
                    "",
                    f"{C.ROUGE}{C.GRAS}  × REFUSE MALGRE UN MEILLEUR SCORE"
                    f"{C.FIN}",
                    f"{C.ROUGE}    score net {float(mrf.group(1)):+.3f} R "
                    f"par occasion, contre {float(mrf.group(2)):+.3f} R au "
                    f"record — mais {mrf.group(3)}.{C.FIN}",
                    f"{C.GRIS}    Le classement etait meilleur et c'est le "
                    f"PORTEFEUILLE qui l'a disqualifie. Ce modele n'est ni "
                    f"retenu{C.FIN}",
                    f"{C.GRIS}    ni transmis au fold suivant : le score par "
                    f"occasion ne voit pas combien de positions sont tenues "
                    f"ensemble.{C.FIN}",
                    "",
                ], None))
            mv, mm = RE_VAL.search(ligne), RE_META.search(ligne)
            # UNE LIGNE META QUI NE SE LIT PAS DOIT LE DIRE. C'est la panne
            # qui s'est repetee huit fois : un champ change de forme dans
            # `training`, le motif ne mord plus, et la veille n'affiche plus
            # rien SANS RIEN SIGNALER. Desormais elle nomme la ligne fautive,
            # une fois, et continue.
            if mm is None and "  META  " in ligne:
                _cle = ligne[:40]
                if _cle not in self._meta_muettes:
                    self._meta_muettes.add(_cle)
                    print(f"{C.ROUGE}  VEILLE : ligne META illisible, "
                          f"motif a reajuster{C.FIN}")
                    print(f"{C.GRIS}  {ligne.strip()[:200]}{C.FIN}")
                    print()
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
                self._fold_lu = mm.group(1)
                mb = RE_BATTRE.search(ligne)
                self.a_battre[(mm.group(1), int(mm.group(2)))] = (
                    float(mb.group(1)) if mb
                    else max(0.0, self.records.get(mm.group(1), 0.0)))
                mt = RE_TENUE.search(ligne)
                if mt:
                    self.tenues[(mm.group(1), int(mm.group(2)))] = tuple(
                        float(x) for x in mt.groups())
                ms = RE_SOMMET.search(ligne)
                if ms:
                    self.sommets[(mm.group(1), int(mm.group(2)))] = (
                        float(ms.group(1)), float(ms.group(2)))
                mn = RE_NET.search(ligne)
                if mn:
                    self.nets[(mm.group(1), int(mm.group(2)))] = tuple(
                        float(x) for x in mn.groups())
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
                self.sommets.get(cle), self.reprises.get(cle),
                self.nets.get(cle), self.tenues.get(cle),
                self.a_battre.get(cle), list(self.hist_nets.get(fold, [])))
            # L'HISTORIQUE NE GARDE QUE LES MESURES REELLES : une epoch
            # `[val epN]` reaffiche le score d'une autre, le compter deux fois
            # dessinerait un plateau qui n'existe pas.
            if _reprise is None and cle in self.nets:
                self.hist_nets.setdefault(fold, []).append(
                    self.nets[cle][0])
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
