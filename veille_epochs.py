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
    r"\[BOTH_(wf\d+)\]\s+EPOCH (\d+)\s+VAL\s+PNL\s+(" + NB + r")\$\s+"
    r"trades=\s*(\d+)\s+WR\s+([\d.]+)%\s+PF\s+([\d.]+)\s+DD\s+([\d.]+)%"
    r".*?L\((\d+)W/(\d+)L\)\s+(" + NB + r")\$"
    r"\s+S\((\d+)W/(\d+)L\)\s+(" + NB + r")\$")
RE_TRAIN = re.compile(
    r"\[BOTH_(wf\d+)\]\s+EPOCH (\d+)\s+TRAIN\s+PNL\s+(" + NB + r")\$\s+"
    r"trades=\s*(\d+)\s+WR\s+([\d.]+)%\s+PF\s+([\d.]+)\s+DD\s+([\d.]+)%")
RE_META = re.compile(
    r"\[BOTH_(wf\d+)\]\s+EPOCH (\d+)\s+META\s+Sortino\s+(" + NB + r").*?"
    r"AvgW\s+(" + NB + r")\$\s+AvgL\s+(" + NB + r")\$.*?H ([\d.]+).*?"
    r"sel\[train\s+([\d.]+)% val\s+([\d.]+)%\].*?"
    r"etendue\[tr ([\d.-]+) val ([\d.-]+)\].*?"
    r"KL\s+(" + NB + r").*?gnorm\s+([\d.]+)\s+"
    r"g\[actor ([\d.e+-]+).*?clipfrac ([\d.]+)%\s+"
    r"temps\[collecte (\d+)s maj PPO (\d+)s calibration (\d+)s "
    r"validation (\d+)s\].*?gpu\[(\d+)C (\d+)/"
    r".*?ENV \[B\s+([\d.]+)% S\s+([\d.]+)% H\s+([\d.]+)%\]")

# Seuil sous lequel le gradient de l'acteur est considere comme NUL, donc la
# politique reellement gelee. Mesure : pendant le warmup du critic il vaut
# 1e-5 a 1e-7, et des la premiere epoch entrainee il saute a 3e-1. Deux ordres
# de grandeur separent les deux regimes, le seuil n'a donc rien de delicat.
GRADIENT_NUL = 1e-3


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
            ent_prec=None):
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
    gele = g_actor < GRADIENT_NUL

    L = []
    L.append((("MODELE DEPLOYE (moyenne des poids)  " if moyenne else "")
              + f"EPOCH {ep:03d}   {par_trade:+.2f}$/trade   {trades} trades   "
                f"WR {wr:.1f}%   PF {pf:.2f}   {duree:.0f} min"))
    L.append(f"  PnL      {pnl:+10.2f}$   cumul run {cumul:+11.2f}$   "
             f"DD {dd:.1f}%   Sortino {sortino:+.3f}")
    L.append(f"  point mort {equilibre:.1f}%  ->  ecart {ecart:+.1f} pt "
             f"(+/- {err_pt:.1f} au mieux)")

    if reference is not None:
        n_ref, moy_ref = reference
        L.append(f"  vs politique gelee du meme run : {ecart - moy_ref:+.1f} pt"
                 f"   (reference {moy_ref:+.1f} pt sur {n_ref} epochs)")
    if precedent is not None:
        L.append(f"  vs epoch precedente : {par_trade - precedent:+.2f}$/trade")

    # LES DEUX SENS SEPAREMENT. Un modele qui ne gagne que d'un cote n'a pas
    # d'avantage, il a un biais directionnel — c'est ainsi que le premier
    # resultat de TabM s'est effondre, tout son gain venant d'un bloc haussier.
    def _wr(w, l):
        return 100.0 * w / max(w + l, 1)
    L.append(f"  sens    LONG  {lw:4d}W/{ll:<4d}L  {_wr(lw, ll):4.1f}%  "
             f"{lpnl:+9.2f}$   |   SHORT {sw:4d}W/{sl_:<4d}L  "
             f"{_wr(sw, sl_):4.1f}%  {spnl:+9.2f}$")
    L.append(f"  actions B {b_pct:.1f}%  S {s_pct:.1f}%  H {h_pct:.1f}%"
             f"   |   selectivite  train {sel_tr:.1f}%  val {sel_val:.1f}%")
    # L'ETAT DE LA POLITIQUE, sur sa propre ligne. L'entropie et l'etendue
    # decident de ce que les autres chiffres VEULENT DIRE : une politique quasi
    # uniforme choisit ses trades presque au hasard, et son resultat mesure le
    # tirage. Les enterrer dans une note conditionnelle les rendait invisibles
    # exactement quand elles importaient le plus.
    d_ent = "" if ent_prec is None else f" ({H - ent_prec:+.3f})"
    L.append(f"  politique  H {H:.3f}/1.099{d_ent}   etendue {et_val:.4f} "
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
    if gele:
        notes.append(f"ACTOR GELE : gradient {g_actor:.1e}, aucune mise a jour. "
                     f"Cette epoch ne mesure aucun apprentissage — elle sert de "
                     f"reference au hasard pour les suivantes.")
    elif H > 1.09:
        notes.append(f"APPREND MAIS RESTE PLAT : le gradient passe "
                     f"({g_actor:.1e}) mais l'entropie tient a {H:.3f} sur "
                     f"1.099. La politique se differencie trop lentement pour "
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
    if (lpnl > 0) != (spnl > 0):
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


def main() -> int:
    print(f"\n{C.CYAN}{C.GRAS}  KAIROS — veille d'analyse par epoch{C.FIN}")
    print(f"{C.GRIS}  {'-' * 66}")
    print(f"  journal : {JOURNAL}")
    print(f"  rapport : {RAPPORT}")
    print(f"  reference : les epochs a actor gele DU RUN EN COURS")
    print(f"  LECTURE SEULE — fermer cette fenetre n'arrete pas l'entrainement")
    print(f"  {'-' * 66}{C.FIN}\n")

    vus = set()
    metas, vals, trains = {}, {}, {}
    precedent = {}          # par fold
    geles = {}              # par fold : ecarts des epochs a actor gele
    cumul = {}              # par fold : PnL de validation cumule
    entropie = {}           # par fold : entropie de l'epoch precedente
    attend_moyenne = set()  # folds dont l'epoch suivante porte la moyenne
    position = 0
    fold_courant = None

    while True:
        try:
            # Le journal est REECRIT a chaque relance de l'entrainement. Si sa
            # taille a diminue, notre position pointe au-dela de la fin : on
            # repart du debut et on oublie ce qu'on croyait avoir vu, sinon la
            # veille resterait muette sur tout un nouveau run.
            if os.path.getsize(JOURNAL) < position:
                print(f"{C.JAUNE}  journal reecrit — nouveau run detecte, "
                      f"relecture depuis le debut{C.FIN}\n")
                position = 0
                vus.clear(); vals.clear(); metas.clear(); trains.clear()
                precedent.clear(); geles.clear(); cumul.clear()
                entropie.clear()
                attend_moyenne.clear()
                fold_courant = None
            with open(JOURNAL, encoding="utf-8", errors="replace") as f:
                f.seek(position)
                nouveau = f.read()
                position = f.tell()
        except FileNotFoundError:
            time.sleep(PAS_SONDAGE)
            continue

        for ligne in ANSI.sub("", nouveau).split("\n"):
            if "[MEMOIRE]" in ligne:
                print(f"{C.CYAN}  {ligne.strip()}{C.FIN}\n")
            if "MOYENNE DES POIDS sur les" in ligne:
                # Annonce emise par l'entrainement AVANT l'epoch concernee :
                # les poids qui suivent sont la moyenne des dernieres epochs,
                # donc le modele qui part au test.
                print(f"{C.CYAN}{C.GRAS}  {ligne.strip()}{C.FIN}")
                print()
                attend_moyenne.add(
                    ligne.split("]")[0].strip("[").split("_")[-1])
            if "[TEST]" in ligne:
                print(f"{C.CYAN}{C.GRAS}  {ligne.strip()}{C.FIN}\n")
            mv, mm = RE_VAL.search(ligne), RE_META.search(ligne)
            mt = RE_TRAIN.search(ligne)
            if mv:
                vals[(mv.group(1), int(mv.group(2)))] = mv.groups()
            if mm:
                metas[(mm.group(1), int(mm.group(2)))] = mm.groups()
            if mt:
                trains[(mt.group(1), int(mt.group(2)))] = mt.groups()

        for cle in sorted(set(vals) & set(metas) - vus):
            vus.add(cle)
            fold, ep = cle
            if fold != fold_courant:
                print(f"{C.CYAN}{C.GRAS}  ===  {fold.upper()}  ==={C.FIN}\n")
                fold_courant = fold
            ref = None
            if len(geles.get(fold, [])) >= 2:
                g = geles[fold]
                ref = (len(g), sum(g) / len(g))
            cumul[fold] = cumul.get(fold, 0.0) + float(vals[cle][2])
            est_moyenne = fold in attend_moyenne
            attend_moyenne.discard(fold)
            console, brut, par_trade, ecart, gele = analyse(
                vals[cle], metas[cle], trains.get(cle), precedent.get(fold),
                ref, cumul[fold], est_moyenne, entropie.get(fold))
            entropie[fold] = float(metas[cle][5])
            precedent[fold] = par_trade
            if gele:
                geles.setdefault(fold, []).append(ecart)
            horo = dt.datetime.now().strftime("%H:%M:%S")
            print(f"{C.GRIS}[{horo}]{C.FIN}")
            for l in console:
                print(l)
            print()
            with open(RAPPORT, "a", encoding="utf-8") as r:
                r.write(f"\n## {horo} — {fold} — {brut[0]}\n\n")
                for l in brut[1:]:
                    r.write(l.strip() + "\n")

        time.sleep(PAS_SONDAGE)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nveille arretee — l'entrainement, lui, continue.")
