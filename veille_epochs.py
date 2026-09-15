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
REELS de l'epoch, ecart au point mort, erreur-type de cet ecart — et il lit
les diagnostics a voix haute plutot que de les laisser sous forme de colonnes.

Il ecrit aussi un rapport cumulatif dans analyse_epochs.md.

TROIS DEFAUTS CORRIGES LE 2026-09-15, tous revelus par le passage en H1.

  1. LE MOTIF NE MATCHAIT PLUS. `trades=(\\d+)` supposait qu'il n'y ait pas
     d'espace apres le signe egal. Le champ du journal est cadre a droite sur
     quatre caracteres : `trades=1482` en M1, mais `trades= 374` en H1, ou la
     validation fait quelques centaines de trades. La veille etait muette et
     ne disait pas pourquoi — elle attendait simplement des lignes qui ne
     viendraient jamais.

  2. LES EPOCHS ETAIENT CONFONDUES ENTRE FOLDS. Le walk-forward repart a
     l'epoch 1 a chaque fold. En ne retenant que le NUMERO, la veille
     considerait les epochs du fold 2 comme deja vues et se taisait sur les
     deux tiers du run. On retient desormais (fold, epoch).

  3. LA LIGNE DU HASARD ETAIT PERIMEE, ET LE RESTERA TOUJOURS si on la
     recopie. Elle valait 26.7 %, mesure sur M1 avec une detention de 240
     barres. En H1, aux memes barrieres, elle vaut 40.9 % : seules 67 % des
     courses se resolvent et les clotures au marche produisent beaucoup de
     petits gagnants. Comparer un winrate H1 a une reference M1 est l'erreur
     exacte qu'on s'etait deja promis de ne plus faire.

     La reference n'est donc plus une constante. Elle est LUE DANS LE RUN
     LUI-MEME : pendant le warmup du critic l'actor est gele et la politique
     est uniforme (H = ln 3, clipfrac 0), donc ces epochs SONT le tirage au
     hasard — meme fenetre, meme moteur, meme friction, meme calibrage. Aucune
     constante a tenir a jour, et la reference suit automatiquement tout
     changement de SL, de R:R, d'unite de temps ou de jeu de donnees.

     Pour memoire, les mesures qui servaient de constante :
         M1,  SL 2.0xATR  R:R 1.4   hasard 32.7 %   point mort 49.3 %
         M1,  SL 2.0xATR  R:R 2.0   hasard 26.7 %   point mort 40.7 %
         H1,  SL 2.0xATR  R:R 2.0   hasard 40.9 %   point mort 43.0 %
     (simulateur de barrieres, qui n'a pas les memes regles de sortie que
     l'environnement — raison de plus pour lire la reference dans le run.)
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
# DEUX PIEGES DE FORMAT, tous deux payes une fois.
#
#   `\s*` apres chaque signe egal : le journal cadre ses champs, donc la
#   presence d'un espace depend du nombre de chiffres. `trades=1482` en M1,
#   `trades= 374` en H1.
#
#   `[+-]?` devant chaque nombre signe : le journal ecrit `PNL  -147.55$` mais
#   `PNL  +667.73$`. Un motif qui n'accepte que le moins ne rate pas des
#   epochs au hasard — il rate PRECISEMENT LES GAGNANTES. La veille n'affichait
#   que les epochs perdantes et le run paraissait pire qu'il n'etait.
RE_NB = r"[+-]?[\d.]+"
RE_VAL = re.compile(
    r"\[BOTH_(wf\d+)\]\s+EPOCH (\d+)\s+VAL\s+PNL\s+(" + RE_NB + r")\$\s+"
    r"trades=\s*(\d+)\s+WR\s+([\d.]+)%\s+PF\s+([\d.]+)\s+DD\s+([\d.]+)%")
RE_META = re.compile(
    r"\[BOTH_(wf\d+)\]\s+EPOCH (\d+)\s+META\s+Sortino\s+(" + RE_NB + r").*?"
    r"AvgW\s+(" + RE_NB + r")\$\s+AvgL\s+(" + RE_NB + r")\$.*?H ([\d.]+).*?"
    r"etendue\[tr ([\d.]+) val ([\d.]+)\].*?clipfrac ([\d.]+)%\s+"
    r"temps\[collecte (\d+)s maj PPO (\d+)s calibration (\d+)s "
    r"validation (\d+)s\].*?gpu\[(\d+)C (\d+)/")


class Couleur:
    VERT, ROUGE, JAUNE, CYAN, GRIS, GRAS, FIN = (
        "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[90m",
        "\033[1m", "\033[0m")


C = Couleur


def erreur_type(wr, avg_w, avg_l, trades):
    """Erreur-type du gain par trade, en dollars — un PLANCHER de bruit.

    Avec un winrate p, un gain moyen w et une perte moyenne l, la variance par
    trade vaut p(1-p)(w+l)^2 si l'on reduit gains et pertes a leurs moyennes.
    C'est une SOUS-ESTIMATION : les gains varient entre eux, les pertes aussi,
    et les episodes de validation se chevauchent, ce qui reduit encore le
    nombre d'observations reellement independantes. Un ecart inferieur a ce
    chiffre n'est donc surement pas separable de zero ; un ecart superieur ne
    l'est pas forcement.

    Ce calcul remplace un plancher constant de 2.2 $/trade, releve sur trois
    epochs 1 en M1. Un plancher fige vieillit exactement comme la ligne du
    hasard : il reste tandis que l'echelle, les barrieres et le nombre de
    trades changent sous lui.
    """
    p = max(min(wr / 100.0, 1.0), 0.0)
    if trades <= 1 or (avg_w + avg_l) <= 0:
        return float("nan")
    return (avg_w + avg_l) * math.sqrt(p * (1 - p) / trades)


def analyse(v, m, precedent, reference):
    """Rend (lignes colorees, lignes brutes, gain par trade, ecart, gele)."""
    ep = int(v[1])
    pnl, trades, wr, pf = float(v[2]), int(v[3]), float(v[4]), float(v[5])
    sortino, avg_w, avg_l = float(m[2]), float(m[3]), abs(float(m[4]))
    H, et_tr, et_val = float(m[5]), float(m[6]), float(m[7])
    clipfrac = float(m[8])
    t_col, t_ppo, t_cal, t_val = (int(m[9]), int(m[10]), int(m[11]), int(m[12]))
    temp, mhz = int(m[13]), int(m[14])

    par_trade = pnl / max(trades, 1)
    # Le point mort se recalcule a CHAQUE epoch sur les gains et pertes
    # REELLEMENT observes, jamais sur le R:R nominal : c'est la distribution
    # realisee qui decide, pas la cible annoncee.
    equilibre = (100.0 * avg_l / (avg_w + avg_l)
                 if (avg_w + avg_l) > 0 else float("nan"))
    ecart = wr - equilibre
    err = erreur_type(wr, avg_w, avg_l, trades)
    err_pt = 100.0 * err / (avg_w + avg_l) if (avg_w + avg_l) > 0 else float("nan")
    duree = (t_col + t_ppo + t_cal + t_val) / 60.0
    # L'actor est gele pendant le warmup du critic : entropie maximale et
    # aucune mise a jour tronquee. Ces epochs sont le tirage au hasard.
    gele = clipfrac < 0.5 and H > 1.09

    L = []
    L.append(f"EPOCH {ep:03d}   {par_trade:+.2f}$/trade   {trades} trades   "
             f"WR {wr:.1f}%   PF {pf:.2f}   {duree:.0f} min")
    L.append(f"  point mort {equilibre:.1f}%  ->  ecart {ecart:+.1f} pt "
             f"(+/- {err_pt:.1f} au mieux)")

    if reference is not None:
        n_ref, moy_ref = reference
        L.append(f"  vs politique gelee du meme run : {ecart - moy_ref:+.1f} pt"
                 f"   (reference {moy_ref:+.1f} pt sur {n_ref} epochs)")
    if precedent is not None:
        L.append(f"  vs epoch precedente : {par_trade - precedent:+.2f}$/trade")

    # ---- lecture des diagnostics ----
    notes = []
    if gele:
        notes.append("WARMUP : l'actor est GELE (H = ln 3, clipfrac 0). Cette "
                     "epoch ne mesure aucun apprentissage — elle sert de "
                     "reference au hasard pour les suivantes.")
    else:
        notes.append(f"entropie {H:.3f} (max 1.099) — la politique "
                     f"{'commence a se differencier' if H < 1.08 else 'reste quasi uniforme'}")
    # Le diagnostic vaut pour TOUTES les epochs, gelees comprises — c'est meme
    # la qu'il mord le plus fort. Mesure du 2026-09-15 sur les 5 epochs gelees
    # d'exec12, qui partagent le meme reseau : le seuil calibre s'est deplace
    # de 0.0010 alors que l'etendue totale des convictions valait 0.0002. Le
    # seuil tremblait CINQ FOIS plus que l'ecart entre la meilleure et la pire
    # situation, donc les 5 % retenus changeaient presque entierement d'une
    # epoch a l'autre : 136 trades d'ecart et jusqu'a 10 points de resultat,
    # a reseau identique.
    #
    # C'est ce qui fabrique les faux champions. Une fois l'etendue au-dessus
    # de 0.05 le tirage cesse, et l'ecart redevient une mesure du modele.
    if et_val < 10 * TREMBLEMENT_SEUIL:
        notes.append(f"SELECTION TIREE AU SORT : etendue val {et_val:.4f} "
                     f"contre un tremblement de seuil de ~{TREMBLEMENT_SEUIL:.4f}. "
                     f"Les 5 % retenus changent presque entierement d'une epoch "
                     f"a l'autre — ce chiffre mesure le tirage, pas le modele.")
    else:
        notes.append(f"etendue val {et_val:.4f}, soit "
                     f"{et_val/TREMBLEMENT_SEUIL:.0f}x le tremblement du seuil : "
                     f"la selection est portee par le modele.")
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
    console = [f"{couleur}{C.GRAS}{L[0]}{C.FIN}", f"{couleur}{L[1]}{C.FIN}"] + \
              [f"{C.GRIS}{x}{C.FIN}" for x in L[2:]]
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
    metas, vals = {}, {}
    precedent = {}          # par fold
    geles = {}              # par fold : ecarts des epochs a actor gele
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
                vus.clear(); vals.clear(); metas.clear()
                precedent.clear(); geles.clear()
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
            mv, mm = RE_VAL.search(ligne), RE_META.search(ligne)
            if mv:
                vals[(mv.group(1), int(mv.group(2)))] = mv.groups()
            if mm:
                metas[(mm.group(1), int(mm.group(2)))] = mm.groups()

        # Tri par fold puis par epoch : le walk-forward les produit dans cet
        # ordre, et les melanger rendrait les comparaisons a l'epoch
        # precedente absurdes.
        for cle in sorted(set(vals) & set(metas) - vus):
            vus.add(cle)
            fold, ep = cle
            if fold != fold_courant:
                print(f"{C.CYAN}{C.GRAS}  ===  {fold.upper()}  ==="
                      f"{C.FIN}\n")
                fold_courant = fold
            ref = None
            if len(geles.get(fold, [])) >= 2:
                g = geles[fold]
                ref = (len(g), sum(g) / len(g))
            console, brut, par_trade, ecart, gele = analyse(
                vals[cle], metas[cle], precedent.get(fold), ref)
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
