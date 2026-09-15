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
REELS de l'epoch, ecart au point mort — et il lit les diagnostics a voix haute
plutot que de les laisser sous forme de colonnes.

Il ecrit aussi un rapport cumulatif dans analyse_epochs.md.
"""

import os
import re
import sys
import time
import datetime as dt

JOURNAL = "training_btc.log"
RAPPORT = "analyse_epochs.md"
# Part de trades gagnants d'un tirage AU HASARD, aux barrieres courantes.
#
# A REMESURER A CHAQUE CHANGEMENT DE SL/TP : une cible plus lointaine est
# mecaniquement moins souvent atteinte, donc cette ligne bouge avec le R:R.
# La laisser figee fait comparer un winrate a une reference qui n'est plus la
# sienne — l'erreur exacte commise en presentant le passage a R:R 2.0.
#
# Mesure sur 1 140 entrees INDEPENDANTES (espacees de 240 barres) de la
# fenetre de validation :
#
#     SL 2.0xATR  R:R 1.4   hasard 32.7 %   point mort 49.3 %   ecart 16.6 pt
#     SL 2.0xATR  R:R 2.0   hasard 26.7 %   point mort 40.7 %   ecart 14.0 pt
#
# L'ecart A COMBLER est la seule grandeur comparable entre configurations.
LIGNE_HASARD = 26.7
PAS_SONDAGE = 20             # secondes

# VIDE VOLONTAIREMENT depuis exec8. Les runs exec3 a exec7 tournaient a
# R:R 1.4 : leur PnL par trade n'est pas comparable, puisque gains et pertes
# ne sont plus dans le meme rapport. Comparer les deux ferait exactement
# l'erreur qu'on vient de corriger sur la ligne du hasard.
#
# Ce qui reste comparable d'une configuration a l'autre, c'est l'ECART AU
# POINT MORT en points, que la veille affiche deja.
REFERENCE = {}

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RE_VAL = re.compile(
    r"EPOCH (\d+)\s+VAL\s+PNL\s+(-?[\d.]+)\$\s+trades=(\d+)\s+WR\s+([\d.]+)%"
    r"\s+PF\s+([\d.]+)\s+DD\s+([\d.]+)%")
RE_META = re.compile(
    r"EPOCH (\d+)\s+META\s+Sortino\s+(-?[\d.]+).*?AvgW\s+\+?(-?[\d.]+)\$"
    r"\s+AvgL\s+(-?[\d.]+)\$.*?H ([\d.]+).*?etendue\[tr ([\d.]+) val ([\d.]+)\]"
    r".*?clipfrac ([\d.]+)%\s+temps\[collecte (\d+)s maj PPO (\d+)s "
    r"calibration (\d+)s validation (\d+)s\].*?gpu\[(\d+)C (\d+)/")


class Couleur:
    VERT, ROUGE, JAUNE, CYAN, GRIS, GRAS, FIN = (
        "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[90m",
        "\033[1m", "\033[0m")


C = Couleur


def _teinte(valeur, bon, moyen):
    """Vert si au-dessus de `bon`, jaune au-dessus de `moyen`, rouge sinon."""
    return C.VERT if valeur >= bon else (C.JAUNE if valeur >= moyen else C.ROUGE)


def analyse(v, m, precedent):
    """Rend (lignes colorees pour la console, lignes brutes pour le rapport)."""
    ep = int(v[0])
    pnl, trades, wr, pf = float(v[1]), int(v[2]), float(v[3]), float(v[4])
    sortino, avg_w, avg_l = float(m[1]), float(m[2]), abs(float(m[3]))
    H, et_tr, et_val = float(m[4]), float(m[5]), float(m[6])
    clipfrac = float(m[7])
    t_col, t_ppo, t_cal, t_val = (int(m[8]), int(m[9]), int(m[10]), int(m[11]))
    temp, mhz = int(m[12]), int(m[13])

    par_trade = pnl / max(trades, 1)
    # Le point mort se recalcule a CHAQUE epoch sur les gains et pertes
    # REELLEMENT observes, jamais sur le R:R nominal : c'est la distribution
    # realisee qui decide, pas la cible annoncee.
    equilibre = 100.0 * avg_l / (avg_w + avg_l) if (avg_w + avg_l) > 0 else float("nan")
    ecart = wr - equilibre
    duree = (t_col + t_ppo + t_cal + t_val) / 60.0

    L = []
    L.append(f"EPOCH {ep:03d}   {par_trade:+.2f}$/trade   {trades} trades   "
             f"WR {wr:.1f}%   PF {pf:.2f}   {duree:.0f} min")
    L.append(f"  point mort {equilibre:.1f}%  ->  ecart {ecart:+.1f} points"
             f"   (hasard {LIGNE_HASARD}%)")

    if precedent is not None:
        d = par_trade - precedent
        L.append(f"  vs epoch precedente : {d:+.2f}$/trade")

    if ep in REFERENCE:
        ecarts = [f"{k} {par_trade - r:+.2f}$" for k, r in sorted(REFERENCE[ep].items())]
        L.append(f"  vs architecture reduite : {'  '.join(ecarts)}")

    # ---- lecture des diagnostics ----
    notes = []
    if clipfrac < 0.5 and H > 1.09:
        notes.append("WARMUP CRITIQUE : l'actor est GELE (H = ln 3, clipfrac 0). "
                     "Ce chiffre ne mesure pas un apprentissage — toute variation "
                     "vient du curriculum et du tirage du seuil.")
    else:
        notes.append(f"entropie {H:.3f} (max 1.099) — la politique "
                     f"{'commence a se differencier' if H < 1.08 else 'reste quasi uniforme'}")
        notes.append(f"etendue val {et_val:.4f} — ecart entre les convictions du "
                     f"modele. Si elle est du meme ordre que le tremblement du "
                     f"seuil (~0.002), la selection des 5 % est quasi arbitraire.")
    if abs(par_trade) < 2.2 and precedent is not None and abs(par_trade - precedent) < 2.2:
        notes.append("ecart sous le plancher de bruit mesure (2.2 $/trade sur "
                     "trois tirages d'epoch 1) — non separable de zero.")
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

    couleur = _teinte(ecart, 0.0, -8.0)
    console = [f"{couleur}{C.GRAS}{L[0]}{C.FIN}"] + \
              [f"{couleur}{L[1]}{C.FIN}"] + \
              [f"{C.GRIS}{x}{C.FIN}" for x in L[2:]]
    return console, L, par_trade


def main() -> int:
    print(f"\n{C.CYAN}{C.GRAS}  KAIROS — veille d'analyse par epoch{C.FIN}")
    print(f"{C.GRIS}  {'-' * 66}")
    print(f"  journal : {JOURNAL}")
    print(f"  rapport : {RAPPORT}")
    print(f"  LECTURE SEULE — fermer cette fenetre n'arrete pas l'entrainement")
    print(f"  {'-' * 66}{C.FIN}\n")

    vus = set()
    metas, vals = {}, {}
    precedent = None
    position = 0

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
                vus.clear()
                vals.clear()
                metas.clear()
                precedent = None
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
                vals[int(mv.group(1))] = mv.groups()
            if mm:
                metas[int(mm.group(1))] = mm.groups()

        for ep in sorted(set(vals) & set(metas) - vus):
            vus.add(ep)
            console, brut, par_trade = analyse(vals[ep], metas[ep], precedent)
            precedent = par_trade
            horo = dt.datetime.now().strftime("%H:%M:%S")
            print(f"{C.GRIS}[{horo}]{C.FIN}")
            for l in console:
                print(l)
            print()
            with open(RAPPORT, "a", encoding="utf-8") as r:
                r.write(f"\n## {horo} — " + brut[0] + "\n\n")
                for l in brut[1:]:
                    r.write(l.strip() + "\n")

        time.sleep(PAS_SONDAGE)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nveille arretee — l'entrainement, lui, continue.")
