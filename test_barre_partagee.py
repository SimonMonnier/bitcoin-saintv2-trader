# -*- coding: utf-8 -*-
"""Le rollout et la validation appliquent-ils LA MEME regle d'entree ?

CE QUI A RENDU CE TEST NECESSAIRE, le 2026-09-21. Le journal montrait :

    rollout      ENV [B 24.2 %  S 0.0 %]
    validation   2 142 longs, 1 743 courts

Deux strategies sous le meme nom. La cause : le rollout decidait avec
`(conf_thr, conf_thr)` — UNE barre, un niveau herite de l'epoch
precedente — quand la validation utilise DEUX barres, une par cote,
recalculees en quantile de leur propre fenetre.

Le bloc de commentaire au-dessus de l'appel expliquait pourtant sur trente
lignes que le rollout devait decider par rang glissant « COMME LE
DEPLOIEMENT ». L'objet etait construit, alimente par `observe()`, et sa
barre n'etait jamais lue. Le bandeau affichait `conf_thr` sous un
commentaire affirmant qu'il n'affichait pas `conf_thr`.

DEUX CONSEQUENCES, ET LES DEUX ETAIENT VISIBLES AU JOURNAL :

  UNE BARRE POUR DEUX TETES. `tete_achat` et `tete_vente` sont deux
  couches lineaires distinctes ; rien n'oblige leurs scores a vivre sur
  la meme echelle. Un decalage de 0.6 sur la sigmoide suffit a ce qu'une
  barre commune ne laisse passer que des achats.

  UN NIVEAU HERITE NE SUIT PAS LA DISTRIBUTION. Barre 0.000 a l'epoch 1,
  0.945 a l'epoch 2 — tout ferme — 0.504 a l'epoch 3.

    python test_barre_partagee.py
"""
import io
import sys

import numpy as np

from saint_core import (EntryDecisionPolicy, decide_avec_barres,
                        rolling_decision_spec)

_ok = _ko = 0


def verifie(nom, cond, detail=""):
    global _ok, _ko
    if cond:
        _ok += 1
        print("  ok   %-52s %s" % (nom, detail))
    else:
        _ko += 1
        print("  ECHEC %-51s %s" % (nom, detail))


# ============================================================
print("\n1. LE ROLLOUT LIT LA BARRE DE SON RANG GLISSANT")
# ============================================================
src = io.open("training.py", encoding="utf-8").read()
verifie("il decide sur `train_decisions[k].thresholds`",
        "train_decisions[k].thresholds, cfg.side" in src)
verifie("il ne decide plus sur `(conf_thr, conf_thr)`",
        "(conf_thr, conf_thr), cfg.side" not in src,
        "le niveau herite ne commande plus")
verifie("le bandeau affiche les DEUX barres du rollout",
        "_thr_roll = train_decisions[0].thresholds" in src
        and "trB" in src and "trS" in src,
        "et non un seul nombre qui cachait l'asymetrie")

# ============================================================
print("\n2. DEUX TETES D'ECHELLES DIFFERENTES : QUE FAIT CHAQUE BARRE ?")
print("   Les scores d'achat sont decales de +0.6 en logit sur ceux de")
print("   vente. C'est un ecart que rien n'interdit entre deux couches.")
# ============================================================
rng = np.random.default_rng(0)
n = 20000
sa = 1.0 / (1.0 + np.exp(-rng.normal(+0.3, 1.0, n)))
sv = 1.0 / (1.0 + np.exp(-rng.normal(-0.3, 1.0, n)))
q = 0.05

commune = float(np.quantile(np.concatenate([sa, sv]), 1 - q))
pa1, pv1 = float(np.mean(sa > commune)), float(np.mean(sv > commune))
part1 = pv1 / max(pa1 + pv1, 1e-12)
verifie("une barre COMMUNE deseequilibre les deux cotes",
        part1 < 0.35,
        "%.0f %% de ventes seulement (barre %.3f)" % (100 * part1, commune))

spec = rolling_decision_spec(q / 2, 4000, [list(sa[:4000]), list(sv[:4000])],
                             "both")
pol = EntryDecisionPolicy(spec)
ba, bv = pol.thresholds
pa2, pv2 = float(np.mean(sa > ba)), float(np.mean(sv > bv))
part2 = pv2 / max(pa2 + pv2, 1e-12)
verifie("deux barres PAR COTE les reequilibrent",
        0.35 < part2 < 0.65,
        "%.0f %% de ventes (barres %.3f / %.3f)" % (100 * part2, ba, bv))
verifie("  et la barre de vente est PLUS BASSE que celle d'achat",
        bv < ba, "%.3f contre %.3f — chaque cote a son echelle" % (bv, ba))

# ============================================================
print("\n3. LA REGLE EST LA MEME FONCTION DES DEUX COTES")
print("   `decide_avec_barres` est la source unique. Ce qui peut diverger")
print("   n'est pas la fonction, c'est le COUPLE de barres qu'on lui donne.")
# ============================================================
# ON COMPTE LA PART DE VENTES PARMI LES ENTREES, pas leur nombre brut.
# Une premiere version de ce test exigeait deux fois plus de ventes en
# nombre ; elle echouait — 70 contre 79 — parce que les deux barres
# retiennent BEAUCOUP MOINS d'occasions au total. C'est la PROPORTION
# qui dit si un cote est etouffe, pas le compte.
def _part_ventes(barres):
    a = [decide_avec_barres(float(x), float(y), barres, "both")
         for x, y in zip(sa, sv)]
    nb, ns = a.count(0), a.count(1)
    return ns / max(nb + ns, 1), nb, ns


_pc, _nb_c, _ns_c = _part_ventes((commune, commune))
_pd, _nb_d, _ns_d = _part_ventes((ba, bv))
verifie("une barre commune etouffe un cote",
        _pc < 0.30,
        "%.1f %% de ventes (%d achats, %d ventes)" % (100 * _pc, _nb_c, _ns_c))
verifie("deux barres retablissent l'equilibre",
        0.40 < _pd < 0.60,
        "%.1f %% de ventes (%d achats, %d ventes)" % (100 * _pd, _nb_d, _ns_d))

# LA RACINE DU DESEQUILIBRE : la regle departage par la marge au-dessus
# de SA PROPRE barre. Avec une barre commune les deux termes se
# simplifient et il ne reste que `p_achat >= p_vente` — donc si une tete
# sort systematiquement au-dessus de l'autre, elle prend TOUT.
verifie("  a barre commune, la regle se reduit a comparer les scores",
        all(decide_avec_barres(float(x), float(y), (0.0, 0.0), "both")
            == (0 if x >= y else 1)
            for x, y in zip(sa[:500], sv[:500])),
        "c'est pourquoi une barre a zero n'est PAS neutre")

# ============================================================
print("\n4. UN NIVEAU FIGE NE SUIT PAS UNE DISTRIBUTION QUI BOUGE")
print("   C'est la seconde faute : `conf_thr` etait herite de l'epoch")
print("   precedente. Si l'etendue des scores triple, il ne vaut plus rien.")
# ============================================================
_serre = 1.0 / (1.0 + np.exp(-rng.normal(0.0, 0.2, n)))   # convictions serrees
_large = 1.0 / (1.0 + np.exp(-rng.normal(0.0, 1.5, n)))   # puis etalees
_herite = float(np.quantile(_serre, 1 - q))
verifie("un niveau herite laisse passer bien trop d'occasions",
        float(np.mean(_large > _herite)) > 3 * q,
        "%.1f %% au lieu de %.1f %% vises"
        % (100 * np.mean(_large > _herite), 100 * q))
_spec2 = rolling_decision_spec(q, 4000, [list(_large[:4000])] * 2, "both")
_b2 = EntryDecisionPolicy(_spec2).thresholds[0]
verifie("un quantile recalcule tient le taux vise",
        abs(float(np.mean(_large > _b2)) - q) < 0.02,
        "%.1f %% pour %.1f %% vises" % (100 * np.mean(_large > _b2), 100 * q))

print("\n%d/%d OK" % (_ok, _ok + _ko))
if _ko:
    print("\nLE ROLLOUT ET LA VALIDATION NE JOUENT PLUS LA MEME REGLE. Le")
    print("modele apprend alors sa barre sur une strategie, et se fait")
    print("mesurer sur une autre.")
sys.exit(1 if _ko else 0)
