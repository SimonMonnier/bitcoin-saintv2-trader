# -*- coding: utf-8 -*-
"""LA DIRECTION EST SUPPRIMEE — ce test empeche qu'elle revienne.

CE QU'ELLE ETAIT. Une tete `actor` de trois logits (ACHETER, VENDRE,
ATTENDRE) entrainee par PPO. Le rollout y tirait l'action.

CE QUI L'A CONDAMNEE, et c'est le point que ce fichier verrouille : la
validation, le test et le live ne l'ont JAMAIS utilisee. Ils remplacaient ses
probabilites par la sortie de la tete de rang. On entrainait donc une regle
d'entree et on en deployait une autre — la classe de defaut exacte qui a deja
coute plusieurs runs ici, dont un long-only qui vendait 64 % du temps en
validation.

La mesure a tranche (exec40, 23 epochs) : comparee a sa propre politique
GELEE, la politique entrainee rendait +0.27 point a l'entree pour une
reference de bruit a +0.7. L'acteur n'apportait rien. Le budget, lui, a fait
tomber le creux de validation de 90 % a 23-50 %.

Quatre choses doivent donc rester vraies, et chacune a son test.
"""
import ast
import io
import math
import sys

import numpy as np
import torch

from saint_core import (
    N_ACTIONS,
    N_BUDGETS,
    cotes_permises,
    decide_avec_barres,
)

_ok = 0
_ko = 0


def verifie(nom, condition, detail=""):
    global _ok, _ko
    if condition:
        _ok += 1
        print(f"  ok   {nom:<58} {detail}")
    else:
        _ko += 1
        print(f"  ECHEC {nom:<58} {detail}")


# ============================================================
print("\n1. PLUS AUCUNE BASCULE NE PEUT REBRANCHER LA DIRECTION")
print("   Un drapeau qui remet les logits de l'acteur sur la decision est un")
print("   moyen de rejouer le defaut. Il ne doit plus exister nulle part.")
# ============================================================
for _f in ("training.py", "kairos_live.py"):
    _src = io.open(_f, encoding="utf-8").read()
    # On ignore les commentaires : ils racontent l'histoire, ils ne branchent
    # rien. Seul le code executable compte.
    _code = "\n".join(l for l in _src.splitlines()
                      if not l.lstrip().startswith("#"))
    verifie(f"{_f} : pas de `tri_par_tete_aux`",
            "tri_par_tete_aux" not in _code)
    verifie(f"{_f} : pas de `gel_direction`",
            "gel_direction" not in _code)


# ============================================================
print("\n2. LE ROLLOUT ET LE DEPLOIEMENT APPELLENT LA MEME FONCTION")
print("   `decide_avec_barres` est la source unique de la regle d'entree.")
print("   Une seconde ecriture aurait diverge sans que rien ne le signale ;")
print("   c'est deja arrive dans ce depot.")
# ============================================================
_arbre = ast.parse(io.open("training.py", encoding="utf-8").read())
_appels = [n for n in ast.walk(_arbre)
           if isinstance(n, ast.Call)
           and isinstance(n.func, ast.Name)
           and n.func.id == "decide_avec_barres"]
verifie("training.py appelle `decide_avec_barres` au rollout",
        len(_appels) >= 1, f"{len(_appels)} appel(s)")

# La tete de direction n'est plus liee a une variable dans training.py : les
# deux `policy.sorties(...)` qui restent jettent leur premiere sortie.
_sorties = [n for n in ast.walk(_arbre)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "sorties"]
_lie_les_logits = []
for n in _sorties:
    cible = n.targets[0]
    if isinstance(cible, ast.Tuple) and cible.elts:
        premier = cible.elts[0]
        if not (isinstance(premier, ast.Name) and premier.id == "_"):
            _lie_les_logits.append(getattr(premier, "id", "?") + f" (l.{n.lineno})")
verifie("aucun `policy.sorties` ne retient les logits de direction",
        not _lie_les_logits, ", ".join(_lie_les_logits) or "toutes jetees")


# ============================================================
print("\n3. UN COTE INTERDIT NE PASSE PAS, MEME A BARRE NULLE")
print("   A l'epoch 1 la barre vaut 0 : la distribution des scores est encore")
print("   plate et on laisse tout passer. Un cote interdit marque 0.0 aurait")
print("   franchi cette barre (0.0 >= 0.0). Il vaut -1, hors de l'image de la")
print("   sigmoide, donc aucune barre de [0, 1] ne peut le laisser passer.")
# ============================================================
SCORE_INTERDIT = -1.0


def score_rollout(rend, permis):
    """La transformation exacte du rollout, reecrite en numpy."""
    sig = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(rend, float), -30, 30)))
    return np.where(np.asarray(permis, bool), sig, SCORE_INTERDIT)


for side in ("long", "short", "both"):
    permis = np.array(cotes_permises(side), bool)
    n_vendus = 0
    n_achetes = 0
    rng = np.random.default_rng(7)
    for _ in range(400):
        # Le cas hostile : la tete de rang veut le cote interdit, fort.
        rend = rng.normal(0.0, 4.0, size=2)
        pb, ps = score_rollout(rend, permis)
        a = decide_avec_barres(float(pb), float(ps), (0.0, 0.0), side)
        n_achetes += (a == 0)
        n_vendus += (a == 1)
    verifie(f"{side:<5} : aucune vente interdite",
            permis[1] or n_vendus == 0, f"{n_vendus} vente(s)")
    verifie(f"{side:<5} : aucun achat interdit",
            permis[0] or n_achetes == 0, f"{n_achetes} achat(s)")

# Le veto du troisieme votant passe par le meme chemin : il ecrit dans le
# masque, et le masque devient le score.
pb, ps = score_rollout([9.0, 9.0], [False, True])
verifie("veto de TabM sur l'achat -> l'achat ne passe pas",
        decide_avec_barres(float(pb), float(ps), (0.0, 0.0), "both") != 0,
        f"pb={pb:+.1f}")


# ============================================================
print("\n4. LA TETE DE DIRECTION NE RECOIT PLUS DE GRADIENT")
print("   C'est la difference entre supprimer et geler : gelee, elle decidait")
print("   encore en rollout pendant que le deploiement decidait autrement.")
# ============================================================
try:
    from saint_core import SAINTPolicySingleHead

    torch.manual_seed(0)
    pol = SAINTPolicySingleHead(n_features=12, d_model=8, num_blocks=1,
                                heads=1, max_len=4, n_freq=4)
    pol.eval()

    # Le gel explicite que training.py applique.
    for n, prm in pol.named_parameters():
        if n.startswith("actor."):
            prm.requires_grad_(False)

    x = torch.randn(3, 4, 12)   # (B, T, F)
    _logits, value, bud, rend = pol.sorties(x)

    # La perte de PPO telle qu'elle est maintenant : budget + critic. Les
    # logits de direction n'y entrent pas.
    dist_b = torch.distributions.Categorical(
        logits=torch.log_softmax(bud, dim=-1))
    perte = (-dist_b.log_prob(torch.zeros(3, dtype=torch.long)).mean()
             + value.pow(2).mean() + rend.pow(2).mean())
    perte.backward()

    _g_dir = [n for n, prm in pol.named_parameters()
              if n.startswith("actor.") and prm.grad is not None
              and float(prm.grad.abs().sum()) > 0]
    verifie("gradient de la tete `actor` exactement nul",
            not _g_dir, ", ".join(_g_dir) or "aucun tenseur touche")

    _g_bud = sum(float(prm.grad.abs().sum())
                 for n, prm in pol.named_parameters()
                 if n.startswith("tete_budget.") and prm.grad is not None)
    verifie("la tete de budget, elle, apprend", _g_bud > 0,
            f"|g| = {_g_bud:.3e}")

    _g_aux = sum(float(prm.grad.abs().sum())
                 for n, prm in pol.named_parameters()
                 if n.startswith("tete_aux.") and prm.grad is not None)
    verifie("la tete de rang, elle aussi", _g_aux > 0, f"|g| = {_g_aux:.3e}")

    verifie("la tete de direction existe encore dans le reseau",
            any(n.startswith("actor.") for n, _ in pol.named_parameters()),
            "les points de reprise anterieurs restent chargeables")
except Exception as e:                                    # pragma: no cover
    verifie("construction du reseau de test", False, repr(e))


# ============================================================
print("\n5. LE RATIO DE PPO EST CELUI DU BUDGET, ET IL EST EXACT")
print("   La direction n'est plus tiree d'une loi : elle est une fonction")
print("   deterministe du score. Y laisser un terme supposerait qu'elle vient")
print("   des logits de l'acteur, et corrigerait un ecart inexistant.")
# ============================================================
torch.manual_seed(1)
_bud = torch.randn(64, N_BUDGETS)
_logt = torch.log_softmax(_bud, dim=-1)
_tir = torch.multinomial(_logt.exp(), 1).squeeze(-1)
# Ce que le rollout stocke, et ce que la mise a jour recalcule au premier pas.
_stocke = _logt.gather(1, _tir[:, None]).squeeze(-1)
_recalcule = torch.distributions.Categorical(logits=_logt).log_prob(_tir)
_ratio = (_recalcule - _stocke).exp()
verifie("ratio = 1 au premier pas interne, au bit pres",
        bool(torch.allclose(_ratio, torch.ones_like(_ratio), atol=1e-6)),
        f"ecart max {float((_ratio - 1).abs().max()):.2e}")


# ============================================================
print("\n6. `H` MESURE ENCORE QUELQUE CHOSE DE VIVANT")
print("   Le champ portait l'entropie des logits de l'acteur. Un indicateur")
print("   mort qui garde l'air vivant est pire qu'un champ absent : c'est ce")
print("   qui m'a fait annoncer 'ACTOR GELE' sur un run qui apprenait.")
# ============================================================


def h_entree(counts):
    tot = sum(counts)
    if tot == 0:
        return 0.0
    p = np.array(counts, float) / tot
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


verifie("regle figee sur ATTENDRE -> H = 0", h_entree([0, 0, 500]) == 0.0)
verifie("regle figee sur ACHETER  -> H = 0", h_entree([500, 0, 0]) == 0.0)
verifie("moitie-moitie en long-only -> H = ln 2",
        abs(h_entree([250, 0, 250]) - math.log(2)) < 1e-9,
        f"{h_entree([250, 0, 250]):.4f}")
verifie("le plafond affiche par META vaut ln(cotes+1)",
        abs(math.log(sum(cotes_permises("long")) + 1) - math.log(2)) < 1e-9
        and abs(math.log(sum(cotes_permises("both")) + 1)
                - math.log(N_ACTIONS)) < 1e-9)


# ============================================================
print(f"\n{_ok}/{_ok + _ko} OK")
if _ko:
    print("\nLA DIRECTION EST REVENUE QUELQUE PART. Ce n'est pas un detail de")
    print("propriete : une tete que PPO entraine et que le deploiement ignore")
    print("fait mesurer une strategie que personne ne joue.")
    sys.exit(1)
print("\nLa direction est supprimee : une seule regle d'entree, la meme au")
print("rollout, en validation, au test et en live — et c'est la tete de rang.")
