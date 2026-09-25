# -*- coding: utf-8 -*-
"""Le motif META mord-il sur la ligne que le run a REELLEMENT ecrite ?

ATTENTION AUX INDICES : `Veilleur.avale` stocke `mm.groups()`, un TUPLE
indexe a partir de zero. `analyse` lit donc `m[i]`, qui vaut le groupe
i+1 du motif. Un test qui interroge l'objet Match directement decale tout
d'un cran — c'est ce que j'ai fait en premier, et j'en ai conclu a tort
que la veille lisait de travers depuis des jours.
"""
import io
import re

import veille_epochs as V

propre = re.sub(r"\x1b\[[0-9;]*m", "",
                io.open("training_btc.log", encoding="utf-8",
                        errors="replace").read())
# `ENV [` exclut l'ECHO de la veille elle-meme : quand elle ne sait pas
# lire une ligne, elle la reaffiche TRONQUEE. Cette copie coupee n'est pas
# une ligne du run et ne doit pas compter comme un echec.
lignes = [l for l in propre.split("\n") if "  META  " in l and "ENV [" in l]
print("lignes META dans le journal du run en cours : %d" % len(lignes))
assert lignes, "aucune ligne META — relancer quand une epoch est finie"

_ok = _ko = 0


def verifie(nom, cond, detail=""):
    global _ok, _ko
    if cond:
        _ok += 1
        print("  ok   %-50s %s" % (nom, detail))
    else:
        _ko += 1
        print("  ECHEC %-49s %s" % (nom, detail))


for l in lignes:
    mm = V.RE_META.search(l)
    if mm is None:
        verifie("la ligne META se lit", False, l[:90])
        continue
    m = mm.groups()          # LE MEME OBJET QUE CELUI QUE `analyse` RECOIT
    verifie("epoch %s : la ligne META se lit" % m[1], True,
            "Sortino %s  AvgW %s  H %s" % (m[2], m[3], m[5]))
    verifie("  les temps sont des temps",
            m[14].isdigit() and m[17].isdigit()
            and int(m[14]) < 10_000 and int(m[17]) < 10_000,
            "collecte %ss  PPO %ss  calib %ss  validation %ss"
            % (m[14], m[15], m[16], m[17]))
    verifie("  le GPU est une temperature et une frequence",
            m[18] is None or (20 <= int(m[18]) <= 110 and int(m[19]) > 100),
            "%sC  %sMHz" % (m[18], m[19]))
    _s = float(m[20]) + float(m[21]) + float(m[22]) + (
        float(m[23]) if m[23] is not None else 0.0)
    verifie("  les parts d'actions somment a 100",
            abs(_s - 100.0) < 1.5,
            "B %s S %s H %s C %s -> %.1f %%"
            % (m[20], m[21], m[22], m[23], _s))

# UN JOURNAL SANS LE CHAMP C DOIT ENCORE SE LIRE : c'est la raison d'etre
# du groupe optionnel.
ancien = lignes[0][:lignes[0].rindex(" C ")] + "]"
ma = V.RE_META.search(ancien)
verifie("un journal SANS le champ C se lit encore",
        ma is not None and ma.groups()[23] is None
        and ma.groups()[20] == V.RE_META.search(lignes[0]).groups()[20],
        "C rendu absent, B inchange")

# ET UN SIXIEME CHAMP NE DOIT PLUS RIEN CASSER.
_i = lignes[0].rindex("ENV [")
futur = lignes[0][:_i] + lignes[0][_i:].replace("%]", "% X 1.0%]")
mf = V.RE_META.search(futur)
verifie("un SIXIEME champ dans ENV ne casse plus le motif",
        mf is not None and mf.groups()[23] == V.RE_META.search(
            lignes[0]).groups()[23])

# ============================================================
# LE SEUIL DE SAUVEGARDE — ajoute le 2026-09-25.
# ============================================================
# La veille montrait le `sommet du tri` en l'annoncant comme « ce qui
# selectionne le checkpoint », et ne montrait ni le score net — celui qui
# decide — ni ce qu'il devait battre.
l0 = lignes[0]
_j = V.RE_NET.search(l0).end()
avec = l0[:_j] + "  a_battre +0.123R" + l0[_j:]
verifie("le champ `a_battre` ne decale AUCUN indice META",
        V.RE_META.search(avec) is not None
        and V.RE_META.search(avec).groups() == V.RE_META.search(l0).groups())
verifie("ni le score net",
        V.RE_NET.search(avec).groups() == V.RE_NET.search(l0).groups())
verifie("et il se lit",
        V.RE_BATTRE.search(avec) is not None
        and V.RE_BATTRE.search(avec).group(1) == "+0.123")


def _a_epoch(l, n):
    """La meme ligne META, pour une autre epoch."""
    m = V.RE_META.search(l)
    return l[:m.start(2)] + "%0*d" % (len(m.group(2)), n) + l[m.end(2):]


from test_veille_retenue import ligne_best  # noqa: E402  la f-string du run

_fold = V.RE_META.search(l0).group(1)
_best = ligne_best(net=0.412, gain=0.687, baisse=0.275,
                   bat="premier retenu du fold", sommet=0.842, hasard=0.623,
                   trades=56, fichier="best_wf1.pth")
# UN SEUL BLOC DE TEXTE : la retenue de l'epoch 1 arrive avec la ligne META
# de l'epoch 2. Le seuil de l'epoch 1 doit rester celui d'AVANT sa decision.
_v = V.Veilleur()
_v.avale("\n".join([_a_epoch(l0, 1), _best, _a_epoch(l0, 2)]) + "\n")
verifie("journal sans le champ : rien retenu -> il suffit d'etre positif",
        _v.a_battre.get((_fold, 1)) == 0.0, str(_v.a_battre.get((_fold, 1))))
verifie("apres une retenue, le seuil devient son score net",
        _v.a_battre.get((_fold, 2)) == 0.412, str(_v.a_battre.get((_fold, 2))))
_v.avale(_a_epoch(avec, 3) + "\n")
verifie("le champ ecrit par le run prime sur la reconstruction",
        _v.a_battre.get((_fold, 3)) == 0.123, str(_v.a_battre.get((_fold, 3))))

_src = io.open("veille_epochs.py", encoding="utf-8").read()
verifie("le sommet ne se dit plus « ce qui selectionne le checkpoint »",
        "f\"   [c'est ce qui selectionne le checkpoint]\"" not in _src)
_ess = [l for l in io.open("veille_fenetre.ps1", encoding="utf-8").read()
        .split("\n") if l.startswith("$essentiel")]
verifie("la vue essentielle montre le score net, son seuil, son evolution",
        len(_ess) == 1 and all(k in _ess[0] for k in
                               ("critere net", "sauvegarde ", "evolution ")))

print("\n%d/%d OK" % (_ok, _ok + _ko))
raise SystemExit(1 if _ko else 0)
