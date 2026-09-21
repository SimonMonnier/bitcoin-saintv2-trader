# -*- coding: utf-8 -*-
"""La veille lit-elle la ligne que l'entrainement ecrit VRAIMENT ?

POURQUOI CE FICHIER. C'est la panne la plus repetee du depot : un champ
change de forme dans `training.py`, le motif de `veille_epochs.py` ne mord
plus, et la veille n'affiche PLUS RIEN — sans rien signaler. On cherche alors
du cote de l'entrainement, qui va parfaitement bien. C'est arrive au moins
neuf fois, dont une ou tout un run est passe sans analyse : `rho`, `rhoAux`,
`[val epN]`, `sommet`, `table`, le bloc `gpu[?]`, les phases du chrono.

CE QUE LE TEST FAIT. Il prend une VRAIE ligne META d'un run archive, y insere
les champs recents exactement comme la f-string de `training.py` les rend, et
verifie que les motifs mordent — apres avoir retire les couleurs, parce que
c'est ce que la veille fait avant de lire (`ANSI.sub` dans `avale`). Comparer
sur la ligne brute testerait un chemin qui n'existe pas : `_Journal` ecrit les
sequences ANSI dans le fichier, elles sont toujours la.

CE QU'IL NE PEUT PAS FAIRE. Il ne fabrique pas la ligne depuis `training.py` —
il faudrait evaluer une f-string de cinquante champs avec tout son contexte.
Il verifie donc la FORME des champs, pas qu'ils soient tous presents. La
derniere garde reste celle de `veille_epochs` elle-meme, qui nomme desormais
toute ligne META illisible au lieu de se taire.

    python test_veille_motif.py
"""
from __future__ import annotations

import glob
import io
import sys

import veille_epochs as V

echecs = []


def _ligne_reference() -> str | None:
    """Une ligne META d'un run archive, couleurs retirees."""
    for motif in ("archive_*training_btc.log", "training_btc.log",
                  "*training_btc.log"):
        for chemin in sorted(glob.glob(motif), reverse=True):
            try:
                lignes = [l for l in io.open(chemin, encoding="utf-8",
                                             errors="replace")
                          if "META " in l]
            except OSError:
                continue
            for l in reversed(lignes):
                nu = V.ANSI.sub("", l).rstrip("\n")
                if V.RE_META.search(nu):
                    print(f"  reference : {chemin}")
                    return nu
    return None


def _champs_net(net, gain, baisse, bud, abst) -> str:
    """LA MEME f-string QUE `training.py`. Si elle change la, elle change ici,
    et c'est tout l'interet : les deux ecritures doivent diverger BRUYAMMENT.
    """
    return (f"net {net:>+6.3f}R "
            f"(gain {gain:>+6.3f}R baisse {baisse:>5.3f}R)  "
            f"bud {bud:>4.2f}pos abst {100*abst:>4.0f}%  ")


def verifie(nom, condition, detail=""):
    if condition:
        print(f"  ok    {nom:<44} {detail}")
    else:
        echecs.append(nom)
        print(f"  ECHEC {nom:<44} {detail}")


def main() -> int:
    base = _ligne_reference()
    if base is None:
        print("  aucune ligne META lisible dans les journaux — rien a "
              "verifier.\n  (ce n'est pas un echec : le depot peut etre "
              "vierge de tout run)")
        return 0

    print("\nLA LIGNE DE REFERENCE SE LIT")
    verifie("motif META", V.RE_META.search(base) is not None)
    verifie("motif sommet", V.RE_SOMMET.search(base) is not None)

    print("\nLES CHAMPS DU CRITERE DE RETENUE S'Y INSERENT SANS RIEN CASSER")
    print("  `net` s'intercale entre `sommet` et `Sortino`. Le motif META est")
    print("  NON GOURMAND jusqu'a `Sortino`, donc il doit tolerer n'importe")
    print("  quel champ insere la — c'est la garde qui a manque cinq fois.")
    # LES VALEURS SONT PAR OCCASION, donc petites : a 3 % de risque et
    # +0.8 R une occasion rend 2.4 % du compte. Les cas couvrent la plage
    # reelle, plus les bords qui cassent les formats.
    # LES VALEURS SONT EN R D'UNE POSITION MINIMALE, par occasion : une
    # occasion a +0.8 R misee a 3 positions vaut +2.4.
    cas = [
        ("valeurs plausibles", 0.412, 0.687, 0.275, 2.40, 0.18),
        ("net negatif", -0.810, 0.310, 1.120, 3.10, 0.02),
        ("aucune perte", 0.800, 0.800, 0.000, 1.00, 0.00),
        ("abstention totale", 0.000, 0.000, 0.000, 0.00, 1.00),
        ("gros levier", 4.870, 6.240, 1.370, 8.00, 0.00),
    ]
    for nom, net, gain, creux, bud, abst in cas:
        ligne = base.replace(
            "  Sortino ", "  " + _champs_net(net, gain, creux, bud, abst)
            + "Sortino ", 1)
        m_meta = V.RE_META.search(ligne)
        m_net = V.RE_NET.search(ligne)
        ok = m_meta is not None and m_net is not None
        if ok:
            lu = tuple(float(x) for x in m_net.groups())
            cible = (round(net, 3), round(gain, 3), round(creux, 3),
                     round(bud, 2), round(100 * abst))
            ok = all(abs(a - b) < 0.01 for a, b in zip(lu, cible))
        verifie(nom, ok,
                f"META={'oui' if m_meta else 'NON'} "
                f"net={m_net.groups() if m_net else 'NON'}")

    print("\nUN CRITERE NON MESURABLE NE REND PAS LA VEILLE MUETTE")
    print("  Sans grille de classement, `net` vaut nan et la f-string ecrit")
    print("  `+nan%`. Le motif `net` ne doit alors PAS mordre — la ligne")
    print("  s'affiche sans le bloc du critere — mais le motif META, lui,")
    print("  doit continuer de mordre, sinon toute l'epoch disparait.")
    _n = float("nan")
    ligne = base.replace("  Sortino ",
                         "  " + _champs_net(_n, _n, _n, _n, _n) + "Sortino ",
                         1)
    verifie("META mord encore", V.RE_META.search(ligne) is not None)
    verifie("net ne mord pas", V.RE_NET.search(ligne) is None)
    verifie("sommet mord encore", V.RE_SOMMET.search(ligne) is not None)

    print()
    if echecs:
        print(f"{len(echecs)} ECHEC(S) : {', '.join(echecs)}")
        print("\nLA VEILLE VA SE TAIRE. Le symptome sera le meme que les neuf")
        print("fois precedentes : plus une seule epoch affichee, aucune")
        print("erreur, et on cherchera du cote de l'entrainement.")
        return 1
    print("La veille lit ce que l'entrainement ecrit, y compris le critere")
    print("de retenue et son creux — et elle survit a un critere non mesure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
