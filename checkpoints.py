"""Quel checkpoint charger — une seule reponse, pour tout le monde.

POURQUOI CE FICHIER EXISTE. Chaque consommateur du modele portait ses propres
chemins ecrits en dur : `kairos_live` sur `exec11`, le stress-test sur un
fichier supprime, l'ensemble sur `exec18` et `exec20`. Ils derivaient
separement, et rien ne le signalait — un chemin obsolete ne leve pas d'erreur
tant que le fichier existe, il fait seulement trader un modele qui n'est plus
celui qu'on croit.

TROIS FAMILLES, ET ELLES NE DISENT PAS LA MEME CHOSE.

    last_*         MOYENNE des N derniers jeux de poids. C'est le modele a
                   deployer : il ne depend d'aucun tirage particulier.
    best_*         meilleur Sortino de validation.
    bestprofit_*   meilleur PnL par trade de validation.

LES DEUX DERNIERES SONT UN CHOIX FAIT SUR LA VALIDATION, et ce depot a mesure
ce que ce choix coute : -3.3 points par rapport a la moyenne des poids. C'est
pourquoi `last_` est le defaut et que demander "le meilleur" rend quand meme
`last_` en le disant. Le "meilleur" checkpoint est le plus mauvais deployable.

LE PLUS RECENT SE LIT SUR LE NUMERO D'EXEC, PAS SUR LA DATE DU FICHIER. Un
`touch`, une copie ou une synchronisation changent la date sans rien changer au
modele ; le numero, lui, ne bouge que quand un run part.

    python checkpoints.py            # ce que chaque famille resout aujourd'hui
"""

from __future__ import annotations

import glob
import json
import os
import re

FAMILLES = ("last", "best", "bestprofit")
# Le prefixe des runs de ce depot. Un nom qui ne le suit pas n'est pas un run.
MOTIF = re.compile(r"^(last|best|bestprofit)_(.*?_exec(\d+)[a-z_]*)_wf(\d+)_")


def runs_disponibles() -> dict[str, dict]:
    """Rend {prefixe_de_run : {exec, folds, familles}}, par numero d'exec."""
    trouves: dict[str, dict] = {}
    for f in glob.glob("*_wf*_both_wf*.pth"):
        m = MOTIF.match(f)
        if not m:
            continue
        famille, base, num, fold = m[1], m[2], int(m[3]), int(m[4])
        d = trouves.setdefault(base, {"exec": num, "folds": set(),
                                      "familles": set()})
        d["folds"].add(fold)
        d["familles"].add(famille)
    return trouves


def _compatible(base: str, famille: str, fold: int) -> bool:
    """Le checkpoint lit-il la MEME observation que le pipeline courant ?

    C'est le filtre le plus important de ce fichier. Sans lui, "le plus recent
    complet" rendait `exec20` — trois folds, mais 107 colonnes, la lignee H1 —
    quand le pipeline en porte 264. `build_policy` le refuserait, donc la
    panne serait visible ; mais proposer un modele incompatible comme "le plus
    recent" est deja une reponse fausse, et un jour la largeur coincidera par
    accident entre deux lignees differentes.
    """
    from saint_core import OBS_N_FEATURES
    try:
        d = decrit(f"{famille}_{base}", fold)
    except (FileNotFoundError, ValueError):
        return False
    n = d.get("n_features")
    # UN CHAMP ABSENT N'EST PAS UNE AUTORISATION. Les calib anterieurs a
    # exec14 n'ecrivaient pas `n_features` ; les accepter par defaut faisait
    # resoudre `exec13` comme "le plus recent complet" alors qu'il lit un autre
    # jeu de colonnes. Une lignee inconnue n'est pas une lignee compatible.
    return n is not None and int(n) == int(OBS_N_FEATURES)


def dernier_run(min_folds: int = 1, famille: str = "last") -> str | None:
    """Le prefixe du run le plus RECENT qui porte assez de folds utilisables.

    `min_folds` refuse un run interrompu au premier fold : trois folds sont ce
    qui permet de dire quoi que ce soit hors echantillon, et charger le fold 1
    d'un run mort donnerait un modele entraine sur le tiers de l'historique
    sans que rien ne l'annonce.
    """
    candidats = [
        (d["exec"], base) for base, d in runs_disponibles().items()
        if len(d["folds"]) >= min_folds and famille in d["familles"]
        and _compatible(base, famille, sorted(d["folds"])[0])
    ]
    if not candidats:
        return None
    return max(candidats)[1]


def resoud(prefixe: str | None = None, famille: str = "last",
           min_folds: int = 1) -> tuple[str, list[int]]:
    """Rend (prefixe complet avec sa famille, folds disponibles).

    `prefixe` force un run precis ; sinon on prend le plus recent. La famille
    demandee est respectee, mais `best`/`bestprofit` declenchent un
    avertissement : ce sont des choix faits sur la validation.
    """
    if famille not in FAMILLES:
        raise ValueError(f"famille inconnue : {famille} (parmi {FAMILLES})")
    base = prefixe or dernier_run(min_folds=min_folds, famille=famille)
    if base is None:
        raise FileNotFoundError(
            f"aucun run avec au moins {min_folds} fold(s) et une famille "
            f"'{famille}' dans {os.getcwd()}")
    infos = runs_disponibles().get(base)
    if infos is None:
        raise FileNotFoundError(f"aucun checkpoint pour le prefixe {base}")
    from saint_core import OBS_N_FEATURES
    fold0 = sorted(infos["folds"])[0]
    if famille in infos["familles"] and not _compatible(base, famille, fold0):
        n = decrit(f"{famille}_{base}", fold0).get("n_features", "un nombre non declare de")
        raise FileNotFoundError(
            f"{famille}_{base} lit {n} colonnes, le pipeline en porte "
            f"{OBS_N_FEATURES} : lignee incompatible, les poids ne decrivent "
            f"pas la meme observation.")
    if famille not in infos["familles"]:
        raise FileNotFoundError(
            f"{base} n'a pas de checkpoint '{famille}' "
            f"(disponibles : {sorted(infos['familles'])})")
    if famille != "last":
        print(f"  ATTENTION : '{famille}' est un choix fait sur la VALIDATION. "
              f"Mesure dans ce depot : -3.3 points contre la moyenne des poids "
              f"('last'), qui ne depend d'aucun tirage.")
    return f"{famille}_{base}", sorted(infos["folds"])


def chemins(prefixe_complet: str, fold: int) -> tuple[str, str, str]:
    """Les trois fichiers d'un checkpoint. Ils ne valent que pris ensemble.

    Le `.pth` seul ne se charge pas — il faut le `_calib.json` pour le
    lookback, qui n'est PAS deductible des poids — et sans `_norm.npz`
    l'observation n'est pas a l'echelle sur laquelle le reseau a appris.
    """
    base = f"{prefixe_complet}_wf{fold}_both_wf{fold}"
    return base + ".pth", base + "_calib.json", base + "_norm.npz"


def decrit(prefixe_complet: str, fold: int) -> dict:
    """Ce que le checkpoint dit de lui-meme : architecture, colonnes, geometrie."""
    _, calib, _ = chemins(prefixe_complet, fold)
    with open(calib, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    trouves = runs_disponibles()
    if not trouves:
        print(f"aucun checkpoint dans {os.getcwd()}")
        return 1
    print(f"{'run':<44} {'exec':>5} {'folds':>7}  familles")
    print("-" * 82)
    for base, d in sorted(trouves.items(), key=lambda x: x[1]["exec"]):
        print(f"{base:<44} {d['exec']:>5} {len(d['folds']):>7}  "
              f"{', '.join(sorted(d['familles']))}")

    print()
    for famille in FAMILLES:
        try:
            pref, folds = resoud(famille=famille)
            n = decrit(pref, folds[0]).get("n_features", "?")
            print(f"{famille:<12} -> {pref}  folds {folds}  {n} colonnes")
        except FileNotFoundError as e:
            print(f"{famille:<12} -> {e}")

    print("\nCe que chargent le live et les backtests, sauf prefixe impose :")
    try:
        pref, folds = resoud(min_folds=3)
        print(f"  {pref}, ses {len(folds)} folds")
    except FileNotFoundError as e:
        print(f"  RIEN : {e}")
        print("  (aucun run COMPATIBLE n'a encore fini ses trois folds ;")
        print("   les lignees anterieures lisent un autre jeu de colonnes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
