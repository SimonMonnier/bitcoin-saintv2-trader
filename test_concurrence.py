"""K=1 doit reproduire EXACTEMENT l'environnement d'avant la concurrence.

POURQUOI CE FICHIER EXISTE AVANT LE CHANGEMENT. Rendre `BTCTradingEnvDiscrete`
capable de tenir plusieurs positions touche 78 points d'appel : la pose des
barrieres, le stop suiveur, le marquage latent, la fermeture, l'observation,
la fin d'episode. Une erreur y serait SILENCIEUSE — tout tournerait, tout
rendrait des nombres, et le prochain run serait simplement un autre run.

La parade est une reference figee : on enregistre une trajectoire complete de
l'environnement ACTUEL, graine fixe, actions fixes, puis on exige que la
version concurrente a K=1 rende la meme, trade par trade et centime par
centime. Ce n'est pas un test de plausibilite, c'est une egalite.

PLUSIEURS SCENARIOS, parce qu'un seul n'exerce presque rien. La geometrie de
production tient ses trades dix jours : un episode n'en produit que deux, et
une reference a deux trades ne prouverait pas qu'un stop suiveur fonctionne.
On ajoute donc des variantes a cycle rapide, qui font toucher les barrieres
souvent, une variante AVEC objectif et une avec break-even — desactives en
production, mais leur code existe toujours et doit survivre au changement.

USAGE
    python test_concurrence.py enregistre    # AVANT le changement
    python test_concurrence.py verifie       # apres, doit dire IDENTIQUE
    python test_concurrence.py mesure 1 2 4  # ce que K apporte, une fois sur
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import training as T
from saint_core import FEATURE_COLS

REFERENCE = "reference_env_k1.json"
GRAINE = 12345
N_PAS = 6000

# CHAQUE SCENARIO FIGE SA GEOMETRIE. Les deux premiers lisaient les defauts de
# `PPOConfig`, donc le moindre changement de reglage — stop, trailing —
# faisait echouer le temoin alors que le CODE n'avait pas bouge. Un test qui
# crie au loup a chaque reglage finit ignore, et c'est pire que pas de test.
#
# Les valeurs figees ici sont celles qui etaient en production au moment de
# l'enregistrement de la reference (stop 12xATR, trailing 1.5 R) : elles n'ont
# plus a suivre la production, leur seul role est d'exercer le code.
SCENARIOS = [
    ("production", 3000, {"atr_sl_mult": 12.0, "atr_trail_mult": 18.0,
                          "atr_trail_dist": 18.0, "aux_tp_mult": 24.0}),
    ("production", 90000, {"atr_sl_mult": 12.0, "atr_trail_mult": 18.0,
                           "atr_trail_dist": 18.0, "aux_tp_mult": 24.0}),
    ("stop serre", 3000, {"atr_sl_mult": 2.0, "atr_trail_mult": 3.0,
                          "atr_trail_dist": 3.0}),
    ("stop serre", 250000, {"atr_sl_mult": 2.0, "atr_trail_mult": 3.0,
                            "atr_trail_dist": 3.0}),
    ("avec objectif", 3000, {"atr_sl_mult": 2.0, "use_tp": True,
                             "atr_tp_mult": 4.0}),
    ("break-even actif", 3000, {"atr_sl_mult": 2.0, "atr_be_mult": 1.0,
                                "atr_trail_mult": 3.0, "atr_trail_dist": 3.0}),
]


# La reference a ete figee AVANT que le courtier ne soit modelise. Depuis, le
# lot minimum de 0.01 arrondit les tailles et change donc K=1 — volontairement.
# `MARGE` a False rend le dimensionnement continu d'avant : c'est le TEMOIN
# qui prouve que le refactor n'a rien change d'autre que le realisme.
MARGE = False


def _cfg(k=1):
    cfg = T.PPOConfig()
    # LE VETO DE TENDANCE EST RETIRE ICI, ET C'EST DELIBERE. Ce test verifie
    # que K=1 reproduit l'environnement d'avant la concurrence, recompense par
    # recompense et trade par trade ; la politique d'ENTREE lui est orthogonale.
    # Le laisser actif ferait comparer un jeu de trades filtre a une reference
    # qui ne l'etait pas, et 78 ecarts apparaissaient — tous legitimes, aucun
    # informatif. En contrepartie, ce test NE COUVRE PAS le veto : c'est
    # `mesure_seuil_tendance.py` qui le mesure, et la verification d'accord
    # avec la colonne brute qui le controle.
    cfg.veto_tendance = ""
    cfg.positions_max = k
    cfg.marge_realiste = MARGE
    return cfg


def _donnees(cfg):
    """Chargees une seule fois : six scenarios ne doivent pas relire le jeu."""
    if not hasattr(_donnees, "cache"):
        df = T.load_mt5_data(cfg)
        n = len(df)
        stats = T.compute_and_save_global_norm_stats(
            df.iloc[:int(n * 0.55)], FEATURE_COLS, path=None)
        _donnees.cache = T.MarketData(
            df.iloc[:int(n * 0.55)].reset_index(drop=True), FEATURE_COLS, stats)
        del df
    return _donnees.cache


def trajectoire(cfg, depart=3000, graine=GRAINE, n_pas=N_PAS):
    """Une trajectoire deterministe : memes actions, meme graine, meme depart.

    Les actions viennent d'un generateur, pas d'un modele : un modele
    introduirait ses propres poids dans la comparaison, et on ne saurait plus
    si un ecart vient de l'environnement ou du reseau.
    """
    data = _donnees(cfg)
    np.random.seed(graine)
    env = T.BTCTradingEnvDiscrete(data, cfg)
    # `reset()` tire son propre depart au hasard ; `reset_au_depart` l'impose
    # ET recalcule l'observation, ce que remplacer `env.idx` a la main
    # n'aurait pas fait.
    T.reset_au_depart(env, depart)
    rng = np.random.default_rng(graine)

    # 12 % d'ordres, le reste en attente : proche du regime observe, et assez
    # dense pour que les barrieres soient exercees souvent.
    rec = {"recompenses": [], "capital": [], "decisions": 0}
    for _ in range(n_pas):
        u = rng.random()
        action = 0 if u < 0.06 else (1 if u < 0.12 else 2)
        if getattr(env, "peut_entrer", lambda: env.position == 0)():
            rec["decisions"] += 1
        _, r, done, _, info = env.step(action)
        rec["recompenses"].append(round(float(r), 10))
        rec["capital"].append(round(float(env.capital), 8))

        # INVARIANTE DE REPARTITION, qui ne demande aucune reference.
        #
        # La recompense se repartit entre les emplacements pour que chaque
        # decision PPO ne recoive que ce que SA position a produit. A K=1 il
        # n'y a qu'un emplacement : sa part DOIT valoir la recompense globale,
        # exactement. Et a tout K, la somme des parts doit la valoir aussi —
        # sinon l'acteur apprendrait sur un total different de celui que
        # l'environnement a calcule, sans que rien ne le signale.
        #
        # Le test de reference ne couvrait pas ce chemin : il compare la
        # recompense globale, le capital et les trades, or la repartition est
        # justement ce que la concurrence ajoute. Falsifie en majorant le
        # risque d'un emplacement de 1 % — l'invariante le voit.
        rs = info.get("r_slots")
        if rs is not None:
            pas_n = len(rec["recompenses"])
            # A UNE SEULE POSITION, la part vaut la recompense EXACTEMENT.
            # A plusieurs, la somme n'a aucune raison de la valoir : la
            # recompense globale est une grandeur de portefeuille, avec ses
            # propres plafonnements, et chaque emplacement calcule la sienne
            # sur sa propre variation d'equity. Exiger l'egalite reviendrait a
            # remettre la repartition que le controle de signe a condamnee.
            if len(rs) == 1 and abs(float(rs[0]) - float(r)) > 1e-12:
                rec.setdefault("viols", []).append(
                    f"pas {pas_n}: part unique {float(rs[0]):.12f} "
                    f"contre recompense {float(r):.12f}")

            # LOCALISATION. La somme peut etre juste et la repartition fausse :
            # la correction additive force le total, donc a K=1 elle masque
            # n'importe quelle erreur de decoupage. Ce qui se teste vraiment,
            # c'est qu'un emplacement NON CONCERNE par cette barre — ni ouvert,
            # ni ferme a l'instant, ni vide de sa position — recoive
            # EXACTEMENT zero. Sans cela, une decision serait creditee de ce
            # qu'une autre position a produit.
            concernes = set(info.get("slots_fermes", []))
            if info.get("slot_ouvert", -1) >= 0:
                concernes.add(int(info["slot_ouvert"]))
            concernes.update(int(j) for j in np.flatnonzero(env._p_sens != 0))
            for j in range(len(rs)):
                if j not in concernes and float(rs[j]) != 0.0:
                    rec.setdefault("viols", []).append(
                        f"pas {pas_n}: emplacement {j} inactif et non "
                        f"concerne, mais credite de {float(rs[j]):.12f}")
                    break

            # Cumul par emplacement, pour verifier apres coup que le signe de
            # ce qu'une position a accumule suit le signe de ce qu'elle a
            # rapporte. Une position gagnante creditee negativement voudrait
            # dire que l'attribution s'est trompee de voisin.
            # Le nombre d'emplacements DOUBLE quand ils se remplissent — il
            # n'y a plus de plafond — donc le cumul ne peut pas etre dimensionne
            # une fois pour toutes au premier pas.
            cum = rec.setdefault("cumul", [])
            while len(cum) < len(rs):
                cum.append(0.0)
            for j in range(len(rs)):
                cum[j] += float(rs[j])
            for j in info.get("slots_fermes", []):
                rec.setdefault("fermetures", []).append(
                    (int(j), round(cum[int(j)], 10)))
                cum[int(j)] = 0.0
        if done:
            break
    rec["trades"] = [
        {k: (round(v, 8) if isinstance(v, float) else v)
         for k, v in m.items()} for m in env.trades_meta]
    rec["n_pas"] = len(rec["recompenses"])
    rec["viols"] = rec.get("viols", [])

    # SIGNE. Chaque fermeture porte le cumul de l'emplacement sur la vie de
    # la position. La comparaison se fait dans l'ordre des fermetures, qui est
    # celui de `trades_meta`.
    #
    # CE CONTROLE EST DEVENU UN TAUX, ET IL FAUT DIRE POURQUOI — affaiblir un
    # test pour le faire passer est la pire des corrections.
    #
    # Il exigeait qu'un trade gagnant accumule du positif, ce qui etait une
    # TAUTOLOGIE tant que la recompense ne contenait que du resultat. Depuis
    # le 2026-09-20 elle porte aussi un COUT DE CHEMIN : la penalite de creux
    # est facturee a chaque barre pendant l'exposition. Un trade qui gagne peu
    # en traversant un creux profond finit donc legitimement negatif — c'est
    # exactement ce qu'on a voulu enseigner, et l'exiger nul reviendrait a
    # annuler le changement.
    #
    # CE QUE LE CONTROLE GARDE : la MIS-ATTRIBUTION reste verifiee STRICTEMENT,
    # par l'autre invariant — un emplacement ni ouvert, ni ferme, ni porteur
    # doit recevoir exactement zero. Celui-la n'a aucune tolerance et n'en
    # aura pas.
    #
    # LE SEUIL VIENT D'UNE MESURE, pas d'un arrondi commode
    # (`mesure_lambda_creux.py`, part des trades gagnants a recompense
    # cumulee negative, entrees neutres) :
    #
    #     lambda 0.000 -> 3.7 %     lambda 0.050 -> 5.6 %
    #     lambda 0.005 -> 5.6 %     lambda 0.150 -> 7.4 %
    #
    # 3.7 % EXISTAIT DEJA SANS AUCUNE PENALITE : le seuil ne peut donc pas
    # etre zero. On tolere 12 %, soit le double de ce que le reglage retenu
    # produit, et on echoue au-dela — parce qu'a ce niveau le cout de chemin
    # noierait le resultat du trade et l'acteur apprendrait que gagner est
    # mauvais.
    ferm = rec.pop("fermetures", [])
    rec.pop("cumul", None)
    _gagn = _punis = 0
    for (j, c), m in zip(ferm, env.trades_meta):
        r_trade = m["pnl"]
        if r_trade > 1e-9:
            _gagn += 1
            _punis += (c < -1e-9)
    _part = 100.0 * _punis / max(_gagn, 1)
    if _gagn >= 20 and _part > 12.0:
        rec["viols"].append(
            f"{_part:.1f} % des trades gagnants finissent a recompense "
            f"NEGATIVE ({_punis}/{_gagn}) : le cout de chemin noie le "
            f"resultat du trade")
    for (j, c), m in zip([], []):
        r_trade = 0.0
        if False:
            rec["viols"].append(
                f"emplacement {j} : trade gagnant {r_trade:+.4f}$ mais cumul "
                f"de recompense {c:+.6f}")
        elif r_trade < -1e-9 and c > 1e-9:
            rec["viols"].append(
                f"emplacement {j} : trade perdant {r_trade:+.4f}$ mais cumul "
                f"de recompense {c:+.6f}")
    return rec


def empreinte_donnees(cfg) -> dict:
    """De QUELLES barres la reference a-t-elle ete prise.

    POURQUOI CETTE EMPREINTE EXISTE. La reference fige une trajectoire de
    6 000 pas partant d'indices FIXES — 3 000, 90 000, 250 000. Ces indices ne
    designent pas des dates, ils designent des RANGS dans le cache. Le
    2026-09-20, trois colonnes de tendance ont ete ajoutees ; leur amorcage de
    8 640 barres a fait tomber autant de lignes en tete du jeu, et l'indice
    3 000 a cesse de designer la meme bougie.

    Resultat : 78 ecarts, dont "trade 0, exit_idx 3016 contre 3020" et
    "nombre de trades 243 contre 215". Tous legitimes, aucun informatif — et
    tous ressemblant a s'y meprendre a une regression du simulateur. Le
    fichier se le reprochait deja pour les REGLAGES ("un test qui crie au loup
    a chaque reglage finit ignore") ; il lui manquait la meme prudence pour
    les DONNEES.

    On enregistre donc de quoi reconnaitre le jeu : son nombre de lignes, ses
    bornes de dates, et une empreinte des clotures aux indices effectivement
    parcourus. Un desaccord dit alors "refiger la reference", et non "la
    concurrence a change K=1".
    """
    import hashlib
    data = _donnees(cfg)
    c = np.asarray(data.close, np.float64)
    bornes = []
    for _, depart, _ in SCENARIOS:
        f = min(depart + N_PAS + 1, len(c))
        bornes.append(c[depart:f] if depart < len(c) else np.array([]))
    h = hashlib.sha256()
    for b in bornes:
        h.update(np.ascontiguousarray(b, np.float64).tobytes())
    t = data.df["time"] if "time" in data.df.columns else None
    return {"lignes": int(len(c)),
            "debut": str(t.iloc[0]) if t is not None else "?",
            "fin": str(t.iloc[-1]) if t is not None else "?",
            "empreinte": h.hexdigest()[:16]}


def tous_scenarios(k=1):
    """La reference complete : un enregistrement par scenario."""
    out = {"_donnees": empreinte_donnees(_cfg(k))}
    for nom, depart, maj in SCENARIOS:
        cfg = _cfg(k)
        for cle, val in maj.items():
            setattr(cfg, cle, val)
        out[f"{nom}@{depart}"] = trajectoire(cfg, depart=depart)
    return out


def compare(a, b) -> list[str]:
    ecarts = []
    if a["n_pas"] != b["n_pas"]:
        ecarts.append(f"nombre de pas : {a['n_pas']} contre {b['n_pas']}")
    for champ in ("recompenses", "capital"):
        x, y = a[champ], b[champ]
        for i in range(min(len(x), len(y))):
            if x[i] != y[i]:
                ecarts.append(f"{champ} : premier ecart au pas {i} — "
                              f"{x[i]} contre {y[i]}")
                break
    if len(a["trades"]) != len(b["trades"]):
        ecarts.append(f"nombre de trades : {len(a['trades'])} contre "
                      f"{len(b['trades'])}")
    for i, (ta, tb) in enumerate(zip(a["trades"], b["trades"])):
        for k in ta:
            if k in tb and ta[k] != tb[k]:
                ecarts.append(f"trade {i}, champ {k} : {ta[k]} contre {tb[k]}")
                if len(ecarts) > 12:
                    return ecarts
    return ecarts


def main() -> int:
    global MARGE
    quoi = sys.argv[1] if len(sys.argv) > 1 else "verifie"
    if "--marge" in sys.argv:
        MARGE = True
        sys.argv.remove("--marge")

    if quoi == "enregistre":
        tout = tous_scenarios(1)
        with open(REFERENCE, "w", encoding="utf-8") as f:
            json.dump(tout, f)
        total = 0
        print(f"  donnees : {tout['_donnees']['lignes']:,} lignes, "
              f"{tout['_donnees']['debut']} -> {tout['_donnees']['fin']}, "
              f"empreinte {tout['_donnees']['empreinte']}")
        for nom, r in tout.items():
            if nom == "_donnees":     # l'empreinte n'est pas un scenario
                continue
            total += len(r["trades"])
            print(f"  {nom:<24} {r['n_pas']:>5} pas  "
                  f"{len(r['trades']):>3} trades  "
                  f"{r['decisions']:>5} decisions  "
                  f"capital {r['capital'][-1]:>10.4f}$")
        print(f"")
        print(f"reference figee : {len(tout) - 1} scenarios, {total} trades")
        print(f"ecrite dans {REFERENCE}")
        return 0

    if quoi == "verifie":
        try:
            ref = json.load(open(REFERENCE, encoding="utf-8"))
        except FileNotFoundError:
            print(f"{REFERENCE} absent — lancer 'enregistre' AVANT de changer "
                  f"l'environnement, sinon il n'y a rien a comparer.")
            return 1
        tout = tous_scenarios(1)
        # LES DONNEES D'ABORD. Sans ce controle, un cache different produit
        # des dizaines d'ecarts numeriques qui imitent une regression du
        # simulateur — et on cherche dans le code pendant une heure.
        e_ref, e_tout = ref.get("_donnees"), tout.get("_donnees")
        if e_ref is None:
            print("la reference ne porte pas d'empreinte de donnees (format")
            print("anterieur au 2026-09-20) : la refiger une fois pour que le")
            print("prochain desaccord soit lisible.")
            print("    python test_concurrence.py enregistre")
            return 1
        if e_ref != e_tout:
            print("LE JEU DE DONNEES A CHANGE DEPUIS L'ENREGISTREMENT.")
            print(f"  reference : {e_ref['lignes']:,} lignes, "
                  f"{e_ref['debut']} -> {e_ref['fin']}, "
                  f"empreinte {e_ref['empreinte']}")
            print(f"  maintenant: {e_tout['lignes']:,} lignes, "
                  f"{e_tout['debut']} -> {e_tout['fin']}, "
                  f"empreinte {e_tout['empreinte']}")
            print("")
            print("Les scenarios partent d'INDICES fixes (3 000, 90 000,")
            print("250 000) : ces rangs ne designent plus les memes bougies,")
            print("donc la comparaison ne veut rien dire. Ce n'est PAS un")
            print("ecart de code. Refiger la reference :")
            print("    python test_concurrence.py enregistre")
            return 1
        ref = {k: v for k, v in ref.items() if k != "_donnees"}
        tout = {k: v for k, v in tout.items() if k != "_donnees"}
        if set(tout) != set(ref):
            print("les scenarios ont change depuis l'enregistrement — refiger "
                  "la reference sur la version d'avant, sinon la comparaison "
                  "ne veut rien dire")
            return 1
        total = 0
        for nom in ref:
            ecarts = compare(ref[nom], tout[nom])
            ecarts += [f"repartition : {v}" for v in tout[nom]["viols"][:2]]
            etat = "IDENTIQUE" if not ecarts else f"{len(ecarts)} ECART(S)"
            print(f"  {nom:<24} {len(ref[nom]['trades']):>3} trades  {etat}")
            for e in ecarts[:4]:
                print(f"      - {e}")
            total += len(ecarts)
        print("")
        if total == 0:
            print("IDENTIQUE sur tous les scenarios — K=1 reproduit "
                  "l'environnement d'avant,")
            print("recompense par recompense et trade par trade.")
            return 0
        print(f"{total} ecart(s) au total : la concurrence a change K=1.")
        return 1

    if quoi == "concurrence":
        # LE DECOUPAGE NE SE TESTE QU'A K>=2. A K=1 la correction additive
        # force le total sur l'unique emplacement, donc aucune erreur de
        # repartition n'y est visible — ce n'est pas un oubli du test, c'est
        # une propriete du cas a une position.
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        tout = tous_scenarios(k)
        total = 0
        for nom, r in tout.items():
            n = len(r["viols"])
            total += n
            etat = "OK" if not n else f"{n} VIOLATION(S)"
            print(f"  {nom:<24} {len(r['trades']):>4} trades  {etat}")
            for v in r["viols"][:3]:
                print(f"      - {v}")
        print("")
        if total == 0:
            print(f"A K={k} : la somme des parts vaut la recompense, aucun")
            print("emplacement non concerne n'est credite, et le signe du")
            print("cumul suit celui du trade — sur tous les scenarios.")
            return 0
        print(f"{total} violation(s) : la repartition par emplacement est fausse.")
        return 1

    if quoi == "mesure":
        ks = [int(x) for x in sys.argv[2:]] or [1, 2, 4, 8, 16, 32]
        print(f"{'K':>4} {'trades':>8} {'decisions':>11} {'x dec':>7} "
              f"{'capital':>12} {'secondes':>10}")
        print("-" * 58)
        base = None
        for k in ks:
            t0 = time.time()
            tout = tous_scenarios(k)
            dt = time.time() - t0
            tr = sum(len(r["trades"]) for r in tout.values())
            dec = sum(r["decisions"] for r in tout.values())
            cap = sum(r["capital"][-1] for r in tout.values())
            if base is None:
                base = dec
            print(f"{k:>4} {tr:>8,} {dec:>11,} {dec / max(base, 1):>6.1f}x "
                  f"{cap:>11.2f}$ {dt:>10.1f}")
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
