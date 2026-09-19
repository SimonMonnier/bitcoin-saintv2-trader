# -*- coding: utf-8 -*-
"""La penalite de creux facture-t-elle le creusement, et rien d'autre ?

POURQUOI CE TEST EXISTE. La penalite de creux valait `-0.2 si creux > 40 %`,
une marche sans gradient : rien n'indiquait au modele qu'il approchait, et il
mourait au quart de l'episode dans 5 cas sur 6. J'ai d'abord remplace la
marche par une rampe sur le NIVEAU, `-lambda x d^2`. Le pas 18 du scenario
`avec objectif` de `test_concurrence` l'a condamnee en une ligne : compte
plat, aucune position, rien d'ouvert ni de ferme, et -4.19e-05 preleve quand
meme, a chaque barre, pour un creux hérité d'un trade deja clos.

Une taxe sur un ETAT est une integrale sur le temps. A d = 0.5 elle coute
0.05 par barre, soit -43 500 sur une epoch de 870 000 barres : elle noie le
signal de trading. Et elle ne s'interrompt qu'en refaisant un sommet, ou en
MOURANT — donc elle rendait la mort plus attirante, exactement l'inverse du
but poursuivi.

LA FORME RETENUE EST CELLE DU POTENTIEL : Phi(s) = -lambda x d^2, et la
recompense recoit `Phi(s') - Phi(s)`, la seule VARIATION.

`d` A CHANGE D'ECHELLE le 2026-09-19. Il valait `creux / 0.40` et saturait
donc a 1 des 40 % de creux : au-dela, un compte a -40 % et un compte a -90 %
donnaient la meme valeur, et la penalite cessait de mordre exactement quand
la situation empirait. Il vaut maintenant le creux lui-meme — 0 au sommet,
1 quand il ne reste plus rien — parce que le garde-fou a 40 % a ete retire et
que seule la ruine termine desormais un episode.

Les quatre proprietes ci-dessous sont ce qui distingue cette forme d'une taxe
sur un etat ; si l'une tombe, le terme est redevenu une taxe.

Ce test est permanent. Il tourne sans MT5 : les donnees viennent du cache.
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

import cibles as C
import instruments as I
import training as T
from saint_core import FEATURE_COLS

N = 8000
GRAINES = (7, 19, 23)
# L'ancrage n'est plus un CHIFFRE mais un RAPPORT a l'echelle de la
# recompense : lambda doit suivre |R|. Deux ancrages successifs sur un
# chiffre (0.2 puis 1.25) ont echoue parce que le levier illimite avait
# multiplie le signal monetaire par cinq sans que la penalite suive. Voir
# `penalite_creux` dans `training.py` pour la mesure qui l'etablit.
LAMBDA_CALIBRE = 6.5
# Amplitude mediane de |R| par decision, mesuree sur exec09 epochs 1 a 7.
R_MESURE = 6.77


def _env(cfg, data, depart):
    np.random.seed(depart)
    e = T.BTCTradingEnvDiscrete(data, cfg)
    T.reset_au_depart(e, depart)
    return e


def main() -> int:
    cfg = T.PPOConfig()
    lam = cfg.penalite_creux
    cfg.episode_length = N + 10

    df = pd.read_pickle(I.INSTRUMENTS[cfg.symbol]["cache"])
    va = C.borne_etude(len(df))
    stats = T.compute_and_save_global_norm_stats(df.iloc[:va], FEATURE_COLS,
                                                 path=None)
    data = T.MarketData(df.iloc[:va].reset_index(drop=True), FEATURE_COLS,
                        stats)
    n_f = data.features.shape[1]

    print(f"lambda {lam}   d = creux, 1 a la ruine   "
          f"{len(GRAINES)} episodes de {N:,} barres au plus\n")

    ecarts = []

    # ------------------------------------------------------------------
    # 1. LE COMPTE PLAT NE PAIE RIEN. C'est le defaut qui a condamne la
    #    rampe sur le niveau, donc le premier controle.
    # ------------------------------------------------------------------
    n_plat, pire_plat = 0, 0.0
    for g in GRAINES:
        env = _env(cfg, data, 3000 + g * 4000)
        rng = np.random.default_rng(g)
        for _ in range(N):
            a = 0 if (rng.random() < 0.25 and env.peut_entrer()) else 2
            _, r, done, _, info = env.step(a)
            plat = (env.n_positions == 0
                    and not info.get("slots_fermes")
                    and int(info.get("slot_ouvert", -1)) < 0)
            if plat:
                n_plat += 1
                pire_plat = max(pire_plat, abs(float(r)))
            if done:
                break
    print("1. LE COMPTE PLAT NE PAIE RIEN")
    print(f"   barres entierement plates : {n_plat:,}")
    print(f"   |recompense| maximale sur ces barres : {pire_plat:.3e}")
    if pire_plat != 0.0:
        ecarts.append("une barre plate a ete facturee")
    print("   -> " + ("zero exact : le creux hérité ne se facture plus"
                      if pire_plat == 0.0
                      else "LE PRELEVEMENT PERMANENT EST REVENU"))

    # ------------------------------------------------------------------
    # 2. TELESCOPAGE. C'est toute la justification du choix : le total
    #    facture sur un episode ne depend PAS du temps passe sous l'eau,
    #    seulement du creux final.
    # ------------------------------------------------------------------
    print("\n2. TELESCOPAGE : le total ne depend que du creux FINAL")
    pire_tel = 0.0
    for g in GRAINES:
        env = _env(cfg, data, 3000 + g * 4000)
        rng = np.random.default_rng(g)
        cumul, k = 0.0, 0
        for _ in range(N):
            a = 0 if (rng.random() < 0.25 and env.peut_entrer()) else 2
            d_av = env._d_prec
            _, _, done, _, _ = env.step(a)
            cumul += lam * (env._d_prec ** 2 - d_av ** 2)
            k += 1
            if done:
                break
        attendu = lam * env._d_prec ** 2
        e = abs(cumul - attendu)
        pire_tel = max(pire_tel, e)
        print(f"   graine {g:>2} : {k:>5,} barres  creux final "
              f"{100*env._d_prec:>5.1f} %  cumul {cumul:+.9f}  "
              f"attendu {attendu:+.9f}  ecart {e:.2e}")
    if pire_tel > 1e-9:
        ecarts.append("le telescopage ne tient pas")
    print("   -> " + ("le total vaut le creux final, quel que soit le chemin"
                      if pire_tel <= 1e-9 else "LE TOTAL DEPEND DU CHEMIN"))

    # ------------------------------------------------------------------
    # 3. LA MAGNITUDE SUIT L'ECHELLE DE LA RECOMPENSE. Deux ancrages sur
    #    un CHIFFRE ont echoue avant celui-ci : 0.2, puis 1.25. Le levier
    #    illimite avait multiplie |R| par cinq sans que la penalite suive,
    #    et elle ne pesait plus que 4 % du signal — sept epochs sans que le
    #    creux de validation bouge d'un point. L'ancrage est donc un
    #    RAPPORT, et ce controle verifie qu'il tient.
    # ------------------------------------------------------------------
    print("\n3. LA MAGNITUDE EST CALEE SUR L'ECHELLE DE LA RECOMPENSE")
    if abs(lam - LAMBDA_CALIBRE) > 1e-12:
        ecarts.append(f"lambda vaut {lam}, calibre a {LAMBDA_CALIBRE}")
    print(f"   lambda en vigueur {lam}   calibre {LAMBDA_CALIBRE}")
    for c in (0.25, 0.40, 0.70, 1.00):
        print(f"      creux {100*c:>5.0f} % -> {lam * c * c:>6.3f}"
              + ("   (ruine)" if c == 1.0 else ""))
    # Ce qu'une position paie en traversant un creux de 50 a 70 %, rapporte
    # a l'amplitude d'une decision. Trop faible, le terme est decoratif :
    # c'est ce qui s'est passe a lambda 1.25, ou il pesait 4 % et n'a rien
    # change au creux sur sept epochs. Trop fort, il noie le signal
    # monetaire et le modele cesse d'ouvrir au lieu d'ouvrir mieux.
    part = lam * (0.7 ** 2 - 0.5 ** 2) / R_MESURE
    print(f"   un creux traverse de 50 a 70 % coute "
          f"{lam * (0.49 - 0.25):.2f} contre |R| {R_MESURE}")
    print(f"   soit {100*part:.0f} % du signal d'une decision")
    if not (0.05 <= part <= 0.40):
        ecarts.append(f"la part du signal vaut {100*part:.0f} %, hors de la "
                      f"plage 5-40 % visee")
    print("   -> " + ("senti sans dominer" if 0.05 <= part <= 0.40
                      else "PROPORTION HORS PLAGE"))

    # ------------------------------------------------------------------
    # 4. LE MODELE VOIT CE QU'IL PAIE. Si l'observation et la penalite
    #    divergent, on punit sur une grandeur invisible — le defaut qui
    #    rend une recompense inapprenable.
    # ------------------------------------------------------------------
    env = _env(cfg, data, 3000)
    rng = np.random.default_rng(5)
    pire_obs, n = 0.0, 0
    for _ in range(N):
        a = 0 if (rng.random() < 0.25 and env.peut_entrer()) else 2
        _, _, done, _, _ = env.step(a)
        # Le pas marque au prix `close[idx]` puis avance `idx` : l'observation
        # qui SUIT lit donc le meme prix, et c'est elle qu'il faut confronter.
        vu = float(env._get_obs()[-1, n_f + 4])
        pire_obs = max(pire_obs, abs(np.float32(env._d_prec) - np.float32(vu)))
        n += 1
        if done:
            break
    print(f"\n4. LE MODELE VOIT LA GRANDEUR QU'IL PAIE")
    print(f"   {n:,} barres, ecart maximal en float32 : {pire_obs:.2e}")
    if pire_obs != 0.0:
        ecarts.append("l'observation et la penalite divergent")
    print("   -> " + ("identique au bit a la 5e colonne d'observation"
                      if pire_obs == 0.0 else "IL PAIE CE QU'IL NE VOIT PAS"))

    print("")
    if ecarts:
        for e in ecarts:
            print(f"ECHEC : {e}")
        return 1
    print("La penalite facture le CREUSEMENT et rien d'autre : un compte")
    print("plat ne paie pas, le total ne depend que du creux final, et sa")
    print("magnitude est calee sur l'echelle reelle de la recompense.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
