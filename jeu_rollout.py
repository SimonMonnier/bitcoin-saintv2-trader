"""Collecte en episodes et evaluation chronologique du meme jeu.

Les episodes tires pour PPO restent independants. En evaluation, les jours
successifs d'un segment partagent positions et solde ; minuit ne ferme pas
un trade et ne libere pas sa place. Aucune issue n'entre dans l'observation
avant la cloture du coup.
"""
from __future__ import annotations

import numpy as np
import torch


def joue(policy, jours, Xn, R, D, S, fin_valide, cfg, device, explore,
         gen=None, collecte=False, rangs=None, marge=None, prix=None, atr=None):
    import jeu_kairos as J
    from jeu_compte import taille_position

    jours = np.asarray(jours, dtype=np.int64).reshape(-1, 2)
    G, H, L = len(jours), int(cfg.horizon_max), int(cfg.lookback)
    P, K = int(cfg.positions_max), len(cfg.sl_atr)
    if P < 1:
        raise ValueError("positions_max doit etre positif")
    scores = np.zeros(G, dtype=np.float64)
    trans = [[] for _ in range(G)] if collecte else None
    if not G:
        return scores, [], trans
    prix = prix if prix is not None else getattr(marge, "prix", None)
    atr = atr if atr is not None else getattr(marge, "atr", None)
    if (prix is None) != (atr is None):
        raise ValueError("prix et atr doivent etre fournis ensemble")

    # Une chaine par segment en evaluation. Le train tire des episodes
    # independants, sans pretendre les rejouer sur un compte commun.
    chaines = []
    for g in np.argsort(jours[:, 0], kind="stable"):
        if (collecte or not chaines
                or int(J._lim(fin_valide, int(jours[g, 0]))) !=
                   int(J._lim(fin_valide, int(jours[chaines[-1][-1], 0])))):
            chaines.append([int(g)])
        else:
            if jours[g, 0] < jours[chaines[-1][-1], 1]:
                raise ValueError("journees chevauchantes dans un meme segment")
            chaines[-1].append(int(g))
    B = len(chaines)
    index = np.zeros(B, dtype=np.int64)
    t = np.array([max(jours[ch[0], 0], L - 1) for ch in chaines])
    jetons = np.full(B, int(cfg.jetons))
    vus = np.zeros(B, dtype=np.float64)
    soldes = np.full(B, float(cfg.capital))
    pics = np.full(B, float(cfg.capital))
    dd_debut_jour = np.zeros(B, dtype=np.float64)
    # Memoire strictement causale du calibrage : elle ne change qu'a la
    # cloture d'un coup. Voir aussi la persistance homologue dans le live.
    memoire_r = np.zeros(B, dtype=np.float64)
    surprises = np.zeros(B, dtype=np.float64)
    series_pertes = np.zeros(B, dtype=np.float64)
    actifs = np.ones(B, dtype=bool)
    ouverts = [[] for _ in range(B)]  # (cloture, R, PnL$, marge$)
    coups = []
    dates_possibles = None
    if not collecte and rangs is not None and cfg.porte_rang_expert > 0:
        dates_possibles = np.flatnonzero(np.any(rangs >= cfg.porte_rang_expert, axis=1))

    def realise(s, positions):
        """Comptabilise les sorties sans jamais laisser une equity negative."""
        if positions:
            soldes[s] = max(0.0, soldes[s] + sum(p[2] for p in positions))
            pics[s] = max(pics[s], soldes[s])
            for p in positions:
                # p = (cloture, R realise, PnL, marge, valeur predite a l'entree)
                r_reel, v_prevue = float(p[1]), float(p[4])
                memoire_r[s] = 0.75 * memoire_r[s] + 0.25 * np.clip(r_reel, -1.0, 1.0)
                surprise = np.clip(v_prevue - r_reel, 0.0, 1.0)
                surprises[s] = 0.80 * surprises[s] + 0.20 * surprise
                series_pertes[s] = (min(1.0, 0.75 * series_pertes[s] + 0.25)
                                    if r_reel < 0.0 else 0.50 * series_pertes[s])

    def termine(s):
        actifs[s] = False
        if collecte:
            g = chaines[s][index[s]]
            if trans[g]:
                x = trans[g][-1]
                trans[g][-1] = x[:12] + (True,) + x[13:]

    policy.eval()
    while actifs.any():
        decide = []
        jours_actuels = []
        stops = []
        marges = []
        marges_libres = []
        echelles_exposition = []
        plafonds_lots = []
        for s in np.flatnonzero(actifs):
            g = chaines[s][index[s]]
            # Clotures anterieures a minuit, puis changement de jour, puis
            # clotures du nouveau jour. La vie du jour ne lit pas le futur.
            while t[s] >= jours[g, 1]:
                limite = int(jours[g, 1])
                faits = [p for p in ouverts[s] if p[0] < limite]
                realise(s, faits)
                ouverts[s] = [p for p in ouverts[s] if p[0] >= limite]
                if index[s] + 1 >= len(chaines[s]):
                    termine(s)
                    break
                index[s] += 1
                g = chaines[s][index[s]]
                # Un trou entre deux parties ne reporte pas ses pertes sur
                # la vie de la prochaine journee jouee.
                avant = [p for p in ouverts[s] if p[0] < jours[g, 0]]
                realise(s, avant)
                ouverts[s] = [p for p in ouverts[s] if p[0] >= jours[g, 0]]
                t[s] = max(t[s], int(jours[g, 0]), L - 1)
                vus[s], jetons[s] = 0.0, int(cfg.jetons)
                _scale, _bloque, dd_debut_jour[s] = J.gouverneur_exposition(
                    soldes[s], pics[s], 0.0, cfg)
            if not actifs[s]:
                continue
            faits = [p for p in ouverts[s] if p[0] <= t[s]]
            realise(s, faits)
            vus[s] += sum(p[1] for p in faits)
            ouverts[s] = [p for p in ouverts[s] if p[0] > t[s]]
            fin = int(jours[g, 1])
            if (jetons[s] <= 0 or vus[s] <= -cfg.vie_R or soldes[s] <= 0
                    or t[s] + 1 + H >= int(J._lim(fin_valide, int(t[s])))):
                if collecte:
                    termine(s)
                else:
                    t[s] = fin
                continue
            if len(ouverts[s]) >= P:
                t[s] = min(fin, min(p[0] for p in ouverts[s]))
                continue
            echelle, bloque, _dd = J.gouverneur_exposition(
                soldes[s], pics[s], dd_debut_jour[s], cfg)
            if bloque:
                t[s] = fin
                continue
            if dates_possibles is not None:
                q = np.searchsorted(dates_possibles, t[s])
                if q == len(dates_possibles) or dates_possibles[q] >= fin:
                    t[s] = fin
                    continue
                suivant = int(dates_possibles[q])
                # Une resolution precedant le prochain signal doit etre
                # comptabilisee avant de prendre une nouvelle decision.
                if suivant > t[s]:
                    t[s] = min(suivant, min((p[0] for p in ouverts[s]), default=suivant))
                    continue
            libre = soldes[s] - sum(p[3] for p in ouverts[s])
            ms = np.zeros(K)
            permis = np.ones(K, dtype=bool)
            if prix is not None:
                for j, sl in enumerate(cfg.sl_atr):
                    distance = float(atr[t[s]]) * sl
                    lot = taille_position(soldes[s], distance, float(prix[t[s]]), cfg,
                                          marge_libre=libre, limite_risque=False)
                    permis[j] = lot > 0
                    ms[j] = lot * cfg.contrat * float(prix[t[s]]) / cfg.levier
            elif marge is not None:
                ms = np.asarray(marge[t[s]]) * soldes[s]
                permis &= ms <= libre
            decide.append(int(s))
            jours_actuels.append(g)
            stops.append(permis)
            marges.append(ms)
            marges_libres.append(float(libre))
            echelles_exposition.append(float(echelle))
            # La sixieme tete choisit une part de l'equite que le stop peut
            # perdre. Chaque case est bornee a la fois par cette enveloppe,
            # par la marge libre et par les limites du courtier. Ainsi un
            # stop ne peut jamais rendre le compte negatif en simulation.
            fractions = np.asarray(
                getattr(cfg, "niveaux_risque_pct", (100.0,))
                if getattr(cfg, "apprendre_risque", False) else (np.inf,), dtype=float) / 100.0
            plafonds = np.zeros((K, len(fractions)), dtype=float)
            for z, sl in enumerate(cfg.sl_atr):
                distance = float(atr[t[s]]) * sl
                marge_max = taille_position(soldes[s], distance, float(prix[t[s]]), cfg,
                                             marge_libre=libre, limite_risque=False)
                par_risque = (soldes[s] * fractions * echelle /
                               max(distance * cfg.contrat, 1e-12))
                maximum = np.minimum(marge_max, par_risque)
                plafonds[z] = np.floor(maximum / cfg.pas_lot + 1e-10) * cfg.pas_lot
            permis &= plafonds[:, -1] + 1e-12 >= cfg.lot_min
            plafonds_lots.append(plafonds)
        if not decide:
            continue
        ix = np.asarray(decide)
        gg = np.asarray(jours_actuels)
        tt = t[ix].copy()
        masques = np.asarray(stops)
        echelles = np.asarray(echelles_exposition, dtype=np.float64)
        et = J.etat_jeu(jetons[ix], vus[ix], jours[gg, 1] - tt, cfg,
                         np.array([len(ouverts[s]) / P for s in ix]),
                         equite=soldes[ix], pic_equite=pics[ix],
                         marge_libre=np.asarray([marges_libres[q] for q in range(len(ix))]),
                         calibration=np.column_stack([memoire_r[ix], surprises[ix],
                                                      series_pertes[ix]]))
        pa, pv = J.portes(masques.any(axis=1), tt, rangs, cfg)
        with torch.no_grad():
            x = torch.from_numpy(J.observations(Xn, tt, et, L)).to(device)
            sortie = policy.jeu(x)
            le, v, ltp, lsl = sortie[:4]
            llo = sortie[4] if len(sortie) >= 5 else None
            lri = sortie[5] if len(sortie) >= 6 else None
            lal = sortie[6] if len(sortie) >= 7 else None
            lco = sortie[7] if len(sortie) >= 8 else None
            le = J._masque_logits(le, torch.as_tensor(pa, device=device),
                                 torch.as_tensor(pv, device=device))
            a = J._choix(le, explore, gen)
            sens = (a == J.VENDRE).long()
            ar = torch.arange(len(ix), device=device)
            lt = ltp[ar, sens]
            ls = lsl[ar, sens].masked_fill(~torch.as_tensor(masques, device=device), J._NEG)
            i, j = J._choix(lt, explore, gen), J._choix(ls, explore, gen)
            ll = llo[ar, sens] if llo is not None else None
            lr = lri[ar, sens] if lri is not None else None
            la = lal[ar, sens] if lal is not None else None
            u = J._choix(la, explore, gen) if la is not None else torch.zeros_like(i)
            lc = lco[ar, sens] if lco is not None else None
            z = J._choix(lc, explore, gen) if lc is not None else torch.zeros_like(i)
            allocation = (np.asarray(getattr(cfg, "niveaux_allocation_pct", (100.0,)), dtype=float) /
                          100.0)
            budget = allocation[u.cpu().numpy()] if la is not None else np.ones(len(ix))
            confiance = (np.asarray(getattr(cfg, "niveaux_confiance_pct", (100.0,)), dtype=float) /
                          100.0)
            budget *= confiance[z.cpu().numpy()] if lc is not None else 1.0
            masque_risque = np.ones((len(ix), 1), dtype=bool)
            if lr is not None:
                plafonds_np = np.asarray(plafonds_lots)
                masque_risque = (plafonds_np[np.arange(len(ix)), j.cpu().numpy()] * budget[:, None]
                                  >= cfg.lot_min)
                lr = lr.masked_fill(~torch.as_tensor(masque_risque, device=device), J._NEG)
                h = J._choix(lr, explore, gen)
            else:
                h = torch.zeros_like(i)
            maximum = (np.asarray(plafonds_lots)[np.arange(len(ix)), j.cpu().numpy(),
                                                  h.cpu().numpy()] * budget)
            niveaux = int(cfg.niveaux_lot)
            grille = np.zeros((len(ix), niveaux), dtype=float)
            valide = maximum + 1e-12 >= cfg.lot_min
            # Un budget choisi par les deux tetes peut etre legal en theorie
            # tout en ne permettant pas le plus petit lot MT5 apres arrondi.
            # Ce n'est pas un trade de taille nulle : c'est une ATTENTE. Le
            # meme cas est refuse par `execute_demo` en live ; le convertir
            # ici avant collecte garde PPO, statistiques et CSV coherents.
            pas_financable = torch.as_tensor(~valide, device=device) & (a != J.ATTENDRE)
            a = a.masked_fill(pas_financable, J.ATTENDRE)
            if valide.any():
                grille[valide] = np.array([np.geomspace(cfg.lot_min, m, niveaux)
                                           for m in maximum[valide]])
                grille[valide] = np.floor(grille[valide] / cfg.pas_lot + 1e-10) * cfg.pas_lot
                grille[valide, 0], grille[valide, -1] = cfg.lot_min, maximum[valide]
            k = J._choix(ll, explore, gen) if ll is not None else torch.zeros_like(i)
            lp = [torch.log_softmax(lg, -1).gather(1, ac[:, None]).squeeze(1).cpu().numpy()
                  for lg, ac in ((le, a), (lt, i), (ls, j))]
            if ll is not None:
                lp.append(torch.log_softmax(ll, -1).gather(1, k[:, None]).squeeze(1).cpu().numpy())
            if lr is not None:
                lp.append(torch.log_softmax(lr, -1).gather(1, h[:, None]).squeeze(1).cpu().numpy())
            if la is not None:
                lp.append(torch.log_softmax(la, -1).gather(1, u[:, None]).squeeze(1).cpu().numpy())
            if lc is not None:
                lp.append(torch.log_softmax(lc, -1).gather(1, z[:, None]).squeeze(1).cpu().numpy())
        a, sens, i, j, k, h, u, z, v = [q.cpu().numpy() for q in (a, sens, i, j, k, h, u, z, v)]
        for q, s in enumerate(ix):
            g, tc = int(gg[q]), int(tt[q])
            r, duree, so = 0.0, 1, 2
            if a[q] != J.ATTENDRE:
                r = float(R[tc, sens[q], i[q], j[q]])
                duree = int(D[tc, sens[q], i[q], j[q]])
                so = int(S[tc, sens[q], i[q], j[q]])
                if not np.isfinite(r):
                    raise ValueError("coup non fini a l'interieur du segment")
                lot = grille[q, k[q]]
                risque = lot * cfg.contrat * float(atr[tc]) * cfg.sl_atr[j[q]]
                nominal = soldes[s] * cfg.risque_pct / 100.0
                # Un gap ne peut pas faire descendre la simulation sous zero.
                # Le plafond est l'equite disponible, non une limite fixe par trade.
                pnl = max(r * risque, -soldes[s])
                r_pondere = pnl / max(nominal, 1e-12)
                risque_pct = float(cfg.niveaux_risque_pct[h[q]]) if lr is not None else float("nan")
                marge_requise = lot * cfg.contrat * float(prix[tc]) / cfg.levier
                # Ces champs supplementaires servent au CSV d'audit : chaque
                # volume est donc verifiable avec son equite et sa marge au
                # moment exact de l'entree.
                coups.append((g, tc, int(sens[q]), int(i[q]), int(j[q]), r_pondere,
                              duree, so, float(lot), risque_pct, float(pnl),
                              float(soldes[s]), float(marges_libres[q]), float(marge_requise),
                               float(prix[tc]), float(atr[tc]) * cfg.sl_atr[j[q]],
                               float(maximum[q]), float(echelles[q]),
                               float((pics[s] - soldes[s]) / max(pics[s], 1e-12)),
                               float(budget[q]),
                               float(confiance[z[q]] if lc is not None else 1.0)))
                ouverts[s].append((tc + duree, r_pondere, pnl,
                                   marge_requise, float(v[q])))
                scores[g] += r_pondere
                jetons[s] -= 1
            pas = duree if P == 1 else 1
            t[s] = min(tc + pas, int(jours[g, 1]))
            if collecte:
                if trans[g]:
                    precedent = trans[g][-1]
                    trans[g][-1] = precedent[:11] + (tc - precedent[0], precedent[12]) + precedent[13:]
                code = int(pa[q]) + 2 * int(pv[q])
                code += sum(int(ok) << (z + 2) for z, ok in enumerate(masques[q]))
                code += sum(int(ok) << (z + 2 + len(cfg.sl_atr))
                            for z, ok in enumerate(masque_risque[q]))
                trans[g].append((tc, et[q], code, int(a[q]), int(i[q]), int(j[q]),
                                 float(lp[0][q]), float(lp[1][q]), float(lp[2][q]),
                                 float(v[q]), r_pondere if a[q] != J.ATTENDRE else 0.0, pas, False,
                                 int(k[q]), float(lp[3][q]) if len(lp) >= 4 else 0.0,
                                 int(h[q]), float(lp[4][q]) if len(lp) >= 5 else 0.0,
                                 int(u[q]), float(lp[5][q]) if len(lp) >= 6 else 0.0,
                                 int(z[q]), float(lp[6][q]) if len(lp) >= 7 else 0.0))
    return scores, coups, trans
