# -*- coding: utf-8 -*-
"""Moteur live KAIROS M5 pour le GUI existant.

Il charge le checkpoint, le pipeline (normalisations + experts) et les memes
features Binance/Coinbase que le jeu. L'execution est refusee hors compte demo.
"""
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dataclasses import fields

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch

import flux_live
import jeu_kairos as J
import prepare_btc_h1_binance as PH
import prepare_btc_m5 as PM5
from jeu_artifacts import charge_modeles_expert, normalise_config_historique

MAGIC = 909510


def _json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "KAIROS-M5-Demo/1.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode("utf-8"))


def _coinbase(debut, fin):
    # L'API publique retourne [time, low, high, open, close, volume].
    def utc(x):
        x = pd.Timestamp(x)
        return (x.tz_localize("UTC") if x.tzinfo is None else x.tz_convert("UTC")).isoformat()
    q = urllib.parse.urlencode({"granularity": 300, "start": utc(debut), "end": utc(fin)})
    rows = _json("https://api.exchange.coinbase.com/products/BTC-USD/candles?" + q)
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=["unix", "low", "high", "open", "close", "volume"])
    d["time"] = pd.to_datetime(d["unix"], unit="s").dt.tz_localize(None)
    d["close"] = d["close"].astype(float)
    # `prepare_btc_m5.features` attend le nom employe par le cache historique.
    return d[["time", "close"]].rename(columns={"close": "cb_close"})


def _funding():
    rows = _json("https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1000")
    d = pd.DataFrame(rows)
    if d.empty:
        return pd.Series(dtype=float)
    d["time"] = pd.to_datetime(d["fundingTime"].astype("int64"), unit="ms")
    return d.drop_duplicates("time").set_index("time")["fundingRate"].astype(float).sort_index()


def _cfg(brut):
    brut = normalise_config_historique(brut)
    noms = {f.name for f in fields(J.JeuConfig)}
    return J.JeuConfig(**{k: tuple(v) if isinstance(v, list) else v
                          for k, v in brut.items() if k in noms})


class _Sortie:
    """Le resultat total d'une position fermee, au format d'un deal MT5."""

    def __init__(self, position_id, pnl):
        self.position_id, self.profit = position_id, pnl
        self.swap = self.commission = self.fee = 0.0


class MoteurKairosM5:
    def __init__(self, live_cfg):
        self.live_cfg = live_cfg
        self.prefixe = getattr(live_cfg, "kairos_prefixe", "kairos_jeu_m5_09")
        self.bloc = int(getattr(live_cfg, "kairos_bloc", 10))
        # Le live joue les memes parties quotidiennes que le jeu : dix
        # jetons, vie en R et remise a zero a chaque nouvelle journee UTC.
        # Ce n'est volontairement plus une option de l'interface.
        self.limite_jetons = True
        self.hist = flux_live.Historique(n=16_000)
        self.dernier_t = None
        self.jetons, self.score, self.jour = 10, 0.0, None
        self.pic_equite = None
        self.dd_debut_jour = None
        self.gouverneur_bloque = False
        # Le live ne doit jamais basculer silencieusement vers l'ancien
        # artefact, qui etait derive d'un seul fold. Seul l'ensemble des dix
        # meilleurs folds est autorise pour un compte demo.
        self.deploy_checkpoint = f"deploy_{self.prefixe}_ensemble_validation_ddsafe.pth"
        self.deploy_pipeline = f"pipeline_{self.prefixe}_deploy_ensemble_validation_ddsafe.json"
        self.utilise_deploy = (os.path.exists(self.deploy_checkpoint) and
                               os.path.exists(self.deploy_pipeline))
        if not self.utilise_deploy:
            raise FileNotFoundError(
                "DEPLOY ENSEMBLE absent : lancer `python jeu_kairos.py "
                "--rebuild-deploy-ensemble` avant le live.")
        self.etat_path = f"etat_{self.prefixe}_deploy_ensemble_validation_ddsafe_demo.json"
        self.risques_positions, self.valeurs_positions, self.deals_comptes = {}, {}, set()
        self.memoire_r, self.surprise, self.serie_pertes = 0.0, 0.0, 0.0
        # LA SERIE NOIRE DU LIVE, mesuree sur les vrais trades : la serie de
        # pertes en cours, la plus longue, la pire perte reelle rapportee a
        # la perte prevue au stop. Voir `serie_noire_live`.
        self.pertes_prevues = {}
        self.serie_live, self.serie_live_max, self.k_live = 0, 0, 1.0
        # LA SORTIE EN DEUX TEMPS : par position, l'objectif proche, le stop
        # a l'entree (spread compris) et si la moitie est deja sortie ; et le
        # resultat cumule des sorties partielles d'une position encore ouverte.
        self.deux_temps, self.pnl_positions = {}, {}
        self._charge_etat()
        self._charge()

    def _charge_etat(self):
        """Retrouve les garde-fous apres un redemarrage de la GUI."""
        try:
            with open(self.etat_path, encoding="utf-8") as fh:
                e = json.load(fh)
            self.jour = datetime.fromisoformat(e["jour"]).date() if e.get("jour") else None
            self.jetons = int(e.get("jetons", self.jetons))
            self.score = float(e.get("score", self.score))
            self.pic_equite = float(e["pic_equite"]) if e.get("pic_equite") is not None else None
            self.dd_debut_jour = (float(e["dd_debut_jour"])
                                   if e.get("dd_debut_jour") is not None else None)
            self.gouverneur_bloque = bool(e.get("gouverneur_bloque", False))
            self.risques_positions = {str(k): float(v) for k, v in e.get("risques_positions", {}).items()}
            self.valeurs_positions = {str(k): float(v) for k, v in e.get("valeurs_positions", {}).items()}
            self.memoire_r = float(e.get("memoire_r", self.memoire_r))
            self.surprise = float(e.get("surprise", self.surprise))
            self.serie_pertes = float(e.get("serie_pertes", self.serie_pertes))
            self.deals_comptes = {int(x) for x in e.get("deals_comptes", [])}
            self.pertes_prevues = {str(k): float(v) for k, v in e.get("pertes_prevues", {}).items()}
            self.serie_live = int(e.get("serie_live", 0))
            self.serie_live_max = int(e.get("serie_live_max", 0))
            self.k_live = float(e.get("k_live", 1.0))
            self.deux_temps = {str(k): dict(v) for k, v in e.get("deux_temps", {}).items()}
            self.pnl_positions = {str(k): float(v) for k, v in e.get("pnl_positions", {}).items()}
        except (OSError, ValueError, TypeError, KeyError):
            pass

    def _sauve_etat(self):
        e = {"jour": self.jour.isoformat() if self.jour else None,
             "jetons": self.jetons, "score": self.score,
             "pic_equite": self.pic_equite,
             "dd_debut_jour": self.dd_debut_jour,
             "gouverneur_bloque": self.gouverneur_bloque,
             "risques_positions": self.risques_positions,
             "valeurs_positions": self.valeurs_positions,
             "memoire_r": self.memoire_r, "surprise": self.surprise,
             "serie_pertes": self.serie_pertes,
             "deals_comptes": sorted(self.deals_comptes)[-200:],
             "pertes_prevues": self.pertes_prevues,
             "serie_live": self.serie_live, "serie_live_max": self.serie_live_max,
             "k_live": self.k_live,
             "deux_temps": self.deux_temps, "pnl_positions": self.pnl_positions}
        with open(self.etat_path, "w", encoding="utf-8") as fh:
            json.dump(e, fh, indent=1)

    def reconcilie(self):
        """Convertit les sorties MT5 du moteur en R, comme le score du jeu."""
        depuis = datetime.now() - timedelta(days=3)
        deals = mt5.history_deals_get(depuis, datetime.now()) or ()
        change = False
        for d in deals:
            ticket = int(d.ticket)
            if (int(d.magic) != MAGIC or int(d.entry) != mt5.DEAL_ENTRY_OUT
                    or ticket in self.deals_comptes):
                continue
            self.deals_comptes.add(ticket)
            # UNE POSITION PEUT SORTIR EN DEUX FOIS (sortie en deux temps) :
            # on cumule ses sorties, et on ne la compte qu'une fois fermee.
            pid = str(int(d.position_id))
            self.pnl_positions[pid] = (self.pnl_positions.get(pid, 0.0) + float(d.profit)
                                       + float(d.swap) + float(d.commission) + float(d.fee))
            change = True
        encore = {str(int(q.ticket)) for q in (mt5.positions_get(symbol=self.live_cfg.symbol) or ())}
        for pid in [x for x in self.pnl_positions if x not in encore]:
            pnl_total = self.pnl_positions.pop(pid)
            self.deux_temps.pop(pid, None)
            d = _Sortie(int(pid), pnl_total)
            change = True
            reference_r = self.risques_positions.pop(str(int(d.position_id)), 0.0)
            valeur_prevue = self.valeurs_positions.pop(str(int(d.position_id)), 0.0)
            prevue = self.pertes_prevues.pop(str(int(d.position_id)), 0.0)
            if prevue > 0:
                # LA SERIE NOIRE DU LIVE, sur le resultat reel du trade.
                pnl_s = float(d.profit) + float(d.swap) + float(d.commission) + float(d.fee)
                self.serie_live = self.serie_live + 1 if pnl_s < 0 else 0
                self.serie_live_max = max(self.serie_live_max, self.serie_live)
                if pnl_s < 0:
                    self.k_live = max(self.k_live, -pnl_s / prevue)
            if reference_r > 0:
                pnl = float(d.profit) + float(d.swap) + float(d.commission) + float(d.fee)
                r_reel = pnl / reference_r
                self.score += r_reel
                self.memoire_r = 0.75 * self.memoire_r + 0.25 * float(np.clip(r_reel, -1.0, 1.0))
                self.surprise = 0.80 * self.surprise + 0.20 * float(np.clip(valeur_prevue - r_reel, 0.0, 1.0))
                self.serie_pertes = (min(1.0, 0.75 * self.serie_pertes + 0.25)
                                      if r_reel < 0.0 else 0.50 * self.serie_pertes)
                print(f"[KAIROS M5] sortie MT5 {pnl:+.2f} $ = {r_reel:+.2f} R ; score jour {self.score:+.2f} R")
        if change:
            self._sauve_etat()

    def _charge(self):
        ck_path = self.deploy_checkpoint
        pipe_path = self.deploy_pipeline
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        self.cfg = _cfg(ck["config"])
        with open(pipe_path, encoding="utf-8") as fh:
            self.pipe = json.load(fh)
        self.experts = charge_modeles_expert(self.pipe)
        self.policy = J.PolitiqueJeu(self.cfg).eval()
        self.policy.load_state_dict(ck["modele"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        print(f"[KAIROS M5] deploy ensemble des 10 folds, epoch {ck.get('epoch')} charge.")

    def _donnees(self):
        brut = self.hist.actualise().copy()
        # Coinbase limite cette granularite a 300 bougies par requete.
        cb = _coinbase(brut["time"].iloc[-295], brut["time"].iloc[-1] + pd.Timedelta(minutes=5))
        f = PM5.features(brut, cb, _funding(), PH.profil_spread())
        f[PM5.FEATURE_COLS_M5] = f[PM5.FEATURE_COLS_M5].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return f

    def decision(self):
        d = self._donnees()
        t = len(d) - 1
        horodatage = pd.Timestamp(d["time"].iloc[t])
        if horodatage == self.dernier_t:
            return None
        self.dernier_t = horodatage
        if self.jour != horodatage.date():
            self.jour, self.jetons, self.score = horodatage.date(), self.cfg.jetons, 0.0
            self.dd_debut_jour, self.gouverneur_bloque = None, False
        # Dans le jeu, une place reste occupee jusqu'a la cloture du coup :
        # avec positions_max=1, aucune decision n'est donc prise entre temps.
        positions = mt5.positions_get(symbol=self.live_cfg.symbol) or ()
        ouvertes = [p for p in positions if int(p.magic) == MAGIC]
        if len(ouvertes) >= int(self.cfg.positions_max):
            return None
        # LE JEU HISTORIQUE (version 1) fermait la journee : un coup devait se
        # resoudre avant minuit UTC. LE JEU VERSION 2 garde les positions
        # d'un jour a l'autre et permet d'entrer jusqu'a la derniere bougie
        # du jour (`jeu_rollout.py`) : 25 % des coups de validation du run
        # m5_20 entraient dans les 8 h avant minuit. Corrige le 2026-10-03 :
        # le live les bloquait encore.
        if int(getattr(self.cfg, "jeu_version", 1)) < 2:
            barre_jour = horodatage.hour * (60 // self.cfg.minutes_par_barre) + \
                          horodatage.minute // self.cfg.minutes_par_barre
            reste_barres = self.cfg.barres_par_partie - barre_jour
            if reste_barres <= self.cfg.horizon_max + 1:
                return None
        st = self.pipe["normalisation_marche"]
        x = d[self.pipe["colonnes"]].to_numpy(np.float32)
        xn = np.clip((x - np.asarray(st["mean"], np.float32)) /
                     (np.asarray(st["std"], np.float32) + 1e-8), -5, 5).astype(np.float32)
        pred = np.column_stack([m.predict(xn) for m in self.experts]).astype(np.float32)
        fe = J.features_expert(pred, self.pipe["fenetre_expert"])
        se = self.pipe["normalisation_expert"]
        fen = np.clip((fe - np.asarray(se["mean"])) / (np.asarray(se["std"]) + 1e-8), -5, 5).astype(np.float32)
        xk = np.concatenate([xn, fen[:, :self.cfg.n_expert]], axis=1)
        ai = mt5.account_info()
        equite = float(ai.equity) if ai is not None else float(self.cfg.capital)
        libre = float(ai.margin_free) if ai is not None else equite
        self.pic_equite = max(equite, self.pic_equite if self.pic_equite is not None else equite)
        dd_courant = (self.pic_equite - max(equite, 0.0)) / max(self.pic_equite, 1e-12)
        if self.dd_debut_jour is None:
            self.dd_debut_jour = float(dd_courant)
        echelle, bloque, dd_courant = J.gouverneur_exposition(
            equite, self.pic_equite, self.dd_debut_jour, self.cfg)
        if bloque and not self.gouverneur_bloque:
            self.gouverneur_bloque = True
            self._sauve_etat()
            print(f"[KAIROS M5] gouverneur : nouvelles entrees bloquees pour ce jour "
                  f"(drawdown {100 * dd_courant:.1f} %)")
        if self.gouverneur_bloque:
            return None
        et = J.etat_jeu(np.array([self.jetons]), np.array([self.score]),
                         np.array([reste_barres]), self.cfg,
                         equite=np.array([equite]), pic_equite=np.array([self.pic_equite]),
                         marge_libre=np.array([libre]),
                         calibration=np.array([[self.memoire_r, self.surprise,
                                                self.serie_pertes]], np.float32))
        ob = J.observations(xk, np.array([t]), et, self.cfg.lookback)
        # Les predictions de l'expert sont a la fois des entrees du reseau et
        # la meme porte par sens qu'en validation. Le live ne peut donc pas
        # ouvrir dans une zone que le jeu avait rendue inaccessible.
        peut_jouer = self.jetons > 0 and self.score > -self.cfg.vie_R
        achat_ok = peut_jouer and fe[t, 2] >= self.cfg.porte_rang_expert
        vente_ok = peut_jouer and fe[t, 3] >= self.cfg.porte_rang_expert
        with torch.no_grad():
            sortie = self.policy.jeu(torch.from_numpy(ob).to(self.device))
            le, valeur, ltp, lsl = sortie[:4]
            llo = sortie[4] if len(sortie) >= 5 else None
            lri = sortie[5] if len(sortie) >= 6 else None
            lal = sortie[6] if len(sortie) >= 7 else None
            lco = sortie[7] if len(sortie) >= 8 else None
            les = sortie[8] if len(sortie) >= 9 else None
        le = J._masque_logits(le, torch.tensor([achat_ok], device=self.device),
                               torch.tensor([vente_ok], device=self.device))
        a = int(le[0].argmax().cpu())
        cloture = horodatage + pd.Timedelta(minutes=self.cfg.minutes_par_barre)
        cloture_mt5 = cloture.tz_localize("UTC").tz_convert(
            ZoneInfo(getattr(self.live_cfg, "kairos_tz_mt5", "Europe/Helsinki")))
        if a == 2:
            print(f"[KAIROS M5] HOLD — bougie MT5 {cloture_mt5:%H:%M} "
                  f"(cloturee, ouverte {cloture_mt5 - pd.Timedelta(minutes=self.cfg.minutes_par_barre):%H:%M}), "
                  f"rang {fe[t,2]:.3f}/{fe[t,3]:.3f}")
            return None
        sens = 0 if a == 0 else 1
        tp = int(ltp[0, sens].argmax().cpu())
        sl = int(lsl[0, sens].argmax().cpu())
        lot_niveau = int(llo[0, sens].argmax().cpu()) if llo is not None else 0
        risque_niveau = int(lri[0, sens].argmax().cpu()) if lri is not None else None
        allocation_niveau = int(lal[0, sens].argmax().cpu()) if lal is not None else None
        confiance_niveau = int(lco[0, sens].argmax().cpu()) if lco is not None else None
        # LE DEJA-VU ET LA METEO : les tetes de mise les ont deja lus dans
        # `jeu` ; ils sont affiches pour suivre ce que voit le modele.
        if getattr(self.policy, "apprendre_dejavu_meteo", False):
            vu, met = (float(q[0].cpu()) for q in self.policy.dernieres_previsions)
            print(f"[KAIROS M5] deja-vu {vu:+.2f} (log de l'erreur, plus haut = plus nouveau)  |  "
                  f"meteo {np.exp(met):.0f} bps d'amplitude prevue sur {self.cfg.horizon_max} bougies")
        # LA MISE DE BASE de la tete d'esperance, comme le jeu.
        base = None
        if les is not None:
            pe = les[0, sens, tp, sl].float().cpu().numpy()
            base = float(J.fraction_esperance(pe[0], pe[1], self.cfg))
        # Le jeu applique ce plancher avant de construire TP et SL.
        atr = float(J.atr_effectif(np.array([d["atr_14"].iloc[t]]),
                                   np.array([d["close"].iloc[t]]), self.cfg)[0])
        return {"time": horodatage, "side": sens, "tp": tp, "sl": sl,
                "atr": atr, "lot_niveau": lot_niveau, "risque_niveau": risque_niveau,
                "allocation_niveau": allocation_niveau,
                "confiance_niveau": confiance_niveau, "valeur": float(valeur[0].cpu()),
                "mise_base": base,
                "gouverneur_echelle": float(echelle), "drawdown": float(dd_courant)}

    def execute_demo(self, signal):
        ai = mt5.account_info()
        if ai is None or ai.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise RuntimeError("KAIROS M5 refuse tout compte non-demo")
        symbol = self.live_cfg.symbol
        mt5.symbol_select(symbol, True)
        if mt5.positions_get(symbol=symbol):
            print("[KAIROS M5] position deja ouverte : signal ignore")
            return
        tick, info = mt5.symbol_info_tick(symbol), mt5.symbol_info(symbol)
        if tick is None or info is None:
            raise RuntimeError("BTCUSD indisponible dans MT5")
        buy = signal["side"] == 0
        price = tick.ask if buy else tick.bid
        sl_dist = self.cfg.sl_atr[signal["sl"]] * signal["atr"]
        tp_dist = self.cfg.tp_atr[signal["tp"]] * signal["atr"]
        sl = price - sl_dist if buy else price + sl_dist
        tp = price + tp_dist if buy else price - tp_dist
        ordre_type = mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL
        perte_lot = abs(mt5.order_calc_profit(ordre_type, symbol, 1.0, price, sl) or 0.0)
        marge_lot = float(mt5.order_calc_margin(ordre_type, symbol, 1.0, price) or 0.0)
        if perte_lot <= 0 or marge_lot <= 0:
            print("[KAIROS M5] calcul MT5 du risque ou de la marge indisponible : ordre refuse")
            return
        plaf_marge = max(float(ai.margin_free), 0.0) / marge_lot
        plaf = min(plaf_marge, float(info.volume_max), float(self.cfg.lot_max))
        if signal["risque_niveau"] is not None:
            pct = float(self.cfg.niveaux_risque_pct[signal["risque_niveau"]]) / 100.0
            allocation = (float(self.cfg.niveaux_allocation_pct[signal["allocation_niveau"]]) / 100.0
                          if signal["allocation_niveau"] is not None else 1.0)
            confiance = (float(self.cfg.niveaux_confiance_pct[signal["confiance_niveau"]]) / 100.0
                          if signal.get("confiance_niveau") is not None else 1.0)
            if signal.get("mise_base") is not None:
                # avec la tete d'esperance, l'allocation et la confiance
                # multiplient la mise de base : la tete de risque seule plafonne
                allocation = confiance = 1.0
            plaf = min(plaf, float(ai.equity) * pct * allocation * confiance *
                       signal["gouverneur_echelle"] / perte_lot)
        # LA SERIE NOIRE, la meme regle que le jeu (`plafond_serie_noire`),
        # avec la serie et la perte reelle les plus prudentes entre celles
        # de l'entrainement et celles du live. Voir `serie_noire_live`.
        plaf = min(plaf, float(ai.equity) * J.plafond_serie_noire(self.serie_noire_live()) / perte_lot)
        plaf = np.floor(plaf / info.volume_step + 1e-10) * info.volume_step
        # LE LOT MINIMUM DU JEU (0.02 : un coup doit pouvoir etre coupe en
        # deux), jamais sous celui du courtier.
        vmin = max(float(info.volume_min), float(getattr(self.cfg, "lot_min", info.volume_min)))
        if plaf + 1e-12 < vmin:
            print("[KAIROS M5] lot minimum non financable par la marge ou le risque : ordre refuse")
            return
        # Meme grille geometrique que `jeu_rollout.py` ; sans tete de lot,
        # le comportement historique est le premier niveau, soit le minimum.
        niveaux = int(self.cfg.niveaux_lot)
        grille = np.geomspace(vmin, plaf, niveaux)
        grille = np.floor(grille / info.volume_step + 1e-10) * info.volume_step
        grille[0], grille[-1] = vmin, plaf
        vol = float(grille[min(signal["lot_niveau"], niveaux - 1)])
        if signal.get("mise_base") is not None:
            # LA MISE DE BASE x les multiplicateurs, comme `jeu_rollout.py`.
            mult = (float(self.cfg.multiplicateurs_lot[signal["lot_niveau"]])
                    * float(self.cfg.multiplicateurs_allocation[signal["allocation_niveau"]])
                    * float(self.cfg.multiplicateurs_confiance[signal["confiance_niveau"]]))
            voulu = float(ai.equity) * signal["mise_base"] * mult / perte_lot
            voulu = np.floor(voulu / info.volume_step + 1e-10) * info.volume_step
            vol = float(min(plaf, max(vmin, voulu)))
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(vol),
               "type": ordre_type, "price": price,
               "sl": sl, "tp": tp, "deviation": 50, "magic": MAGIC,
               "comment": "KAIROS_M5_DEMO", "type_filling": mt5.ORDER_FILLING_IOC}
        r = mt5.order_send(req)
        if r is None or r.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[KAIROS M5] ordre refuse: {None if r is None else r.retcode}")
            return
        self.jetons -= 1
        perte_prevue = float(vol) * perte_lot
        # Le ticket de position est la cle stable qui relie une future sortie
        # MT5 a son risque initial. Le score est donc en R, comme dans le jeu.
        positions = mt5.positions_get(symbol=symbol) or ()
        ours = [p for p in positions if int(p.magic) == MAGIC]
        if ours:
            # Dans le jeu, 1 R reste la reference `risque_pct` de l'equite,
            # meme lorsque la tete de risque a choisi davantage ou moins.
            self.risques_positions[str(int(ours[-1].ticket))] = (
                float(ai.equity) * self.cfg.risque_pct / 100.0)
            self.valeurs_positions[str(int(ours[-1].ticket))] = float(signal.get("valeur", 0.0))
            self.pertes_prevues[str(int(ours[-1].ticket))] = perte_prevue
            # LA SORTIE EN DEUX TEMPS, comme le jeu (`sortie_deux_temps`) : si
            # l'objectif choisi est plus loin que `tp1_atr`, la moitie sortira
            # a `tp1_atr` et le stop du reste remontera a l'entree, spread et
            # glissement de sortie compris, plus `be_marge_bps`.
            k1 = J.tp1_objectif(self.cfg, signal["tp"])
            if k1 > 0.0:
                p0 = float(ours[-1].price_open)
                sx = float(self.cfg.glissement_sortie_bps) / 1e4
                mbe = float(getattr(self.cfg, "be_marge_bps", 0.0)) / 1e4
                self.deux_temps[str(int(ours[-1].ticket))] = {
                    "achat": bool(buy),
                    "tp1": p0 + k1 * signal["atr"] if buy else p0 - k1 * signal["atr"],
                    "be": p0 / (1 - sx) * (1 + mbe) if buy else p0 / (1 + sx) * (1 - mbe),
                    "fait": False}
        self._sauve_etat()
        print(f"[KAIROS M5] {'BUY' if buy else 'SELL'} {vol:g} BTCUSD, TP {tp_dist:.2f}, "
              f"SL {sl_dist:.2f}, budget "
              f"{(self.cfg.niveaux_allocation_pct[signal['allocation_niveau']] if signal['allocation_niveau'] is not None else 100):g} %, "
              f"thermostat {(self.cfg.niveaux_confiance_pct[signal['confiance_niveau']] if signal.get('confiance_niveau') is not None else 100):g} %, "
              f"jetons restants {self.jetons}")

    def serie_noire_live(self):
        """La configuration du jeu avec la serie noire la plus prudente entre
        l'entrainement (N et k du modele) et le live (la plus longue serie de
        pertes et la pire perte reelle des vrais trades). Elle ne peut que
        se resserrer : le live n'assouplit jamais la regle apprise."""
        n = max(int(self.cfg.serie_noire_n), int(self.serie_live_max))
        k = max(float(self.cfg.serie_noire_k), float(self.k_live))
        return J.replace(self.cfg, serie_noire_n=n, serie_noire_k=k)

    def gere_deux_temps(self):
        """Sort la moitie a l'objectif proche et remonte le stop du reste a
        l'entree, spread compris. Voir `sortie_deux_temps`."""
        if not any(not v.get("fait") for v in self.deux_temps.values()):
            return
        symbol = self.live_cfg.symbol
        tick, info = mt5.symbol_info_tick(symbol), mt5.symbol_info(symbol)
        if tick is None or info is None:
            return
        change = False
        for p in mt5.positions_get(symbol=symbol) or ():
            e = self.deux_temps.get(str(int(p.ticket)))
            if int(p.magic) != MAGIC or not e or e.get("fait"):
                continue
            achat = p.type == mt5.POSITION_TYPE_BUY
            prix = tick.bid if achat else tick.ask
            if not (prix >= e["tp1"] if achat else prix <= e["tp1"]):
                continue
            moitie = np.floor(float(p.volume) / 2 / info.volume_step + 1e-9) * info.volume_step
            if moitie + 1e-12 >= float(info.volume_min):
                r = mt5.order_send({
                    "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(moitie),
                    "position": int(p.ticket),
                    "type": mt5.ORDER_TYPE_SELL if achat else mt5.ORDER_TYPE_BUY,
                    "price": prix, "deviation": 50, "magic": MAGIC,
                    "comment": "KAIROS_M5_TP1", "type_filling": mt5.ORDER_FILLING_IOC})
                if r is None or r.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"[KAIROS M5] sortie de la moitie refusee : {None if r is None else r.retcode}")
                    continue
            be = round(float(e["be"]), int(info.digits))
            r2 = mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "symbol": symbol,
                                 "position": int(p.ticket), "sl": be, "tp": float(p.tp),
                                 "magic": MAGIC})
            ok = r2 is not None and r2.retcode == mt5.TRADE_RETCODE_DONE
            e["fait"] = True
            change = True
            print(f"[KAIROS M5] objectif proche : moitie sortie ({moitie:g} lot), stop a l'entree "
                  f"spread compris {be} : " + ("fait" if ok else f"REFUSE ({None if r2 is None else r2.retcode})"))
        if change:
            self._sauve_etat()

    def ferme_si_expiree(self):
        """Reproduit la sortie au temps limite du jeu pour les positions KAIROS.

        Les positions manuelles et celles d'un autre moteur ne sont jamais
        touchees : elles n'ont pas notre magic number.
        """
        limite = self.cfg.horizon_max * self.cfg.minutes_par_barre * 60
        positions = mt5.positions_get(symbol=self.live_cfg.symbol) or ()
        for p in positions:
            if int(p.magic) != MAGIC or time.time() - int(p.time) < limite:
                continue
            tick = mt5.symbol_info_tick(self.live_cfg.symbol)
            if tick is None:
                return
            vente = p.type == mt5.POSITION_TYPE_BUY
            req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": self.live_cfg.symbol,
                   "volume": float(p.volume), "position": int(p.ticket),
                   "type": mt5.ORDER_TYPE_SELL if vente else mt5.ORDER_TYPE_BUY,
                   "price": tick.bid if vente else tick.ask, "deviation": 50,
                   "magic": MAGIC, "comment": "KAIROS_M5_TIME", "type_filling": mt5.ORDER_FILLING_IOC}
            r = mt5.order_send(req)
            print("[KAIROS M5] sortie temps limite : " +
                  ("executee" if r is not None and r.retcode == mt5.TRADE_RETCODE_DONE else "refusee"))


def live_loop(cfg, doit_continuer):
    if not mt5.initialize():
        raise RuntimeError("MT5 indisponible")
    try:
        moteur = MoteurKairosM5(cfg)
        while doit_continuer():
            moteur.reconcilie()
            moteur.gere_deux_temps()
            moteur.ferme_si_expiree()
            signal = moteur.decision()
            if signal is not None:
                moteur.execute_demo(signal)
            time.sleep(10)
    finally:
        mt5.shutdown()
