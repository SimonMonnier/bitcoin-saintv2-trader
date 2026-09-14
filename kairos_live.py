import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from saint_core import (
    MASK_VALUE,
    NORM_STATS_PATH,
    FEATURE_COLS,
    SCALPING_MAX_HOLDING,
    merge_m1_h1,
    load_calib_thresholds,
    decide_avec_barres,
    charge_source_externe,
    SOURCE_EXT_NOM,
    safe_normalize,
    load_norm_stats,
    load_model_norm_stats,
    load_shared_model_norm_stats,
    build_policy,
    get_device,
    build_mask_from_pos_scalar,
    compute_dynamic_volume,
    compute_risk_volume,
    compute_sl_tp,
    compute_entry_atr,
)

# ============================================================
# CONFIG & CONSTANTES
# ============================================================

# Modèles pré-entraînés — best Sortino (comme dans le training)
BEST_MODEL_LONG_PATH = "bestprofit_saintv2_loup_long_wf1_long_wf1.pth"
BEST_MODEL_SHORT_PATH = "bestprofit_saintv2_loup_short_wf1_short_wf1.pth"
# Modèle unifié (entraîné avec side="both") : décide BUY/SELL/HOLD dans un seul fichier
BEST_MODEL_DUEL_PATH = "bestprofit_saintv2_loup_duel_exec5_wf1_both_wf1.pth"

# ============================================================
# MULTI-AGENT : 3 modèles WF tradent en parallèle (comme dans le backtest)
# Chaque modèle a son propre magic MT5 pour identifier ses positions.
# ============================================================
# Lignee BTCUSD, prefixe exec5 (detention de reference 30 barres, sortie par
# le temps retiree). Les fichiers "exec3" sont du BTC mais avec une AUTRE
# observation : leurs poids ne sont pas interchangeables. Ceux sans prefixe
# d'exec sont les modeles OR — 1.96 Mo contre 1.20 Mo, architecture d'avant la
# reduction du bloc SAINTv2. D'ou le nommage explicite plutot qu'un chemin
# generique : un mauvais fichier charge ici trade sans erreur visible.
# wf2/wf3 n'existent pas encore en BTC : leur activation doit echouer bruyamment.
MULTI_AGENT_PATHS: Dict[str, str] = {
    "wf1": "bestprofit_saintv2_loup_duel_exec5_wf1_both_wf1.pth",
    "wf2": "bestprofit_saintv2_loup_duel_exec5_wf2_both_wf2.pth",
    "wf3": "bestprofit_saintv2_loup_duel_exec5_wf3_both_wf3.pth",
}
MULTI_AGENT_MAGICS: Dict[str, int] = {
    "wf1": 424241,
    "wf2": 424242,
    "wf3": 424243,
}


@dataclass
class LiveConfig:
    symbol: str = "BTCUSD"
    timeframe: int = mt5.TIMEFRAME_M1
    htf_timeframe: int = mt5.TIMEFRAME_H1   # identique au training

    lookback: int = 25

    # nombre de bougies pour recalculer les indicateurs
    n_bars_m1: int = 50000
    n_bars_h1: int = 20000

    # config training originale (R:R 1:1.4)
    tp_shrink: float = 1.0  # pas de shrink (formule explicite : atr_tp_mult contient déjà le facteur final)

    # trading (mêmes valeurs que PPOConfig)
    position_size: float = 0.01
    # Taille par le RISQUE (prioritaire sur dynamic_volume / position_size).
    # Meme valeur que training.PPOConfig.risk_per_trade : le modele doit
    # trader le risque sous lequel il a appris.
    risk_volume: bool = True
    risk_per_trade: float = 0.012
    leverage: float = 100.0   # aligné sur training.py (BTCUSD)
    fee_rate: float = 0.0   # ce courtier ne facture pas de commission sur BTCUSD
    atr_sl_mult: float = 2.0     # SL = 2.0 x ATR   — training.PPOConfig.atr_sl_mult
    atr_tp_mult: float = 2.8     # TP = 2.8 x ATR   — R:R 1:1.4, identique au training

    spread_bps: float = 0.0
    slippage_bps: float = 0.0

    # fréquence de décision (en secondes)
    poll_interval: int = 2

    # device
    force_cpu: bool = False

    # mode d’agent :
    #   "both"  : modèle unifié duel (1 seul .pth, décide BUY/SELL/HOLD)
    #   "duel"  : 2 modèles séparés long+short, arbitrage par max(prob)
    #   "long"  : uniquement agent LONG (pas de short)
    #   "short" : uniquement agent SHORT (pas de long)
    side: str = "both"

    # ======= BREAK-EVEN + TRAILING (en ATR) =======
    # INACTIFS : l'appel a update_sl_be_trailing_live est commente dans la
    # boucle (training.use_be_trail = False, et le backtest donne PF 0.98 avec
    # contre 1.70 sans). Ces valeurs ne servent que si on le reactive.
    breakeven_atr_mult: float = 1.0
    trailing_start_atr_mult: float = 1.5
    trailing_dist_atr_mult: float = 1.0

    # ======= DIVERGENCE CONNUE AVEC LE TRAINING =======
    # training.PPOConfig.max_holding_bars = 240 : l'environnement ferme au
    # marche toute position encore ouverte apres 240 minutes, et les 92.3 % de
    # resolution comme le +0.317 ATR/trade en dependent. Le live n'a AUCUN
    # chemin de fermeture au marche (les positions ne sortent que par SL/TP du
    # courtier), donc cette regle n'est pas appliquee ici. A implementer avant
    # tout deploiement reel, sinon la strategie exécutée n'est pas celle qui a
    # ete mesuree.

    # ======= Seuil de confiance pour ouvrir un trade =======
    # CALIBRÉ, plus fixé à la main : le training écrit le seuil réalisant la
    # sélectivité visée dans `<checkpoint>_calib.json`, et c'est ce seuil-là qui
    # a servi à sélectionner le checkpoint. En choisir un autre ici ferait
    # tourner en production une politique différente de celle qui a été mesurée.
    #
    # La règle est : meilleur côté (BUY vs SELL) puis comparaison à la barre.
    # Surtout PAS d'argmax sur les trois actions — une stratégie qui ne trade que
    # 5 % du temps a p(HOLD) majoritaire presque partout, et l'argmax renverrait
    # HOLD en permanence (mesuré : 0 trade en validation, même seuil à zéro).
    #
    # Valeur de repli si le .json est absent (None = refuser de trader).
    min_confidence: Optional[float] = None

    # ======= Volume dynamique selon l'equity du compte =======
    # True  : lot = 0.01 sous 2000$, +0.01 par tranche de 1000$ au-dessus
    #         (cap 100.00 lot)
    # False : utilise cfg.position_size constant (slider GUI)
    dynamic_volume: bool = True
    max_lot: float = 100.0

    # ======= MULTI-AGENT (wf1 + wf2 + wf3 en parallèle) =======
    # True  : charge les checkpoints listés dans active_agents et chaque agent
    #         peut ouvrir SA position indépendamment (max len(active_agents)
    #         positions simultanées, 1 par agent).
    # False : mode single-agent classique selon cfg.side
    multi_agent: bool = True

    # Liste des agents actifs en mode multi_agent.
    # Mettre ["wf1","wf3"] pour 2 agents, None = tous (wf1 + wf2 + wf3).
    # N'activer qu'un agent dont le .pth existe réellement : le chargement
    # échoue volontairement (FileNotFoundError) plutôt que de trader à vide.
    active_agents: Optional[List[str]] = field(default_factory=lambda: ["wf1"])

    # ======= Gestion de marge =======
    # Fraction max de margin_free utilisée par un ordre (0.0–1.0).
    # 0.80 = on n'utilise jamais plus de 80% de la marge libre pour un trade
    # → laisse une marge de sécurité contre les rejects "no money"
    margin_safety: float = 0.80
    # Si vrai, scale down automatiquement le lot quand la marge est insuffisante
    auto_scale_volume_to_margin: bool = True
    # Volume minimum acceptable (sinon ordre annulé)
    min_volume: float = 0.01



# ============================================================
# SEUIL CALIBRÉ
# ============================================================

def seuil_calibre(pth: str):
    """Barres (BUY, SELL) du checkpoint. Source unique : saint_core."""
    return load_calib_thresholds(pth)


def get_calib_threshold(agent_name: str):
    """Barres du checkpoint associé à un agent du mode multi-agent."""
    return seuil_calibre(MULTI_AGENT_PATHS.get(agent_name, BEST_MODEL_DUEL_PATH))


# ============================================================
# DATA LIVE : M1 + H1 => MERGE + FEATURES
# ============================================================

def fetch_ohlc_with_indicators(cfg: LiveConfig) -> pd.DataFrame:
    rates_m1 = mt5.copy_rates_from_pos(
        cfg.symbol, cfg.timeframe, 0, cfg.n_bars_m1
    )
    rates_h1 = mt5.copy_rates_from_pos(
        cfg.symbol, cfg.htf_timeframe, 0, cfg.n_bars_h1
    )

    if rates_m1 is None or rates_h1 is None:
        raise RuntimeError("MT5 n'a renvoyé aucune donnée M1 ou H1 (live).")

    # Argent : MEME source que l'or (MT5), donc meme horodatage et meme fuseau.
    # Plus aucun appel REST, plus de detection de decalage horaire, plus de
    # risque de divergence entre l'historique et le live — tout ce qui rendait
    # le pipeline Binance fragile disparait ici.
    #
    # On ne demande que la fenetre utile : lookback + marge pour les moyennes
    # glissantes des indicateurs.
    t_to = datetime.now() + timedelta(minutes=2)
    t_from = t_to - timedelta(minutes=cfg.lookback + 400)
    feats_ext = charge_source_externe(t_from, t_to)
    if feats_ext is None or len(feats_ext) == 0:
        raise RuntimeError(
            f"Source externe absente : {SOURCE_EXT_NOM}. Deux des dix "
            f"features en dependent, tourner sans elles ferait decider le "
            f"modele sur un vecteur qu'il n'a jamais vu."
        )
    _si = mt5.symbol_info(cfg.symbol)
    point = float(_si.point) if _si is not None else 1.0

    return merge_m1_h1(rates_m1, rates_h1, feats_ext=feats_ext, point=point)


def build_live_obs(
    df_merged: pd.DataFrame,
    stats: Dict[str, np.ndarray],
    cfg: LiveConfig,
    pos: int,
    entry_price: float,
    last_risk_scale: float,
    bars_in_position: int = 0,
) -> Optional[np.ndarray]:
    if len(df_merged) < cfg.lookback + 1:
        return None

    X = df_merged[FEATURE_COLS].values.astype(np.float32)
    X_norm = safe_normalize(X, stats, clip_sigma=5.0)

    base = X_norm[-cfg.lookback:]

    current_price = float(df_merged["close"].iloc[-1]) if len(df_merged) > 0 else 0.0

    # PnL latent en unités d'ATR — IDENTIQUE training/backtest :
    # entry_atr est FIGÉ à l'ouverture (via cache par ticket), pas recalculé.
    if pos != 0 and entry_price > 0.0:
        entry_atr = get_entry_atr_cached(cfg.symbol, df_merged)
        if entry_atr > 1e-8:
            unrealized_atr = float(pos * (current_price - entry_price) / entry_atr)
        else:
            unrealized_atr = 0.0
    else:
        unrealized_atr = 0.0

    # Durée de détention normalisée (même référence que l'env d'entraînement)
    bars_held_norm = float(min(bars_in_position / max(SCALPING_MAX_HOLDING, 1), 3.0))

    pos_feature  = float(pos)
    risk_feature = float(last_risk_scale)

    extra_vec = np.array(
        [pos_feature, unrealized_atr, bars_held_norm, risk_feature],
        dtype=np.float32
    )
    extra_block = np.repeat(extra_vec[None, :], cfg.lookback, axis=0)

    obs = np.concatenate([base, extra_block], axis=-1).astype(np.float32)
    return obs


# ============================================================
# POSITION LIVE (lecture MT5)
# ============================================================

def get_current_position(symbol: str) -> Tuple[int, float]:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return 0, 0.0

    p = positions[0]
    if p.type == mt5.POSITION_TYPE_BUY:
        pos = 1
    elif p.type == mt5.POSITION_TYPE_SELL:
        pos = -1
    else:
        pos = 0

    entry_price = float(p.price_open)
    return pos, entry_price


def get_position_by_magic(symbol: str, magic: int):
    """Retourne la première position MT5 du symbole avec ce magic (None si aucune)."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return None
    for p in positions:
        if int(getattr(p, "magic", 0)) == magic:
            return p
    return None


def get_current_position_by_magic(symbol: str, magic: int) -> Tuple[int, float, Optional[int]]:
    """Variante multi-agent : trouve une position par magic.
    Retour : (side ±1 / 0, entry_price, ticket ou None)."""
    p = get_position_by_magic(symbol, magic)
    if p is None:
        return 0, 0.0, None
    if p.type == mt5.POSITION_TYPE_BUY:
        side = 1
    elif p.type == mt5.POSITION_TYPE_SELL:
        side = -1
    else:
        return 0, 0.0, None
    return side, float(p.price_open), int(p.ticket)


# ============================================================
# Cache entry_atr par ticket — pour figer la feature unrealized_atr
# IDENTIQUE training/backtest (qui freezent l'ATR à l'ouverture)
# ============================================================
_ENTRY_ATR_CACHE: dict[int, float] = {}


def get_position_ticket(symbol: str) -> Optional[int]:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return None
    return int(positions[0].ticket)


def get_entry_atr_cached(symbol: str, fallback_df: pd.DataFrame) -> float:
    """Retourne l'entry_atr mémorisé pour la position courante.
    Si pas de position ou cache miss : fallback sur la DF actuelle (drift accepté
    pour le 1er appel après reprise/redémarrage)."""
    ticket = get_position_ticket(symbol)
    if ticket is not None and ticket in _ENTRY_ATR_CACHE:
        return _ENTRY_ATR_CACHE[ticket]
    return compute_entry_atr(fallback_df)


def gc_entry_atr_cache(symbol: str) -> None:
    """Supprime du cache les tickets qui ne correspondent plus à des positions
    ouvertes (le ticket survit en MT5 history mais on n'en a plus besoin)."""
    positions = mt5.positions_get(symbol=symbol)
    open_tickets = {int(p.ticket) for p in positions} if positions else set()
    for t in list(_ENTRY_ATR_CACHE.keys()):
        if t not in open_tickets:
            _ENTRY_ATR_CACHE.pop(t, None)


def adjust_volume_to_margin(cfg: LiveConfig, side: int, price: float, desired_volume: float,
                            agent_name: str = "") -> float:
    """Réduit le volume si la marge libre ne permet pas le notional requis.

    - Utilise mt5.order_calc_margin pour obtenir la marge exacte requise par 1 lot
    - Calcule le volume max autorisé : margin_free × safety / margin_per_lot
    - Arrondit au pas de volume du broker (volume_step)
    - Retourne 0.0 si même min_volume n'est pas tenable
    """
    symbol = cfg.symbol
    info = mt5.account_info()
    if info is None:
        return desired_volume  # pas d'info, on tente
    margin_free = float(info.margin_free)

    order_type = mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL
    margin_for_desired = mt5.order_calc_margin(order_type, symbol, desired_volume, price)
    if margin_for_desired is None or margin_for_desired <= 0:
        return desired_volume

    safety = float(getattr(cfg, "margin_safety", 0.80))
    margin_budget = margin_free * safety

    if margin_for_desired <= margin_budget:
        return desired_volume  # marge suffisante, pas de réduction

    # Margin par lot (extrapolation linéaire)
    margin_per_lot = margin_for_desired / desired_volume if desired_volume > 0 else 0.0
    if margin_per_lot <= 0:
        return 0.0

    max_volume = margin_budget / margin_per_lot

    # Arrondit au pas du broker
    sym_info = mt5.symbol_info(symbol)
    volume_step = float(getattr(sym_info, "volume_step", 0.01)) if sym_info else 0.01
    volume_min  = float(getattr(sym_info, "volume_min",  0.01)) if sym_info else 0.01
    adjusted = (max_volume // volume_step) * volume_step
    adjusted = max(adjusted, 0.0)
    adjusted = round(adjusted, 2)

    min_acceptable = max(volume_min, float(getattr(cfg, "min_volume", 0.01)))
    if adjusted < min_acceptable:
        tag = f"[{agent_name}] " if agent_name else ""
        print(
            f"  {tag}⚠ MARGE INSUFFISANTE : margin_free={margin_free:.2f}$ × "
            f"safety={safety:.0%} = {margin_budget:.2f}$ ; required pour {desired_volume:.2f} lot = "
            f"{margin_for_desired:.2f}$ → adjusted={adjusted:.2f} < min={min_acceptable:.2f}, ORDRE ANNULÉ"
        )
        return 0.0

    tag = f"[{agent_name}] " if agent_name else ""
    print(
        f"  {tag}↘ SCALE-DOWN volume {desired_volume:.2f} → {adjusted:.2f} "
        f"(margin_free={margin_free:.2f}$, budget {safety:.0%}={margin_budget:.2f}$, "
        f"margin/lot={margin_per_lot:.2f}$)"
    )
    return adjusted


def send_order(cfg: LiveConfig, side: int, risk_scale: float, df_merged_closed: pd.DataFrame,
               magic: int = 424242, agent_name: str = ""):
    """Envoie un ordre MT5. Le paramètre magic permet d'identifier l'agent
    qui a ouvert la position (utile pour multi-agent)."""
    symbol = cfg.symbol
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("Erreur : pas de tick MT5 pour", symbol)
        return

    if side == 1:
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL

    # Le SL doit etre connu AVANT la taille : c'est sa distance qui la fixe.
    entry_atr = compute_entry_atr(df_merged_closed)
    sl, tp = compute_sl_tp(cfg, price, side, entry_atr)

    etiq = ("/" + agent_name) if agent_name else ""

    # Volume : par le RISQUE en priorite, sinon paliers d'equity, sinon fixe.
    if getattr(cfg, "risk_volume", True):
        info = mt5.account_info()
        equity = float(info.equity) if info is not None else 0.0
        # symbol_select AVANT symbol_info : MT5 renvoie trade_tick_value = 0.0
        # pour un symbole absent du Market Watch, ce qui annulerait l'ordre.
        mt5.symbol_select(symbol, True)
        sinfo = mt5.symbol_info(symbol)
        if sinfo is None or equity <= 0.0:
            print(f"[VOL{etiq}] symbol_info/account_info indisponible — ordre annule")
            return
        base_volume, risque_eff, plancher = compute_risk_volume(
            equity=equity,
            risk_frac=float(getattr(cfg, "risk_per_trade", 0.012)),
            sl_dist=abs(price - sl),
            tick_value=float(sinfo.trade_tick_value),
            tick_size=float(sinfo.trade_tick_size),
            vol_min=float(sinfo.volume_min),
            vol_step=float(sinfo.volume_step),
            vol_max=min(float(sinfo.volume_max), float(getattr(cfg, "max_lot", 100.0))),
            contract_size=float(getattr(sinfo, "trade_contract_size", 0.0) or 0.0),
        )
        if base_volume <= 0.0:
            print(f"[VOL{etiq}] volume par le risque = 0 — ordre annule")
            return
        msg = (f"[VOL{etiq}] equity={equity:.2f}$ SL={abs(price - sl):.2f} "
               f"→ lot={base_volume:g} (risque {100*risque_eff:.2f} %)")
        if plancher:
            # Le seul cas ou le controle du risque echoue : il doit etre visible.
            msg += (f"  ⚠ VOLUME MINIMUM DU COURTIER — risque impose "
                    f"{100*risque_eff:.2f} % au lieu de "
                    f"{100*float(getattr(cfg, 'risk_per_trade', 0.012)):.2f} %")
        print(msg)
    elif getattr(cfg, "dynamic_volume", False):
        info = mt5.account_info()
        equity = float(info.equity) if info is not None else 0.0
        base_volume = compute_dynamic_volume(equity, getattr(cfg, "max_lot", 100.0))
        print(f"[VOL{etiq}] equity={equity:.2f}$ → lot={base_volume:.2f}")
    else:
        base_volume = float(cfg.position_size)

    volume = base_volume * (risk_scale if risk_scale > 0 else 1.0)

    # Ajustement marge : reduit le lot si margin_free insuffisante (anti reject NO_MONEY)
    if getattr(cfg, "auto_scale_volume_to_margin", True):
        volume = adjust_volume_to_margin(cfg, side, price, volume, agent_name=agent_name)
        if volume <= 0.0:
            return  # ordre annule pour cause de marge

    comment = f"SAINTv2_{agent_name}" if agent_name else "SAINTv2_Live_duel"
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 50,
        "magic": magic,
        "comment": comment,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC,
    }

    result = mt5.order_send(request)
    if result is None:
        print("Erreur order_send : None")
        return

    # Retry "no money" : retcode 10019 → on retente avec volume divisé par 2
    # jusqu'à atteindre min_volume (au cas où le pré-check n'avait pas suffi)
    retry_volume = volume
    retry_count = 0
    while result is not None and result.retcode == 10019 and retry_count < 4:
        retry_count += 1
        retry_volume = round(retry_volume / 2.0, 2)
        min_acceptable = max(0.01, float(getattr(cfg, "min_volume", 0.01)))
        if retry_volume < min_acceptable:
            print(f"  [{agent_name}] ⚠ NO_MONEY × {retry_count} : volume {retry_volume:.2f} < min, abandon.")
            return
        print(f"  [{agent_name}] ↘ NO_MONEY retry #{retry_count} : volume {volume:.2f} → {retry_volume:.2f}")
        request["volume"] = retry_volume
        result = mt5.order_send(request)
        volume = retry_volume

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        rc = result.retcode if result is not None else "None"
        print(f"Order_send échoué (magic={magic}), retcode={rc}")
    else:
        new_pos = get_position_by_magic(symbol, magic)
        if new_pos is not None:
            _ENTRY_ATR_CACHE[int(new_pos.ticket)] = float(entry_atr)
        tag = f" [{agent_name}]" if agent_name else ""
        print(f"Order exécuté{tag} : side={side}, vol={volume}, prix={price}, SL={sl:.2f}, TP={tp:.2f}, magic={magic}")


def modify_sl_tp(position, new_sl: float | None = None, new_tp: float | None = None):
    symbol = position.symbol
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"[modify_sl_tp] Impossible de récupérer symbol_info pour {symbol}")
        return

    point = info.point
    stops_level_points = getattr(info, "trade_stops_level", 0) or 0
    freeze_level_points = getattr(info, "trade_freeze_level", 0) or 0

    MIN_EXTRA_POINTS = 100
    min_points = max(stops_level_points, MIN_EXTRA_POINTS)
    min_price_dist = min_points * point

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[modify_sl_tp] Pas de tick pour {symbol}")
        return

    bid = tick.bid
    ask = tick.ask

    current_sl = float(position.sl) if position.sl > 0 else 0.0
    current_tp = float(position.tp) if position.tp > 0 else 0.0

    desired_sl = current_sl if new_sl is None else float(new_sl)

    if new_tp is None:
        desired_tp = 0.0
    else:
        desired_tp = float(new_tp)

    if position.type == mt5.POSITION_TYPE_BUY:
        if desired_sl > 0:
            max_sl_allowed = bid - min_price_dist
            if desired_sl > max_sl_allowed:
                print(
                    f"[WARN] new_sl ({desired_sl:.2f}) trop proche du BID ({bid:.2f}), "
                    f"clamp → {max_sl_allowed:.2f} (min_dist={min_price_dist:.5f})"
                )
                desired_sl = max_sl_allowed

        if desired_sl <= 0 or (current_sl > 0 and desired_sl <= current_sl):
            print(
                f"[INFO] SL BUY non modifié : "
                f"old_sl={current_sl:.2f}, candidate={desired_sl:.2f}"
            )
            desired_sl = current_sl

    elif position.type == mt5.POSITION_TYPE_SELL:
        if desired_sl > 0:
            min_sl_allowed = ask + min_price_dist
            if desired_sl < min_sl_allowed:
                print(
                    f"[WARN] new_sl ({desired_sl:.2f}) trop proche de l'ASK ({ask:.2f}), "
                    f"clamp → {min_sl_allowed:.2f} (min_dist={min_price_dist:.5f})"
                )
                desired_sl = min_sl_allowed

        if desired_sl <= 0 or (current_sl > 0 and desired_sl >= current_sl):
            print(
                f"[INFO] SL SELL non modifié : "
                f"old_sl={current_sl:.2f}, candidate={desired_sl:.2f}"
            )
            desired_sl = current_sl

    if (
        abs(desired_sl - current_sl) < point / 2.0
        and abs(desired_tp - current_tp) < point / 2.0
    ):
        print(
            f"[INFO] SL/TP identiques (sl={current_sl:.2f}, tp={current_tp:.2f}), "
            "aucune modification envoyée."
        )
        return

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": position.ticket,
        "sl": desired_sl,
        "tp": desired_tp,
        "magic": 424242,
        "comment": "SAINTv2_update_sl_tp_noTP",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    print(
        f"Envoi TRADE_ACTION_SLTP : "
        f"ticket={position.ticket}, old_sl={current_sl:.2f}, new_sl={desired_sl:.2f}, "
        f"old_tp={current_tp:.2f}, new_tp={desired_tp:.2f}, "
        f"bid={bid:.2f}, ask={ask:.2f}, "
        f"stops_level_pts={stops_level_points}, freeze_level_pts={freeze_level_points}"
    )

    result = mt5.order_send(request)
    if result is None:
        print("[modify_sl_tp] Erreur : result=None")
        return

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[modify_sl_tp] Erreur, retcode={result.retcode}")
    else:
        print("[modify_sl_tp] Modification SL/TP OK.")


def update_sl_be_trailing_live(cfg: LiveConfig, df_closed: pd.DataFrame, position):
    """
    Break-even + trailing, mais en se basant sur le PRIX LIVE (tick),
    et pas seulement sur la dernière bougie fermée.

    - ATR est toujours calculé sur les bougies fermées (df_closed)
    - Le "favorable_move" et le trailing se font sur le bid/ask courant.
    """
    if len(df_closed) == 0:
        return

    # ATR "lissé" comme pour l'entrée
    atr = compute_entry_atr(df_closed)
    if atr <= 0:
        return

    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        print("[TRAIL] Pas de tick disponible.")
        return

    # BUY → sortie au BID
    # SELL → sortie à l'ASK
    if position.type == mt5.POSITION_TYPE_BUY:
        side = 1
        current_price = float(tick.bid)
    elif position.type == mt5.POSITION_TYPE_SELL:
        side = -1
        current_price = float(tick.ask)
    else:
        return

    entry_price = float(position.price_open)
    current_sl = float(position.sl) if position.sl is not None else 0.0

    if side == 1:
        favorable_move = current_price - entry_price
    else:
        favorable_move = entry_price - current_price

    if favorable_move <= 0:
        # Rien à faire tant qu'on n'est pas au moins un peu en gain
        return

    new_sl = None
    reason = ""

    # ----- Break-even -----
    be_trigger = cfg.breakeven_atr_mult * atr

    if favorable_move >= be_trigger:
        if side == 1:
            # LONG : on remonte le SL à l'entry si encore sous l'entry
            if current_sl < entry_price or current_sl == 0.0:
                new_sl = entry_price
                reason = "BREAKEVEN_LONG"
        else:
            # SHORT
            if current_sl > entry_price or current_sl == 0.0:
                new_sl = entry_price
                reason = "BREAKEVEN_SHORT"

    # ----- Trailing -----
    trail_trigger = cfg.trailing_start_atr_mult * atr
    trail_dist = cfg.trailing_dist_atr_mult * atr

    if favorable_move >= trail_trigger:
        if side == 1:
            # LONG : SL = prix courant - trail_dist (mais jamais sous l'entry)
            candidate_sl = current_price - trail_dist
            candidate_sl = max(candidate_sl, entry_price)
            if current_sl < candidate_sl:
                new_sl = candidate_sl
                reason = "TRAIL_LONG"
        else:
            # SHORT : SL = prix courant + trail_dist (mais jamais au-dessus de l'entry)
            candidate_sl = current_price + trail_dist
            candidate_sl = min(candidate_sl, entry_price)
            if current_sl == 0.0 or current_sl > candidate_sl:
                new_sl = candidate_sl
                reason = "TRAIL_SHORT"

    if new_sl is not None:
        if abs(new_sl - current_sl) > 1e-5:
            print(
                f"[TRAIL] Update SL ({reason}) : "
                f"old={current_sl:.2f} → new={new_sl:.2f} "
                f"(prix courant={current_price:.2f}, atr={atr:.2f})"
            )
            # new_tp=None → TP forcé à 0.0 dans modify_sl_tp (no TP)
            modify_sl_tp(position, new_sl, None)


# ============================================================
# BOUCLE LIVE MULTI-AGENT (wf1 + wf2 + wf3 en parallèle)
# Chaque agent identifie ses positions via son magic dédié.
# ============================================================

def live_loop_multi(cfg: LiveConfig, should_continue):
    print("Connexion MT5 (live multi-agent)…")
    if not mt5.initialize():
        raise RuntimeError("Erreur MT5.initialize() en live multi-agent.")

    device = get_device(cfg)
    agent_stats = {}

    def _build_policy():
        return build_policy(device, lookback=cfg.lookback)

    # Filtre des agents actifs (None = tous)
    active = getattr(cfg, "active_agents", None)
    if active is None:
        agents_to_load = list(MULTI_AGENT_PATHS.keys())
    else:
        agents_to_load = [a for a in active if a in MULTI_AGENT_PATHS]
        if not agents_to_load:
            raise ValueError(f"active_agents={active} ne contient aucun agent valide. "
                             f"Choix possibles : {list(MULTI_AGENT_PATHS.keys())}")
    print(f"Agents actifs : {' / '.join(a.upper() for a in agents_to_load)}")

    # Chargement des checkpoints
    policies: Dict[str, nn.Module] = {}
    for agent_name in agents_to_load:
        path = MULTI_AGENT_PATHS[agent_name]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint {agent_name} introuvable : {path}")
        p = _build_policy()
        p.load_state_dict(torch.load(path, map_location=device))
        p.eval()
        policies[agent_name] = p
        agent_stats[agent_name] = load_model_norm_stats(path)
        magic = MULTI_AGENT_MAGICS[agent_name]
        print(f"Modèle {agent_name.upper():3s} chargé (magic={magic}) : {path}")

    action_labels = {0: "BUY", 1: "SELL", 2: "HOLD"}
    last_bar_time = None

    try:
        while should_continue():
            # GC du cache entry_atr (positions fermées)
            gc_entry_atr_cache(cfg.symbol)

            # Vérifie qu'au moins une bougie M1 fermée existe
            try:
                df_merged_full = fetch_ohlc_with_indicators(cfg)
            except Exception as e:
                print(f"[ERREUR MT5] {e} → pause 5s puis retry.")
                time.sleep(5)
                continue

            if len(df_merged_full) < cfg.lookback + 3:
                time.sleep(cfg.poll_interval)
                continue

            current_last_time = df_merged_full["time"].iloc[-1]

            # On ne décide qu'une fois par nouvelle bougie M1 fermée
            if last_bar_time is not None and current_last_time == last_bar_time:
                time.sleep(cfg.poll_interval)
                continue
            last_bar_time = current_last_time

            df_closed = df_merged_full.iloc[:-1].reset_index(drop=True)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Nouvelle bougie M1 fermée à {current_last_time}")

            # ========================================================
            # Pour chaque agent : check sa position, sinon décision
            # ========================================================
            for agent_name, policy in policies.items():
                magic = MULTI_AGENT_MAGICS[agent_name]
                pos, entry_price, ticket = get_current_position_by_magic(cfg.symbol, magic)

                if pos != 0:
                    print(f"  [{agent_name.upper()}] déjà en position (ticket={ticket}, side={pos}) → SKIP")
                    continue

                # Construction de l'obs (cet agent est flat)
                obs = build_live_obs(
                    df_closed, agent_stats[agent_name], cfg,
                    pos=0,
                    entry_price=0.0,
                    last_risk_scale=1.0,
                    bars_in_position=0,
                )
                if obs is None:
                    print(f"  [{agent_name.upper()}] obs None → SKIP")
                    continue

                with torch.no_grad():
                    s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    logits_d, _ = policy(s)
                    logits_d = logits_d[0]
                    mask_d = build_mask_from_pos_scalar(0, device, "both")
                    logits_d_m = logits_d.masked_fill(~mask_d, MASK_VALUE)
                    probs = torch.softmax(logits_d_m, dim=-1)
                    pb, ps = float(probs[0]), float(probs[1])

                barres = get_calib_threshold(agent_name)
                a_pred = decide_avec_barres(pb, ps, barres)
                print(
                    f"  [{agent_name.upper()}] probas "
                    f"BUY={probs[0]:.2f} SELL={probs[1]:.2f} HOLD={probs[2]:.2f}  "
                    f"barres B={barres[0]:.3f} S={barres[1]:.3f}  "
                    f"→ {action_labels[a_pred]}"
                )
                if a_pred == 2:
                    print(f"  [{agent_name.upper()}] aucun côté au-dessus de sa "
                          f"barre → HOLD")
                    continue

                # Action finale
                if a_pred == 0:
                    side = 1
                elif a_pred == 1:
                    side = -1
                else:
                    continue  # HOLD

                # Ouverture de la position pour cet agent
                send_order(cfg, side, risk_scale=1.0, df_merged_closed=df_closed,
                           magic=magic, agent_name=agent_name.upper())

            time.sleep(cfg.poll_interval)

    finally:
        mt5.shutdown()
        print("Multi-agent live_loop terminé.")


# ============================================================
# BOUCLE LIVE  (avec callback should_continue)
# ============================================================

def live_loop(cfg: LiveConfig, should_continue):
    print("Connexion MT5 (live)…")
    if not mt5.initialize():
        raise RuntimeError("Erreur MT5.initialize() en live.")

    device = get_device(cfg)
    norm_paths = ([BEST_MODEL_DUEL_PATH] if cfg.side == "both" else
                  [p for side, p in (("long", BEST_MODEL_LONG_PATH), ("short", BEST_MODEL_SHORT_PATH))
                   if cfg.side in (side, "duel")])
    stats = load_shared_model_norm_stats(norm_paths)

    def _build_policy():
        return build_policy(device, lookback=cfg.lookback)

    policy_long = None
    policy_short = None
    policy_duel = None

    if cfg.side == "both":
        if not os.path.exists(BEST_MODEL_DUEL_PATH):
            raise FileNotFoundError(f"Modèle DUEL introuvable : {BEST_MODEL_DUEL_PATH}")
        policy_duel = _build_policy()
        policy_duel.load_state_dict(torch.load(BEST_MODEL_DUEL_PATH, map_location=device))
        policy_duel.eval()
        print(f"Modèle DUEL chargé : {BEST_MODEL_DUEL_PATH}")
    else:
        if cfg.side in ("duel", "long"):
            if not os.path.exists(BEST_MODEL_LONG_PATH):
                raise FileNotFoundError(f"Modèle LONG introuvable : {BEST_MODEL_LONG_PATH}")
            policy_long = _build_policy()
            policy_long.load_state_dict(torch.load(BEST_MODEL_LONG_PATH, map_location=device))
            policy_long.eval()
            print(f"Modèle LONG chargé : {BEST_MODEL_LONG_PATH}")

        if cfg.side in ("duel", "short"):
            if not os.path.exists(BEST_MODEL_SHORT_PATH):
                raise FileNotFoundError(f"Modèle SHORT introuvable : {BEST_MODEL_SHORT_PATH}")
            policy_short = _build_policy()
            policy_short.load_state_dict(torch.load(BEST_MODEL_SHORT_PATH, map_location=device))
            policy_short.eval()
            print(f"Modèle SHORT chargé : {BEST_MODEL_SHORT_PATH}")

    print(f"Mode side='{cfg.side}'…")

    last_bar_time   = None
    last_risk_scale = 1.0


    try:
        while should_continue():

            # Nettoie le cache entry_atr des tickets fermés
            gc_entry_atr_cache(cfg.symbol)

            # ====================================================
            # 1) TRAILING / BREAK-EVEN TICK-BY-TICK SI POSITION OUVERTE
            # ====================================================
            positions = mt5.positions_get(symbol=cfg.symbol)
            if positions is not None and len(positions) > 0:
                position = positions[0]
                print(
                    f"\n[TRAIL LOOP] Position MT5 : ticket={position.ticket}, "
                    f"type={'BUY' if position.type == mt5.POSITION_TYPE_BUY else 'SELL'}, "
                    f"volume={position.volume}, sl={position.sl}, tp={position.tp}"
                )

                # On récupère quand même les bougies fermées pour l'ATR
                try:
                    df_merged_full = fetch_ohlc_with_indicators(cfg)
                except Exception as e:
                    print(f"[ERREUR MT5] {e} → pause 5s puis retry (trailing).")
                    time.sleep(5)
                    continue

                if len(df_merged_full) < cfg.lookback + 3:
                    print("[TRAIL LOOP] Pas assez de données pour ATR, on attend…")
                    time.sleep(cfg.poll_interval)
                    continue

                df_closed = df_merged_full.iloc[:-1].reset_index(drop=True)

                # BE + TRAILING DÉSACTIVÉS — signal brut SL/TP fixes uniquement.
                # Aligné avec backtest_saintv2_no_be_trail.py qui montre que le
                # BE/trail coupe trop tôt les wins (PF 0.98 avec → 1.70 sans).
                # update_sl_be_trailing_live(cfg, df_closed, position)

                time.sleep(cfg.poll_interval)
                continue  # On NE prend PAS de nouvelles positions tant qu'on est déjà en trade.

            # ====================================================
            # 2) AUCUNE POSITION → LOGIQUE D'ENTRÉE (M1)
            # ====================================================
            try:
                df_merged_full = fetch_ohlc_with_indicators(cfg)
            except Exception as e:
                print(f"[ERREUR MT5] {e} → pause 5s puis retry.")
                time.sleep(5)
                continue

            if len(df_merged_full) < cfg.lookback + 3:
                print("Pas assez de données pour construire l'obs, on attend…")
                time.sleep(cfg.poll_interval)
                continue

            current_last_time = df_merged_full["time"].iloc[-1]
            # On ne déclenche une DÉCISION que sur nouvelle bougie fermée
            if last_bar_time is not None and current_last_time == last_bar_time:
                # Pas de nouvelle bougie → on attend juste (mais pas de trailing ici, car on serait flat)
                time.sleep(cfg.poll_interval)
                continue

            last_bar_time = current_last_time

            # Dernière bougie FERMÉE
            df_closed = df_merged_full.iloc[:-1].reset_index(drop=True)
            closed_bar_time = df_closed["time"].iloc[-1]

            print(f"\n[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] Nouvelle bougie M1 FERMÉE, time={closed_bar_time}")

            pos, entry_price = get_current_position(cfg.symbol)
            print(f"Position actuelle (net) : {pos}, entry_price={entry_price}")

            if pos != 0:
                # Normalement on ne devrait pas arriver ici car le bloc "positions != 0" est géré plus haut,
                # mais on laisse la sécurité.
                print("Incohérence : pos != 0 dans la branche FLAT, on skip.")
                time.sleep(cfg.poll_interval)
                continue

            last_risk_scale = 1.0

            # ================= FILTRE LONG-ONLY DEMANDÉ =================
            # Identique à celui du backtest :
            # si close[-1] < close[-3] * 0.9975 → on ne prend PAS d'entrée LONG
            #if cfg.side == "long" and len(df_closed) >= 3:
             #   last_close = df_closed["close"].iloc[-1]
              #  close_3 = df_closed["close"].iloc[-3]
               # if last_close < close_3 * 0.9975:
                #    print(
                 #       f"[FILTER LONG LIVE] close[-1]={last_close:.2f} < "
                  #      f"close[-3]*0.9975={close_3*0.9975:.2f} → aucun ordre ouvert sur cette bougie."
                   # )
                    #reset_confirmation()
                    #time.sleep(cfg.poll_interval)
                    #continue
            # ============================================================

            obs = build_live_obs(df_closed, stats, cfg, pos, entry_price, last_risk_scale)
            if obs is None:
                print("Impossible de construire l'obs (manque de données), on attend…")
                time.sleep(cfg.poll_interval)
                continue

            # ========== DÉCISION DU MODELE ==========

            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                a = 2  # default HOLD (index HOLD = 2 dans la convention 3-actions)
                action_labels = ['BUY', 'SELL', 'HOLD']

                if cfg.side == "both":
                    if policy_duel is None:
                        print("policy_duel non chargé alors que side='both' → HOLD.")
                        a = 2
                    else:
                        logits_d, _ = policy_duel(s)
                        logits_d = logits_d[0]
                        mask_d = build_mask_from_pos_scalar(0, device, "both")
                        logits_d_m = logits_d.masked_fill(~mask_d, MASK_VALUE)
                        probs_d = torch.softmax(logits_d_m, dim=-1)

                        print("Logits DUEL :", logits_d_m.cpu().numpy().round(4))
                        print("Probas DUEL :", probs_d.cpu().numpy().round(4))

                        # Une barre par côté — identique au training et au
                        # mode multi-agent.
                        barres_d = seuil_calibre(BEST_MODEL_DUEL_PATH)
                        a = decide_avec_barres(float(probs_d[0]),
                                               float(probs_d[1]), barres_d)
                        print(f"BEST DUEL : {action_labels[a]}  "
                              f"barres B={barres_d[0]:.3f} S={barres_d[1]:.3f}")

                elif cfg.side == "duel":
                    if policy_long is None or policy_short is None:
                        a = 2
                        print("Policies LONG/SHORT non chargées → HOLD.")
                    else:
                        logits_long, _ = policy_long(s)
                        logits_long = logits_long[0]
                        mask_long = build_mask_from_pos_scalar(0, device, "long")
                        logits_long_m = logits_long.masked_fill(~mask_long, MASK_VALUE)
                        probs_long = torch.softmax(logits_long_m, dim=-1)

                        logits_short, _ = policy_short(s)
                        logits_short = logits_short[0]
                        mask_short = build_mask_from_pos_scalar(0, device, "short")
                        logits_short_m = logits_short.masked_fill(~mask_short, MASK_VALUE)
                        probs_short = torch.softmax(logits_short_m, dim=-1)

                        print("Logits LONG  :", logits_long_m.cpu().numpy().round(4))
                        print("Probas LONG  :", probs_long.cpu().numpy().round(4))
                        print("Logits SHORT :", logits_short_m.cpu().numpy().round(4))
                        print("Probas SHORT :", probs_short.cpu().numpy().round(4))

                        # Compare la conviction d'entrée (prob de BUY pour long vs SELL pour short)
                        p_long_buy  = probs_long[0].item()
                        p_short_sell = probs_short[1].item()

                        if p_long_buy > p_short_sell:
                            print(f"[DUEL] LONG choisi, p(BUY)={p_long_buy:.3f}")
                            cand = 0 if int(torch.argmax(probs_long).item()) == 0 else 2
                            p_cand = p_long_buy
                        else:
                            print(f"[DUEL] SHORT choisi, p(SELL)={p_short_sell:.3f}")
                            cand = 1 if int(torch.argmax(probs_short).item()) == 1 else 2
                            p_cand = p_short_sell

                        # Filtre confiance
                        if cand in (0, 1) and p_cand < max(seuil_calibre(BEST_MODEL_LONG_PATH)):
                            print(f"[CONF] p {p_cand:.3f} sous la barre calibree -> HOLD")
                            a = 2
                        else:
                            a = cand

                elif cfg.side == "long":
                    if policy_long is None:
                        print("policy_long non chargé alors que side='long' → HOLD.")
                        a = 2
                    else:
                        logits_long, _ = policy_long(s)
                        logits_long = logits_long[0]
                        mask_long = build_mask_from_pos_scalar(0, device, "long")
                        logits_long_m = logits_long.masked_fill(~mask_long, MASK_VALUE)
                        probs_long = torch.softmax(logits_long_m, dim=-1)

                        print("Logits LONG  :", logits_long_m.cpu().numpy().round(4))
                        print("Probas LONG  :", probs_long.cpu().numpy().round(4))

                        a_long = int(torch.argmax(probs_long, dim=-1).item())
                        p_long = float(probs_long[a_long].item())
                        print(f"BEST LONG : action={a_long}, prob={p_long:.3f}")
                        if a_long in (0, 1) and p_long < max(seuil_calibre(BEST_MODEL_LONG_PATH)):
                            print(f"[CONF] p {p_long:.3f} sous la barre calibree -> HOLD")
                            a = 2
                        else:
                            a = a_long

                elif cfg.side == "short":
                    if policy_short is None:
                        print("policy_short non chargé alors que side='short' → HOLD.")
                        a = 2
                    else:
                        logits_short, _ = policy_short(s)
                        logits_short = logits_short[0]
                        mask_short = build_mask_from_pos_scalar(0, device, "short")
                        logits_short_m = logits_short.masked_fill(~mask_short, MASK_VALUE)
                        probs_short = torch.softmax(logits_short_m, dim=-1)

                        print("Logits SHORT :", logits_short_m.cpu().numpy().round(4))
                        print("Probas SHORT :", probs_short.cpu().numpy().round(4))

                        a_short = int(torch.argmax(probs_short, dim=-1).item())
                        p_short = float(probs_short[a_short].item())
                        print(f"BEST SHORT : action={a_short}, prob={p_short:.3f}")
                        if a_short in (0, 1) and p_short < max(seuil_calibre(BEST_MODEL_SHORT_PATH)):
                            print(f"[CONF] p {p_short:.3f} sous la barre calibree -> HOLD")
                            a = 2
                        else:
                            a = a_short
                else:
                    print(f"cfg.side invalide : {cfg.side}, on HOLD.")
                    a = 2


            print(f"Action finale (0:BUY, 1:SELL, 2:HOLD) : {a} ({action_labels[a]})")

            # Espace 3 actions, risk_scale toujours 1.0
            if a == 0:    # BUY
                env_action = 0
                risk_scale = 1.0
            elif a == 1:  # SELL
                env_action = 1
                risk_scale = 1.0
            else:         # HOLD ou autre
                env_action = 2
                risk_scale = 1.0

            print(f"Env_action (0=BUY,1=SELL,2=HOLD) : {env_action}, risk_scale={risk_scale}")

            # Sécurité juste avant l’envoi
            pos_check, _ = get_current_position(cfg.symbol)
            if pos_check != 0:
                print("Position détectée juste avant l'envoi de l'ordre → annulation de l'ouverture.")
                last_risk_scale = 1.0
                time.sleep(cfg.poll_interval)
                continue

            if env_action == 0:
                send_order(cfg, side=1, risk_scale=risk_scale, df_merged_closed=df_closed)
                last_risk_scale = risk_scale
            elif env_action == 1:
                send_order(cfg, side=-1, risk_scale=risk_scale, df_merged_closed=df_closed)
                last_risk_scale = risk_scale
            else:
                print("HOLD (flat) → aucune ouverture.")
                last_risk_scale = 1.0

            time.sleep(cfg.poll_interval)

    finally:
        mt5.shutdown()
        print("MT5 shutdown, fin du live agent.")


# ============================================================
# WRAPPER POUR LA GUI : TradingAgent
# ============================================================

class TradingAgent:
    def __init__(self, cfg: LiveConfig):
        self.cfg = cfg
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _should_continue(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            print("[AGENT] Déjà en cours d’exécution.")
            return

        print("[AGENT] Démarrage du bot…")
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            if getattr(self.cfg, "multi_agent", False):
                print("[AGENT] Mode MULTI-AGENT (wf1 + wf2 + wf3)")
                live_loop_multi(self.cfg, self._should_continue)
            else:
                live_loop(self.cfg, self._should_continue)
        except Exception as e:
            print(f"[AGENT] Erreur dans live_loop : {e}")
        finally:
            self._running = False
            print("[AGENT] live_loop terminé.")

    def stop(self):
        if not self._running:
            print("[AGENT] Bot déjà arrêté.")
            return

        print("[AGENT] Arrêt demandé…")
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        print("[AGENT] Bot arrêté.")
