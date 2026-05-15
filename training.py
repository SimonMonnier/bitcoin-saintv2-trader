# ======================================================================
# PPO + SAINTv2 — SCALPING BTCUSD M1 (SINGLE-HEAD + ACTION MASK + H1)
# Version "Loup Ω" LONG / SHORT / CLOSE
# ======================================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import random
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import MetaTrader5 as mt5
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.distributions import Categorical

# Optimisations PyTorch
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ============================================================
# SEED GLOBAL
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# CONSTANTES
# ============================================================

# 0:BUY1, 1:SELL1, 2:BUY1.8, 3:SELL1.8, 4:HOLD
N_ACTIONS = 5
MASK_VALUE = -1e4  # valeur de masquage compatible float16
CONF_THRESHOLD = 0.95  # seuil de confiance (exploration entraînement)

# Fichier de normalisation global
NORM_STATS_PATH = "norm_stats_ohlc_indics.npz"

# Modèles pré-entraînés pour le mode CLOSE
BEST_MODEL_LONG_PATH = "best_saintv2_loup_long_wf1_long_wf1.pth"
BEST_MODEL_SHORT_PATH = "best_saintv2_loup_short_wf1_short_wf1.pth"


# ============================================================
# CONFIG
# ============================================================

@dataclass
class PPOConfig:
    # Données
    symbol: str = "BTCUSD"
    timeframe: int = mt5.TIMEFRAME_M1
    htf_timeframe: int = mt5.TIMEFRAME_H1
    n_bars: int = 261800
    lookback: int = 25

    # PPO Training
    epochs: int = 160
    episodes_per_epoch: int = 3
    episode_length: int = 2333
    updates_per_epoch: int = 2
    tp_shrink: float = 0.7

    batch_size: int = 256
    gamma: float = 0.97
    lambda_gae: float = 0.95
    clip_eps: float = 0.18
    lr: float = 3e-4
    target_kl: float = 0.03
    value_coef: float = 0.5
    entropy_coef: float = 0.15
    max_grad_norm: float = 1.0

    # SAINT
    d_model: int = 80

    # Trading
    initial_capital: float = 1000.0
    position_size: float = 0.06
    leverage: float = 6.0
    fee_rate: float = 0.0004
    min_capital_frac: float = 0.2
    max_drawdown: float = 0.8

    # Risk management / position sizing
    risk_per_trade: float = 0.012
    max_position_frac: float = 0.35
    position_vol_penalty: float = 1e-3

    # StopLoss / TakeProfit ATR
    atr_sl_mult: float = 1.2
    atr_tp_mult: float = 2.4

    # Microstructure
    spread_bps: float = 0.0
    slippage_bps: float = 0.0  # non utilisé dans la version déterministe

    # Scalp
    scalping_max_holding: int = 12
    scalping_holding_penalty = 1e-6
    scalping_flat_penalty    = 3.5e-5
    scalping_flat_bonus      = 8e-5

    # Break-even & trailing stop (identique au live)
    atr_be_mult: float    = 1.0   # gain en ATR pour déclencher le break-even
    atr_trail_mult: float = 1.5   # gain en ATR pour déclencher le trailing
    atr_trail_dist: float = 1.0   # distance du trailing (en ATR)

    # Warmup critique : N epochs où seul le critique est mis à jour
    critic_warmup_epochs: int = 5

    # Seuil minimum de trades en val pour sauvegarder le meilleur modèle
    min_val_trades_save: int = 20

    # Curriculum vol
    use_vol_curriculum: bool = True

    # Device
    force_cpu: bool = False
    use_amp: bool = True

    # Spécialisation d'agent
    # "both"  → BUY + SELL
    # "long"  → seulement BUY1 / BUY1.8 / HOLD
    # "short" → seulement SELL1 / SELL1.8 / HOLD
    # "close" → agent de clôture
    side: str = "both"

    # Préfixe pour nommer les fichiers de modèle
    model_prefix: str = "saintv2_singlehead_scalping_ohlc_indics_h1_loup"


# ============================================================
# INDICATEURS (M1 & H1) — RSI + VOL + MOMENTUM
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    # ---------- RSI ----------
    def rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - 100 / (1 + rs)

    df["rsi_14"] = rsi(c, 14)

    # ---------- ATR ----------
    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # ---------- Vol / range ----------
    df["returns"] = c.pct_change()
    df["vol_20"] = df["returns"].rolling(20).std()
    df["range_norm"] = (h - l) / (c + 1e-8)

    # ==================================================================
    # BONUS SURPRISE : Momentum-Confirmed Entry Filter (M1 & H1)
    # ==================================================================
    df["mom_5"] = df["close"] > df["close"].shift(5)           # True si prix > close d’il y a 5 barres
    df["rsi_ok"] = (df["rsi_14"] > 28) & (df["rsi_14"] < 72)   # zone "saine" du RSI
    # Volatilité relative du jour (rolling 1440 = 24h en M1)
    df["vol_rank"] = df["vol_20"].rolling(1440).rank(pct=True)
    df["high_vol_regime"] = df["vol_rank"] > 0.65              # top 35% de vol

    return df


# ============================================================
# CHARGEMENT M1 + H1
# ============================================================

def load_mt5_data(cfg: PPOConfig) -> pd.DataFrame:
    print("Connexion MT5…")
    if not mt5.initialize():
        raise RuntimeError("Erreur MT5.init()")

    rates_m1 = mt5.copy_rates_from_pos(
        cfg.symbol, cfg.timeframe, 0, cfg.n_bars
    )

    n_h1 = max(cfg.n_bars // 5, 5000)
    rates_h1 = mt5.copy_rates_from_pos(
        cfg.symbol, cfg.htf_timeframe, 0, n_h1
    )

    mt5.shutdown()

    if rates_m1 is None or rates_h1 is None:
        raise RuntimeError("MT5 n'a renvoyé aucune donnée M1 ou H1")

    df_m1 = pd.DataFrame(rates_m1)
    df_m1["time"] = pd.to_datetime(df_m1["time"], unit="s")
    df_m1.set_index("time", inplace=True)
    df_m1 = df_m1[["open", "high", "low", "close", "tick_volume"]]
    df_m1 = add_indicators(df_m1)

    df_h1 = pd.DataFrame(rates_h1)
    df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
    df_h1.set_index("time", inplace=True)
    df_h1 = df_h1[["open", "high", "low", "close", "tick_volume"]]
    df_h1 = add_indicators(df_h1)
    df_h1 = df_h1.add_suffix("_h1")

    df_m1_reset = df_m1.reset_index()
    df_h1_reset = df_h1.reset_index()
    df_h1_reset = df_h1_reset.rename(columns={"time_h1": "time"})

    merged = pd.merge_asof(
        df_m1_reset.sort_values("time"),
        df_h1_reset.sort_values("time"),
        on="time",
        direction="backward"
    )

    merged = merged.dropna().reset_index(drop=True)
    print(f"{len(merged)} bougies M1 alignées avec H1 après indicateurs.")
    return merged


# ============================================================
# FEATURES / NORMALISATION
# ============================================================

FEATURE_COLS_M1 = [
    "open", "high", "low", "close",
    "rsi_14",
    "returns", "vol_20", "range_norm",
    "mom_5", "rsi_ok", "high_vol_regime",   # 3 features momentum / régime
]

FEATURE_COLS_H1 = [
    "close_h1",
    "rsi_14_h1",
    "returns_h1", "vol_20_h1", "range_norm_h1",
]

FEATURE_COLS = FEATURE_COLS_M1 + FEATURE_COLS_H1

N_BASE_FEATURES = len(FEATURE_COLS)
N_POS_FEATURES = 4
OBS_N_FEATURES = N_BASE_FEATURES + N_POS_FEATURES


def compute_and_save_global_norm_stats(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, np.ndarray]:
    X = df[feature_cols].values.astype(np.float32)
    mean, std = X.mean(0), X.std(0)
    stats = {"mean": mean, "std": std}
    np.savez(NORM_STATS_PATH, mean=mean, std=std)
    print(f"Stats de normalisation GLOBALes sauvegardées → {NORM_STATS_PATH}")
    return stats


def load_global_norm_stats() -> Dict[str, np.ndarray]:
    if not os.path.exists(NORM_STATS_PATH):
        raise FileNotFoundError(f"Fichier de stats globales introuvable : {NORM_STATS_PATH}")
    data = np.load(NORM_STATS_PATH)
    stats = {"mean": data["mean"], "std": data["std"]}
    print(f"Stats de normalisation GLOBALes chargées depuis → {NORM_STATS_PATH}")
    return stats


class MarketData:
    def __init__(self, df: pd.DataFrame, feature_cols: List[str],
                 stats: Optional[Dict[str, np.ndarray]] = None):
        X = df[feature_cols].values.astype(np.float32)

        if stats is not None:
            mean, std = stats["mean"], stats["std"]
            std = np.where(std < 1e-8, 1.0, std)
            X = (X - mean) / std

        self.features = X
        self.close = df["close"].values.astype(np.float32)
        self.length = len(df)

        self.atr14 = df["atr_14"].values.astype(np.float32) if "atr_14" in df.columns else np.zeros(len(df), np.float32)
        self.ema20_h1 = df["ema_20_h1"].values.astype(np.float32) if "ema_20_h1" in df.columns else np.zeros(len(df), np.float32)
        self.high = df["high"].values.astype(np.float32) if "high" in df.columns else np.zeros(len(df), np.float32)
        self.low = df["low"].values.astype(np.float32) if "low" in df.columns else np.zeros(len(df), np.float32)

    def __len__(self):
        return self.length


def create_datasets(df: pd.DataFrame, feature_cols: List[str], stats: Dict[str, np.ndarray]):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    df_train = df[:train_end].reset_index(drop=True)
    df_val = df[train_end:val_end].reset_index(drop=True)
    df_test = df[val_end:].reset_index(drop=True)

    train_data = MarketData(df_train, feature_cols, stats)
    val_data   = MarketData(df_val,   feature_cols, stats)
    test_data  = MarketData(df_test,  feature_cols, stats)

    print(f"SPLIT simple : train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    return train_data, val_data, test_data


def create_datasets_from_slices(
    df: pd.DataFrame,
    feature_cols: List[str],
    start: int,
    train_len: int,
    val_len: int,
    test_len: int,
    stats: Dict[str, np.ndarray]
):
    n = len(df)
    end = start + train_len + val_len + test_len
    assert end <= n, "Fenêtre walk-forward hors limites"

    df_train = df[start:start + train_len].reset_index(drop=True)
    df_val   = df[start + train_len:start + train_len + val_len].reset_index(drop=True)
    df_test  = df[start + train_len + val_len:end].reset_index(drop=True)

    train_data = MarketData(df_train, feature_cols, stats)
    val_data   = MarketData(df_val,   feature_cols, stats)
    test_data  = MarketData(df_test,  feature_cols, stats)

    print(f"  • Fenêtre WF : train={len(df_train)}, val={len(df_val)}, test={len(df_test)} (start={start}, end={end})")
    return train_data, val_data, test_data


# ======================================================================
# ENVIRONNEMENT
# ======================================================================

class BTCTradingEnvDiscrete(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data: MarketData, cfg: PPOConfig):
        super().__init__()
        self.data = data
        self.cfg = cfg
        self.lookback = cfg.lookback

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback, OBS_N_FEATURES),
            dtype=np.float32
        )

        if self.cfg.use_vol_curriculum:
            self._init_vol_curriculum()
        else:
            self.low_vol_starts = None
            self.high_vol_starts = None

        self.risk_scale = 1.0
        self.last_risk_scale = 1.0
        self.reset()

    # ---------- Curriculum vol ----------

    def _init_vol_curriculum(self):
        close = self.data.close
        ret = np.diff(close) / (close[:-1] + 1e-8)
        vol20 = pd.Series(ret).rolling(20).std().to_numpy()
        vol20 = np.concatenate([[np.nan], vol20])

        valid = ~np.isnan(vol20)
        if valid.sum() < 30:
            self.low_vol_starts = None
            self.high_vol_starts = None
            return

        q_low, q_high = np.quantile(vol20[valid], [0.3, 0.7])

        candidate_low = np.where((vol20 <= q_low) & valid)[0]
        candidate_high = np.where((vol20 >= q_high) & valid)[0]

        max_start = self.data.length - self.cfg.episode_length - 2
        low = candidate_low[
            (candidate_low >= self.lookback) &
            (candidate_low <= max_start)
        ]
        high = candidate_high[
            (candidate_high >= self.lookback) &
            (candidate_high <= max_start)
        ]

        self.low_vol_starts = low if len(low) > 0 else None
        self.high_vol_starts = high if len(high) > 0 else None

    def set_risk_scale(self, scale: float):
        self.risk_scale = float(max(scale, 0.0))
        if self.risk_scale <= 0.0:
            self.risk_scale = 1.0
        self.last_risk_scale = float(self.risk_scale)

    # ---------- Gym API ----------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = self.data.length - self.cfg.episode_length - 2

        start_idx = None
        if self.cfg.use_vol_curriculum and self.low_vol_starts is not None and self.high_vol_starts is not None:
            if np.random.rand() < 0.5 and len(self.low_vol_starts) > 0:
                start_idx = int(np.random.choice(self.low_vol_starts))
            elif len(self.high_vol_starts) > 0:
                start_idx = int(np.random.choice(self.high_vol_starts))

        if start_idx is None:
            start_idx = np.random.randint(self.lookback, max_start)

        self.start_idx = start_idx
        self.end_idx = self.start_idx + self.cfg.episode_length
        self.idx = self.start_idx

        self.capital = self.cfg.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.current_size = 0.0

        self.sl_price = 0.0
        self.tp_price = 0.0
        self.entry_idx = -1
        self.entry_atr = 0.0

        self.last_realized_pnl = 0.0
        self.peak_capital = self.capital
        self.trades_pnl: List[float] = []

        self.bars_in_position = 0
        self.risk_scale = 1.0
        self.last_risk_scale = 1.0

        self.max_dd = 0.0

        # Gestion dynamique du SL
        self.break_even_done = False
        self.trail_active    = False

        obs = self._get_obs()
        return obs, {
            "capital": self.capital,
            "position": self.position,
            "drawdown": 0.0,
            "done_reason": None
        }

    def _get_obs(self):
        start = self.idx - self.lookback
        base = self.data.features[start:self.idx]

        if self.idx > 0:
            current_price = float(self.data.close[self.idx - 1])
        else:
            current_price = 0.0

        # PnL latent en unités d'ATR (scale-invariant, typiquement dans [-5, 5])
        if self.position != 0 and self.entry_atr > 1e-8 and self.entry_price > 0.0:
            unrealized_atr = float(
                self.position * (current_price - self.entry_price) / self.entry_atr
            )
        else:
            unrealized_atr = 0.0

        # Durée de détention normalisée (0=flat, 1=max_holding, >1=overtime)
        bars_held_norm = float(
            min(self.bars_in_position / max(self.cfg.scalping_max_holding, 1), 3.0)
        )

        pos_feature  = float(self.position)
        risk_feature = float(self.last_risk_scale)

        extra_vec = np.array(
            [pos_feature, unrealized_atr, bars_held_norm, risk_feature],
            dtype=np.float32
        )
        extra_block = np.repeat(extra_vec[None, :], self.lookback, axis=0)

        obs = np.concatenate([base, extra_block], axis=-1).astype(np.float32)
        return obs

    def _apply_micro(self, price: float, side: int, is_entry: bool = True) -> float:
        spread = self.cfg.spread_bps / 10_000.0
        if is_entry:
            price *= (1 + side * spread * 0.5)
        else:
            price *= (1 - side * spread * 0.5)
        return price

    def _compute_dynamic_size(self, price: float) -> float:
        base = float(self.cfg.position_size)
        scale = float(max(self.risk_scale, 0.0))
        if scale <= 0.0:
            scale = 1.0

        # Taille réduite pendant les 40 premières epochs
        if hasattr(self.cfg, "current_epoch") and self.cfg.current_epoch <= 40:
            size = base * scale * 0.5
        else:
            size = base * scale

        return float(size)

    def step(self, action: int):
        price = self.data.close[self.idx]
        high_bar = self.data.high[self.idx]
        low_bar = self.data.low[self.idx]

        old_pos = self.position
        prev_capital = self.capital

        # ==================================================================
        # 1. PREVIOUS EQUITY (worst-case à t-1)
        # ==================================================================
        prev_latent = 0.0
        if old_pos != 0 and self.current_size > 0 and self.entry_price > 0:
            if self.idx > 0:
                prev_price_worst = self.data.low[self.idx - 1] if old_pos == 1 else self.data.high[self.idx - 1]
            else:
                prev_price_worst = price
            prev_latent = old_pos * (prev_price_worst - self.entry_price) * self.current_size * self.cfg.leverage
        prev_equity = prev_capital + prev_latent

        realized = 0.0
        realized_trade = 0.0
        hit_sl = hit_tp = False

        if old_pos != 0:
            self.bars_in_position += 1
        else:
            self.bars_in_position = 0

        # --------- FERMETURE MANUELLE (mode close) ---------
        manual_close = False
        if self.cfg.side == "close" and old_pos != 0 and action == 0:
            exit_price = self._apply_micro(price, -old_pos)
            pnl = old_pos * (exit_price - self.entry_price) * self.current_size * self.cfg.leverage
            fee = self.cfg.fee_rate * exit_price * self.current_size
            realized = pnl - fee
            realized_trade = realized

            self.capital += realized
            self.last_realized_pnl = realized
            self.trades_pnl.append(realized)

            self.position = 0
            self.current_size = 0.0
            self.entry_price = 0.0
            self.sl_price = 0.0
            self.tp_price = 0.0
            self.entry_idx = -1
            self.entry_atr = 0.0
            self.risk_scale = 1.0
            self.last_risk_scale = 1.0
            self.break_even_done = False
            self.trail_active    = False
            manual_close = True

        # --------- OUVERTURE DIRECTE (pas de confirmation dans l'env) ---------
        # La confirmation signal→pause→re-signal est gérée dans loup_live.py uniquement.
        # En training, le reward shaping pénalise déjà les mauvaises entrées.
        if not manual_close and action in (0, 1) and old_pos == 0:
            side = 1 if action == 0 else -1
            size = self._compute_dynamic_size(price)
            if size > 0.0:
                self.current_size = size
                self.position = side
                exec_price = self._apply_micro(price, side, is_entry=True)
                self.entry_price = exec_price
                self.entry_idx = self.idx

                atr_raw = float(self.data.atr14[self.idx - 1]) if self.idx - 1 >= 0 else 0.0
                fallback = 0.0015 * exec_price
                self.entry_atr = max(atr_raw, fallback, 1e-8)

                sl_dist = self.cfg.atr_sl_mult * self.entry_atr
                tp_dist = self.cfg.atr_tp_mult * self.entry_atr * self.cfg.tp_shrink

                if side == 1:
                    self.sl_price = max(1e-8, exec_price - sl_dist)
                    self.tp_price = max(1e-8, exec_price + tp_dist)
                else:
                    self.sl_price = max(1e-8, exec_price + sl_dist)
                    self.tp_price = max(1e-8, exec_price - tp_dist)

                self.break_even_done = False
                self.trail_active    = False

        # --------- BREAK-EVEN + TRAILING STOP ---------
        if (not manual_close and self.position != 0
                and self.current_size > 0 and self.entry_atr > 1e-8
                and self.idx > self.entry_idx):
            fav_price = high_bar if self.position == 1 else low_bar
            fav_move  = self.position * (fav_price - self.entry_price)

            # Break-even : déplace SL à l'entrée quand gain >= atr_be_mult × ATR
            if not self.break_even_done and fav_move >= self.cfg.atr_be_mult * self.entry_atr:
                if self.position == 1:
                    self.sl_price = max(self.sl_price, self.entry_price)
                else:
                    self.sl_price = min(self.sl_price, self.entry_price)
                self.break_even_done = True

            # Trailing stop : suit le prix à atr_trail_dist × ATR quand gain >= atr_trail_mult × ATR
            if fav_move >= self.cfg.atr_trail_mult * self.entry_atr:
                trail_sl = fav_price - self.position * self.cfg.atr_trail_dist * self.entry_atr
                if self.position == 1:
                    self.sl_price = max(self.sl_price, trail_sl)
                else:
                    self.sl_price = min(self.sl_price, trail_sl)
                self.trail_active = True

        # --------- SL/TP AUTO ---------
        if not manual_close and self.position != 0 and self.current_size > 0 and self.entry_price > 0 and self.idx > self.entry_idx:
            exit_price = None
            if self.position == 1:
                if self.sl_price > 0 and low_bar <= self.sl_price:
                    exit_price = self.sl_price
                    hit_sl = True
                elif self.tp_price > 0 and high_bar >= self.tp_price:
                    exit_price = self.tp_price
                    hit_tp = True
            else:
                if self.sl_price > 0 and high_bar >= self.sl_price:
                    exit_price = self.sl_price
                    hit_sl = True
                elif self.tp_price > 0 and low_bar <= self.tp_price:
                    exit_price = self.tp_price
                    hit_tp = True

            if exit_price is not None:
                exit_price = self._apply_micro(exit_price, -self.position, is_entry=False)
                pnl = self.position * (exit_price - self.entry_price) * self.current_size * self.cfg.leverage
                fee = self.cfg.fee_rate * exit_price * self.current_size
                realized = pnl - fee
                realized_trade = realized

                self.capital += realized
                self.last_realized_pnl = realized
                self.trades_pnl.append(realized)

                self.position = 0
                self.current_size = 0.0
                self.entry_price = 0.0
                self.sl_price = 0.0
                self.tp_price = 0.0
                self.entry_idx = -1
                self.entry_atr = 0.0
                self.risk_scale = 1.0
                self.last_risk_scale = 1.0
                self.break_even_done = False
                self.trail_active    = False

        # ==================================================================
        # REWARD SHAPING Ω — LONG + SHORT AVEC BONUS MOMENTUM CONFIRMÉ
        # ==================================================================
        latent = 0.0
        if self.position != 0 and self.current_size > 0 and self.entry_price > 0:
            price_for_pnl = low_bar if self.position == 1 else high_bar
            latent = self.position * (price_for_pnl - self.entry_price) * self.current_size * self.cfg.leverage

        equity = self.capital + latent
        equity_clamped = max(equity, 1e-8)
        prev_equity_clamped = max(prev_equity, 1e-8)
        log_ret = math.log(equity_clamped / prev_equity_clamped)

        # Base reward plus fort maintenant que le modèle devient plus mature
        reward = log_ret * 10.0

        # TP / SL
        if hit_tp:
            reward += 1.0
        if hit_sl:
            reward -= 0.5

        # Bonus réalisé gagnant
        if realized_trade > 0:
            reward += 0.4 + realized_trade * 0.3

        # Malus flat très léger
        move = abs(price - self.data.close[self.idx - 1]) / price
        if self.position == 0 and move > 0.00015:
            reward -= move * 2.5

        # Bonus latent doux
        if self.position != 0 and self.entry_price > 0:
            unrealized = self.position * (price - self.entry_price) / self.entry_price
            if unrealized * self.position > 0:
                reward += abs(unrealized) * 2.8

        # ------------------------------------------------------------------
        # Momentum-Confirmed Entry Bonus (LONG + SHORT)
        # ------------------------------------------------------------------
        if self.idx >= 10:
            # features normalisées à t-1
            mom_5_val    = self.data.features[self.idx - 1, FEATURE_COLS.index("mom_5")]
            rsi_ok_val   = self.data.features[self.idx - 1, FEATURE_COLS.index("rsi_ok")]
            high_vol_val = self.data.features[self.idx - 1, FEATURE_COLS.index("high_vol_regime")]

            just_opened_long  = (self.position == 1 and old_pos == 0)
            just_opened_short = (self.position == -1 and old_pos == 0)

            # LONG : momentum haussier + RSI sain + haut régime de vol
            if just_opened_long and mom_5_val > 0.5 and rsi_ok_val > 0.5 and high_vol_val > 0.5:
                reward += 1.8

            # SHORT : momentum baissier + RSI sain + haut régime de vol
            if just_opened_short and mom_5_val < 0.5 and rsi_ok_val > 0.5 and high_vol_val > 0.5:
                reward += 1.8

        # Holding penalty plus tôt et progressive
        if self.bars_in_position > 9:
            reward -= 1e-4 * (self.bars_in_position - 9) ** 1.5

        # Clip final
        if hasattr(self.cfg, "current_epoch") and self.cfg.current_epoch >= 10:
            reward = float(np.clip(reward, -0.7, 1.4))

        # ==================================================================
        # Finalisation
        # ==================================================================
        self.peak_capital = max(self.peak_capital, equity)
        dd = (self.peak_capital - equity) / (self.peak_capital + 1e-8)
        self.max_dd = max(self.max_dd, dd)

        if dd > self.cfg.max_drawdown:
            reward -= 2.0

        self.idx += 1
        done = (self.idx >= self.end_idx)
        done_reason = "episode_end" if done else None

        # Terminaison anticipée : max drawdown ou capital insuffisant
        if not done:
            if dd > self.cfg.max_drawdown:
                done = True
                done_reason = "max_drawdown"
            elif self.capital < self.cfg.initial_capital * self.cfg.min_capital_frac:
                done = True
                done_reason = "min_capital"

        obs = self._get_obs()

        return obs, float(reward), done, False, {
            "capital": self.capital,
            "drawdown": self.max_dd,
            "done_reason": done_reason,
            "position": self.position
        }


# ======================================================================
# SAINT v2 — SINGLE-HEAD (ACTOR + CRITIC)
# ======================================================================

class GatedFFN(nn.Module):
    def __init__(self, d: int, mult: int = 2, dropout: float = 0.05):
        super().__init__()
        inner = d * mult
        self.lin1 = nn.Linear(d, inner * 2)
        self.lin2 = nn.Linear(inner, d)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, gate = self.lin1(h).chunk(2, dim=-1)
        h = a * torch.sigmoid(gate)
        h = self.lin2(self.dropout(h))
        return x + h


class ColumnAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F, D = x.shape
        h = x.reshape(B * T, F, D)
        h2 = self.norm(h)
        out, _ = self.attn(h2, h2, h2)
        h = h + self.drop(out)
        return h.reshape(B, T, F, D)


class RowAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F, D = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * F, T, D)
        h2 = self.norm(h)
        out, _ = self.attn(h2, h2, h2)
        h = h + self.drop(out)
        h = h.reshape(B, F, T, D).permute(0, 2, 1, 3)
        return h


class SAINTv2Block(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float, mult: int):
        super().__init__()
        self.ra1 = RowAttention(d, heads, dropout)
        self.ff1 = GatedFFN(d, mult, dropout)

        self.ra2 = RowAttention(d, heads, dropout)
        self.ff2 = GatedFFN(d, mult, dropout)

        self.ca = ColumnAttention(d, heads, dropout)
        self.ff3 = GatedFFN(d, mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ra1(x)
        x = self.ff1(x)
        x = self.ra2(x)
        x = self.ff2(x)
        x = self.ca(x)
        x = self.ff3(x)
        return x


class SAINTPolicySingleHead(nn.Module):
    def __init__(
        self,
        n_features: int = OBS_N_FEATURES,
        d_model: int = 80,
        num_blocks: int = 2,
        heads: int = 4,
        dropout: float = 0.05,
        ff_mult: int = 2,
        max_len: int = 64,
        n_actions: int = N_ACTIONS,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_actions = n_actions

        self.input_proj = nn.Linear(1, d_model)
        self.scale = math.sqrt(d_model)
        self.row_emb = nn.Embedding(max_len, d_model)
        self.col_emb = nn.Embedding(n_features, d_model)

        self.blocks = nn.ModuleList([
            SAINTv2Block(d_model, heads, dropout, ff_mult)
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 3, f"Input x must be (B,T,F), got {x.shape}"
        B, T, F = x.shape

        tok = self.input_proj(x.unsqueeze(-1)) * self.scale

        rows = torch.arange(T, device=x.device).view(1, T, 1).expand(B, T, F)
        cols = torch.arange(F, device=x.device).view(1, 1, F).expand(B, T, F)

        tok = tok + self.row_emb(rows) + self.col_emb(cols)

        for blk in self.blocks:
            tok = blk(tok)

        h_time = tok.mean(dim=1)
        h_feat = tok.mean(dim=2)

        cls_time = h_time.mean(dim=1)
        cls_feat = h_feat.mean(dim=1)

        h = cls_time + cls_feat
        h = self.norm(h)
        h = self.mlp(h)

        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)

        return logits, value


# ======================================================================
# PPO UTILITAIRES
# ======================================================================

def get_device(cfg: PPOConfig):
    if cfg.force_cpu:
        print("CPU forcé.")
        return torch.device("cpu")
    if torch.cuda.is_available():
        print("CUDA détecté — utilisation GPU.")
        return torch.device("cuda")
    print("Pas de CUDA — utilisation CPU.")
    return torch.device("cpu")


class RewardNormalizer:
    """Normalise les rewards par l'écart-type glissant (algorithme de Welford)."""
    def __init__(self):
        self._mean = 0.0
        self._var  = 1.0
        self._count = 0

    def update(self, x: float):
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        self._var  += (delta * (x - self._mean) - self._var) / self._count

    @property
    def std(self) -> float:
        return max(math.sqrt(abs(self._var)), 1e-8)

    def normalize(self, x: float) -> float:
        return x / self.std


def compute_gae(rewards, values, dones, gamma, lam, last_value=0.0):
    values = values + [last_value]
    gae = 0.0
    adv = []

    for t in reversed(range(len(rewards))):
        mask = 1 - int(dones[t])
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv.insert(0, gae)

    returns = [adv[i] + values[i] for i in range(len(adv))]
    return adv, returns


def build_mask_from_pos_scalar(pos: int, device, side: str) -> torch.Tensor:
    mask = torch.zeros(N_ACTIONS, dtype=torch.bool, device=device)

    if side == "close":
        if pos == 0:
            mask[4] = True
        else:
            mask[0] = True
            mask[4] = True
        return mask

    if pos != 0:
        mask[4] = True
        return mask

    if side == "both":
        mask[:] = True
    elif side == "long":
        mask[0] = True
        mask[2] = True
        mask[4] = True
    elif side == "short":
        mask[1] = True
        mask[3] = True
        mask[4] = True
    else:
        mask[:] = True

    return mask


def build_action_mask_from_positions(positions: torch.Tensor, side: str) -> torch.Tensor:
    device = positions.device
    B = positions.shape[0]
    mask = torch.zeros(B, N_ACTIONS, dtype=torch.bool, device=device)

    flat = (positions == 0)
    inpos = ~flat

    if side == "close":
        if flat.any():
            mask[flat, 4] = True
        if inpos.any():
            mask[inpos, 0] = True
            mask[inpos, 4] = True
        return mask

    if flat.any():
        if side == "both":
            mask[flat] = True
        elif side == "long":
            mask[flat, 0] = True
            mask[flat, 2] = True
            mask[flat, 4] = True
        elif side == "short":
            mask[flat, 1] = True
            mask[flat, 3] = True
            mask[flat, 4] = True
        else:
            mask[flat] = True

    if inpos.any():
        mask[inpos, 4] = True

    return mask


def map_agent_action_to_env_action(
    a: int,
    pos: int,
    cfg: PPOConfig,
    device: torch.device,
    state_tensor: torch.Tensor,
    policy_long: Optional[nn.Module] = None,
    policy_short: Optional[nn.Module] = None,
    epoch: int = 1,
) -> Tuple[int, float]:
    env_action = 2
    risk_scale = 1.0

    if state_tensor.dim() == 2:
        state_tensor = state_tensor.unsqueeze(0)
    elif state_tensor.dim() == 3 and state_tensor.size(0) > 1:
        state_tensor = state_tensor[:1]
    elif state_tensor.dim() != 3:
        raise ValueError(f"state_tensor doit être (B,T,F) ou (T,F), reçu {state_tensor.shape}")

    # Mode CLOSE : utilisation des modèles gelés SI disponibles
    if cfg.side == "close":
        if pos == 0:
            if policy_long is None or policy_short is None:
                env_action = 2
                risk_scale = 1.0
            else:
                with torch.no_grad():
                    logits_long, _ = policy_long(state_tensor)
                    logits_long = logits_long[0]
                    mask_long = build_mask_from_pos_scalar(pos, device, "long")
                    logits_long_m = logits_long.masked_fill(~mask_long, MASK_VALUE)

                    logits_short, _ = policy_short(state_tensor)
                    logits_short = logits_short[0]
                    mask_short = build_mask_from_pos_scalar(pos, device, "short")
                    logits_short_m = logits_short.masked_fill(~mask_short, MASK_VALUE)

                    open_long_max = torch.maximum(logits_long_m[0], logits_long_m[2])
                    score_long = (open_long_max - logits_long_m[4]).item()

                    open_short_max = torch.maximum(logits_short_m[1], logits_short_m[3])
                    score_short = (open_short_max - logits_short_m[4]).item()

                    if score_long <= 0.0 and score_short <= 0.0:
                        env_action = 2
                        risk_scale = 1.0
                    else:
                        if score_long >= score_short:
                            chosen_side = "long"
                            chosen_logits = logits_long_m
                        else:
                            chosen_side = "short"
                            chosen_logits = logits_short_m

                        if chosen_side == "long":
                            if chosen_logits[0] >= chosen_logits[2]:
                                a_entry = 0
                            else:
                                a_entry = 2
                        else:
                            if chosen_logits[1] >= chosen_logits[3]:
                                a_entry = 1
                            else:
                                a_entry = 3

                        if a_entry in (0, 2):  # BUY
                            env_action = 0
                            risk_scale = 1.8 if a_entry == 2 else 1.0
                        elif a_entry in (1, 3):  # SELL
                            env_action = 1
                            risk_scale = 1.8 if a_entry == 3 else 1.0
                        else:
                            env_action = 2
                            risk_scale = 1.0
        else:
            if a == 0:
                env_action = 0
                risk_scale = 1.0
            else:
                env_action = 2
                risk_scale = 1.0

        if epoch <= 35:
            risk_scale = 1.0

        return env_action, risk_scale

    # Modes both / long / short
    if pos == 0:
        if a == 4:
            env_action = 2
            risk_scale = 1.0
        elif a in (0, 2):  # BUY
            env_action = 0
            risk_scale = 1.8 if a == 2 else 1.0
        elif a in (1, 3):  # SELL
            env_action = 1
            risk_scale = 1.8 if a == 3 else 1.0
        else:
            env_action = 2
            risk_scale = 1.0
    else:
        env_action = 2
        risk_scale = 1.0

    if epoch <= 35:
        risk_scale = 1.0

    return env_action, risk_scale


# ======================================================================
# TRAINING PPO (sur un split donné)
# ======================================================================

def run_training_on_split(
    train_data: MarketData,
    val_data: MarketData,
    test_data: MarketData,
    stats: Dict[str, np.ndarray],
    cfg: PPOConfig,
    suffix: str = ""
):
    import csv as _csv

    device = get_device(cfg)

    # CSV de métriques
    csv_path = f"training_log_{cfg.side}{suffix}.csv"
    _csv_fields = [
        "epoch",
        "train_pnl", "train_trades", "train_win", "train_loss",
        "train_totalW", "train_totalL", "train_avgW", "train_avgL", "train_nbL",
        "train_pf", "train_dd",
        "val_pnl", "val_trades", "val_win", "val_loss",
        "val_totalW", "val_totalL", "val_avgW", "val_avgL", "val_nbL",
        "val_pf", "val_dd",
        "sortino", "sortino30",
        "actor_loss", "critic_loss", "entropy", "kl",
        "buy_ratio", "sell_ratio", "hold_ratio",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as _f:
        _csv.DictWriter(_f, fieldnames=_csv_fields).writeheader()

    env = BTCTradingEnvDiscrete(train_data, cfg)
    val_env = BTCTradingEnvDiscrete(val_data, cfg)

    policy = SAINTPolicySingleHead(
        n_features=OBS_N_FEATURES,
        d_model=cfg.d_model,
        num_blocks=2,
        heads=4,
        dropout=0.05,
        ff_mult=2,
        max_len=cfg.lookback,
        n_actions=N_ACTIONS
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-8)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.05
    )

    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=(cfg.use_amp and device.type == "cuda")
    )

    best_path = f"best_{cfg.model_prefix}_{cfg.side}{suffix}.pth"
    best_profit_path = f"bestprofit_{cfg.model_prefix}_{cfg.side}{suffix}.pth"

    if os.path.exists(best_path):
        print(f"→ Chargement du modèle existant ({best_path}) pour continuation…")
        policy.load_state_dict(torch.load(best_path, map_location=device))

    # Modèles gelés LONG/SHORT pour le mode "close"
    policy_long = None
    policy_short = None
    if cfg.side == "close":
        if os.path.exists(BEST_MODEL_LONG_PATH) and os.path.exists(BEST_MODEL_SHORT_PATH):
            policy_long = SAINTPolicySingleHead(
                n_features=OBS_N_FEATURES,
                d_model=cfg.d_model,
                num_blocks=2,
                heads=4,
                dropout=0.05,
                ff_mult=2,
                max_len=cfg.lookback,
                n_actions=N_ACTIONS
            ).to(device)
            policy_long.load_state_dict(torch.load(BEST_MODEL_LONG_PATH, map_location=device))
            policy_long.eval()

            policy_short = SAINTPolicySingleHead(
                n_features=OBS_N_FEATURES,
                d_model=cfg.d_model,
                num_blocks=2,
                heads=4,
                dropout=0.05,
                ff_mult=2,
                max_len=cfg.lookback,
                n_actions=N_ACTIONS
            ).to(device)
            policy_short.load_state_dict(torch.load(BEST_MODEL_SHORT_PATH, map_location=device))
            policy_short.eval()

            print("[CLOSE] Modèles LONG & SHORT gelés chargés pour générer les entrées pendant l'entraînement.")
        else:
            print("[CLOSE] ATTENTION : modèles LONG/SHORT introuvables.")
            print("        → le mode CLOSE restera flat quand il est en dehors d'une position (HOLD en flat).")

    best_val_profit = -1e9
    best_metric = -1e9
    best_state = None
    epochs_no_improve = 0
    patience = 100
    metric_history: List[float] = []

    reward_normalizer = RewardNormalizer()

    for epoch in range(1, cfg.epochs + 1):
        cfg.current_epoch = epoch

        batch_states = []
        batch_actions = []
        batch_oldlog = []
        batch_adv = []
        batch_returns = []
        batch_values = []
        batch_positions = []

        total_reward_epoch = 0.0
        epoch_pnl = []
        epoch_dd = []
        epoch_trades_pnl: List[float] = []

        action_counts_env = np.zeros(3, dtype=np.int64)

        policy.train()

        # --------- collecte expériences ---------
        for ep in range(cfg.episodes_per_epoch):
            state, info = env.reset()
            done = False

            ep_states = []
            ep_actions = []
            ep_logprobs = []
            ep_rewards = []
            ep_values = []
            ep_dones = []
            ep_positions = []

            while not done:
                pos = info.get("position", 0)
                ep_positions.append(pos)

                s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits, value = policy(s_tensor)
                    logits = logits[0]

                    mask = build_mask_from_pos_scalar(pos, device, cfg.side)
                    logits_masked = logits.masked_fill(~mask, MASK_VALUE)

                    dist = Categorical(logits=logits_masked)

                    # ============ CURRICULUM D'OUVERTURE FORCÉE V2 ============
                    force_opening = False
                    chosen_action = None

                    if pos == 0:
                        if epoch <= 15:
                            force_prob = max(0.15, 0.92 - epoch * 0.05)
                        elif epoch <= 35:
                            force_prob = 0.25
                        else:
                            force_prob = 0.05

                        if np.random.rand() < force_prob:
                            force_opening = True
                            if cfg.side == "long":
                                chosen_action = 0 if np.random.rand() < 0.7 else 2
                            elif cfg.side == "short":
                                chosen_action = 1 if np.random.rand() < 0.8 else 3
                            else:
                                chosen_action = random.choice([0, 1, 2, 3])
                    # =======================================================

                    if force_opening and chosen_action is not None:
                        agent_action = torch.tensor(chosen_action, device=device, dtype=torch.long)
                        logprob = dist.log_prob(agent_action).squeeze()
                    else:
                        agent_action = dist.sample()
                        # Filtre confiance 70% : entrée uniquement si prob >= seuil
                        if pos == 0 and int(agent_action.item()) != 4:
                            probs = torch.softmax(logits_masked, dim=-1)
                            if probs[int(agent_action.item())].item() < CONF_THRESHOLD:
                                agent_action = torch.tensor(4, device=device, dtype=torch.long)
                        logprob = dist.log_prob(agent_action).squeeze()
                    a = int(agent_action.item())

                    env_action, risk_scale = map_agent_action_to_env_action(
                        a=a,
                        pos=pos,
                        cfg=cfg,
                        device=device,
                        state_tensor=s_tensor,
                        policy_long=policy_long,
                        policy_short=policy_short,
                        epoch=epoch,
                    )

                env.set_risk_scale(risk_scale)
                action_counts_env[env_action] += 1

                ns, reward, done, _, info = env.step(env_action)
                total_reward_epoch += reward
                reward_normalizer.update(reward)
                reward = reward_normalizer.normalize(reward)

                ep_rewards.append(reward)
                ep_states.append(state)
                ep_actions.append(a)
                ep_logprobs.append(logprob.detach())
                ep_values.append(value.item())
                ep_dones.append(done)

                state = ns

            epoch_dd.append(info["drawdown"])

            final_close = env.data.close[env.idx - 1]
            latent = 0.0
            if env.position != 0 and env.current_size > 0 and env.entry_price > 0:
                latent = (
                    env.position *
                    (final_close - env.entry_price) *
                    env.current_size *
                    env.cfg.leverage
                )
            final_equity = env.capital + latent
            epoch_pnl.append(final_equity - cfg.initial_capital)
            epoch_trades_pnl.extend(env.trades_pnl)

            if done and info.get("done_reason") == "max_drawdown":
                last_value = 0.0
            else:
                with torch.no_grad():
                    s_last = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    _, v_last = policy(s_last)
                    last_value = v_last.item()

            adv, ret = compute_gae(
                ep_rewards, ep_values, ep_dones,
                cfg.gamma, cfg.lambda_gae, last_value
            )

            batch_states.extend(ep_states)
            batch_actions.extend(ep_actions)
            batch_oldlog.extend(ep_logprobs)
            batch_adv.extend(adv)
            batch_returns.extend(ret)
            batch_values.extend(ep_values)
            batch_positions.extend(ep_positions)

        # --------- tenseurs batch ---------
        states_np = np.stack(batch_states, axis=0)
        states = torch.tensor(states_np, dtype=torch.float32, device=device)

        actions = torch.tensor(batch_actions, dtype=torch.long, device=device)
        oldlog = torch.stack(batch_oldlog).to(device).view(-1)
        advantages = torch.tensor(batch_adv, dtype=torch.float32, device=device)
        returns = torch.tensor(batch_returns, dtype=torch.float32, device=device)
        values_old = torch.tensor(batch_values, dtype=torch.float32, device=device)
        positions = torch.tensor(batch_positions, dtype=torch.long, device=device)

        assert states.size(0) == oldlog.size(0) == actions.size(0) == values_old.size(0) == positions.size(0)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.clamp(-10, 10)
        advantages = advantages * 1.5

        epoch_actor_loss = []
        epoch_critic_loss = []
        epoch_entropy = []
        epoch_kl = []

        n_samples = states.size(0)
        idx = np.arange(n_samples)

        for upd in range(cfg.updates_per_epoch):
            np.random.shuffle(idx)

            for start in range(0, n_samples, cfg.batch_size):
                end = start + cfg.batch_size
                ids = idx[start:end]

                sb = states[ids]
                ab = actions[ids]
                lb_old = oldlog[ids]
                adv_b = advantages[ids]
                ret_b = returns[ids]
                val_old = values_old[ids]
                pos_b = positions[ids]

                with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
                    logits, value = policy(sb)
                    mask_batch = build_action_mask_from_positions(pos_b, cfg.side)
                    logits_masked = logits.masked_fill(~mask_batch, MASK_VALUE)

                    dist = Categorical(logits=logits_masked)
                    new_log = dist.log_prob(ab)
                    entropy = dist.entropy().mean()

                    ratio = (new_log - lb_old).exp()
                    surr1 = adv_b * ratio
                    surr2 = adv_b * torch.clamp(
                        ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps
                    )
                    actor_loss = -torch.min(surr1, surr2).mean()

                    value_pred = value.squeeze(-1)
                    v_clipped = val_old + (value_pred - val_old).clamp(-0.2, 0.2)
                    unclipped_loss = (value_pred - ret_b).pow(2)
                    clipped_loss = (v_clipped - ret_b).pow(2)
                    critic_loss = torch.max(unclipped_loss, clipped_loss).mean()

                    # Décroissance linéaire de 3× → 0.3× sur 80 epochs
                    t = min((epoch - 1) / 80.0, 1.0)
                    entropy_coef_epoch = cfg.entropy_coef * (3.0 * (1 - t) + 0.3 * t)
                    entropy_bonus = entropy_coef_epoch * entropy

                    # Warmup critique : N premières epochs → critique seul
                    if epoch <= cfg.critic_warmup_epochs:
                        loss = cfg.value_coef * critic_loss
                    else:
                        loss = actor_loss + cfg.value_coef * critic_loss - entropy_bonus

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    approx_kl = (lb_old - new_log).mean().item()

                epoch_actor_loss.append(actor_loss.item())
                epoch_critic_loss.append(critic_loss.item())
                epoch_entropy.append(entropy.item())
                epoch_kl.append(approx_kl)

            if np.mean(epoch_kl) > 1.5 * cfg.target_kl:
                print(f"[PPO] Early stop KL, KL={np.mean(epoch_kl):.4f}")
                break

        scheduler.step()

        profit_epoch = float(sum(epoch_pnl))
        num_trades_epoch = len(epoch_trades_pnl)
        winrate_epoch = (
            float(np.mean([p > 0 for p in epoch_trades_pnl]))
            if num_trades_epoch > 0 else 0.0
        )
        max_dd_epoch = float(max(epoch_dd) if epoch_dd else 0.0)

        wins_train   = [p for p in epoch_trades_pnl if p > 0]
        losses_train = [p for p in epoch_trades_pnl if p <= 0]
        avg_win_train  = float(np.mean(wins_train))   if wins_train  else 0.0
        avg_loss_train = float(np.mean(losses_train)) if losses_train else 0.0
        num_loss_train = len(losses_train)
        total_profit_train  = float(sum(wins_train))
        total_loss_train    = float(sum(losses_train))
        profit_factor_train = total_profit_train / (abs(total_loss_train) + 1e-8)

        total_actions_env = int(action_counts_env.sum()) if action_counts_env.sum() > 0 else 1
        buy_count, sell_count, hold_count = action_counts_env
        buy_ratio = buy_count / total_actions_env
        sell_ratio = sell_count / total_actions_env
        hold_ratio = hold_count / total_actions_env

        # --------- validation (greedy) ---------
        policy.eval()
        val_pnl = []
        val_dd = []
        val_trades = []

        with torch.no_grad():
            for _ in range(2):
                s, info = val_env.reset()
                done = False
                while not done:
                    pos = info.get("position", 0)
                    st = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                    logits, _ = policy(st)
                    logits = logits[0]
                    mask = build_mask_from_pos_scalar(pos, device, cfg.side)
                    logits_masked = logits.masked_fill(~mask, MASK_VALUE)

                    a = int(logits_masked.argmax(dim=-1).item())
                    # Pas de filtre 70% en validation : on mesure la policy brute
                    # pour la sélection du meilleur modèle (le filtre est dans loup_live.py)

                    env_action, risk_scale = map_agent_action_to_env_action(
                        a=a,
                        pos=pos,
                        cfg=cfg,
                        device=device,
                        state_tensor=st,
                        policy_long=policy_long,
                        policy_short=policy_short,
                        epoch=epoch,
                    )
                    val_env.set_risk_scale(risk_scale)

                    ns, r, done, _, info = val_env.step(env_action)
                    s = ns

                if val_env.idx > 0:
                    final_close = val_env.data.close[val_env.idx - 1]
                else:
                    final_close = val_env.data.close[-1]

                latent = 0.0
                if val_env.position != 0 and val_env.current_size > 0 and val_env.entry_price > 0:
                    latent = (
                        val_env.position *
                        (final_close - val_env.entry_price) *
                        val_env.current_size *
                        val_env.cfg.leverage
                    )

                final_equity = val_env.capital + latent
                val_pnl.append(final_equity - cfg.initial_capital)
                val_dd.append(info["drawdown"])
                val_trades.extend(val_env.trades_pnl)

        val_profit = float(sum(val_pnl))
        val_max_dd = float(max(val_dd) if val_dd else 0.0)
        val_num_trades = len(val_trades)
        val_winrate = (
            float(np.mean([t > 0 for t in val_trades]))
            if val_num_trades > 0 else 0.0
        )

        wins_val   = [t for t in val_trades if t > 0]
        losses_val = [t for t in val_trades if t <= 0]
        avg_win_val  = float(np.mean(wins_val))   if wins_val  else 0.0
        avg_loss_val = float(np.mean(losses_val)) if losses_val else 0.0
        num_loss_val = len(losses_val)
        total_profit_val  = float(sum(wins_val))
        total_loss_val    = float(sum(losses_val))
        profit_factor_val = total_profit_val / (abs(total_loss_val) + 1e-8)

        if val_num_trades > 10:
            rets = np.array(val_trades, dtype=np.float32) / cfg.initial_capital
            mean_ret = float(rets.mean())
            downside = rets[rets < 0.0]
            downside_std = float(downside.std()) if downside.size > 0 else 1e-4
            sortino = mean_ret / (downside_std + 1e-8)
        else:
            sortino = 0.0

        metric = sortino
        metric_history.append(metric)
        if len(metric_history) >= 30:
            recent_metric = float(np.mean(metric_history[-30:]))
        else:
            recent_metric = float(np.mean(metric_history))

        train_loss_rate = (1 - winrate_epoch) if num_trades_epoch > 0 else 0.0
        val_loss_rate   = (1 - val_winrate)   if val_num_trades  > 0 else 0.0

        print(
            f"[{cfg.side.upper()}{suffix}][EPOCH {epoch:03d}] "
            f"TrainPNL={profit_epoch:+9.2f}$  "
            f"Trades={num_trades_epoch:4d}  "
            f"Win={winrate_epoch:5.1%}  Loss={train_loss_rate:5.1%}  "
            f"TotalW={total_profit_train:+8.2f}$  TotalL={total_loss_train:+8.2f}$  "
            f"AvgW={avg_win_train:+7.3f}$  AvgL={avg_loss_train:+7.3f}$  "
            f"NbL={num_loss_train:4d}  PF={profit_factor_train:.2f}  "
            f"DD={max_dd_epoch:5.1%}"
        )
        print(
            f"[{cfg.side.upper()}{suffix}][EPOCH {epoch:03d}]  "
            f"ValPNL={val_profit:+9.2f}$  "
            f"ValTrades={val_num_trades:4d}  "
            f"ValWin={val_winrate:5.1%}  ValLoss={val_loss_rate:5.1%}  "
            f"ValTotalW={total_profit_val:+8.2f}$  ValTotalL={total_loss_val:+8.2f}$  "
            f"ValAvgW={avg_win_val:+7.3f}$  ValAvgL={avg_loss_val:+7.3f}$  "
            f"ValNbL={num_loss_val:4d}  ValPF={profit_factor_val:.2f}  "
            f"ValDD={val_max_dd:5.1%}  "
            f"Sortino={metric:6.3f}  Sortino30={recent_metric:6.3f}  "
            f"ActorL={np.mean(epoch_actor_loss):.4f}  "
            f"CriticL={np.mean(epoch_critic_loss):.4f}  "
            f"Entropy={np.mean(epoch_entropy):.4f}  "
            f"KL={np.mean(epoch_kl):.4f}  "
            f"ENV B:{buy_ratio:4.1%} S:{sell_ratio:4.1%} H:{hold_ratio:4.1%}"
        )

        # Écriture CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as _f:
            _csv.DictWriter(_f, fieldnames=_csv_fields).writerow({
                "epoch": epoch,
                "train_pnl": round(profit_epoch, 4),
                "train_trades": num_trades_epoch,
                "train_win": round(winrate_epoch, 4),
                "train_loss": round(train_loss_rate, 4),
                "train_totalW": round(total_profit_train, 4),
                "train_totalL": round(total_loss_train, 4),
                "train_avgW": round(avg_win_train, 4),
                "train_avgL": round(avg_loss_train, 4),
                "train_nbL": num_loss_train,
                "train_pf": round(profit_factor_train, 4),
                "train_dd": round(max_dd_epoch, 4),
                "val_pnl": round(val_profit, 4),
                "val_trades": val_num_trades,
                "val_win": round(val_winrate, 4),
                "val_loss": round(val_loss_rate, 4),
                "val_totalW": round(total_profit_val, 4),
                "val_totalL": round(total_loss_val, 4),
                "val_avgW": round(avg_win_val, 4),
                "val_avgL": round(avg_loss_val, 4),
                "val_nbL": num_loss_val,
                "val_pf": round(profit_factor_val, 4),
                "val_dd": round(val_max_dd, 4),
                "sortino": round(metric, 6),
                "sortino30": round(recent_metric, 6),
                "actor_loss": round(float(np.mean(epoch_actor_loss)), 6),
                "critic_loss": round(float(np.mean(epoch_critic_loss)), 6),
                "entropy": round(float(np.mean(epoch_entropy)), 6),
                "kl": round(float(np.mean(epoch_kl)), 6),
                "buy_ratio": round(buy_ratio, 4),
                "sell_ratio": round(sell_ratio, 4),
                "hold_ratio": round(hold_ratio, 4),
            })

        # Best sur PNL (pour info) — nécessite un minimum de trades
        if val_num_trades >= cfg.min_val_trades_save and val_profit > best_val_profit:
            best_val_profit = val_profit
            state_profit = policy.state_dict().copy()
            torch.save(state_profit, best_profit_path)
            print(f"[{cfg.side.upper()}{suffix}][EPOCH {epoch:03d}] Nouveau best PROFIT (ValPNL={best_val_profit:.3f}, trades={val_num_trades}).")

        # Best réel pour live : Sortino30 + min trades en validation
        if val_num_trades >= cfg.min_val_trades_save and recent_metric > best_metric:
            best_metric = recent_metric
            best_state = policy.state_dict().copy()
            torch.save(best_state, best_path)
            epochs_no_improve = 0
            print(f"[{cfg.side.upper()}{suffix}][EPOCH {epoch:03d}] Nouveau best (Sortino30={recent_metric:.3f}, trades={val_num_trades}).")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{cfg.side.upper()}{suffix}] Early stopping après {epoch} epochs (Sortino rolling ne progresse plus).")
                break

    torch.save(policy.state_dict(), f"last_{cfg.model_prefix}_{cfg.side}{suffix}.pth")

    print(f"[{cfg.side.upper()}{suffix}] Entraînement terminé, passage en TEST…")

    if best_state is not None:
        policy.load_state_dict(best_state)
    policy.eval()

    test_env = BTCTradingEnvDiscrete(test_data, cfg)
    all_trades = []
    all_dd = []
    all_equity = []

    with torch.no_grad():
        for ep in range(5):
            s, info = test_env.reset()
            done = False
            while not done:
                pos = info.get("position", 0)
                st = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                logits, _ = policy(st)
                logits = logits[0]
                mask = build_mask_from_pos_scalar(pos, device, cfg.side)
                logits_masked = logits.masked_fill(~mask, MASK_VALUE)

                a = int(logits_masked.argmax(dim=-1).item())
                # Pas de filtre 70% en test : évaluation policy brute (filtre dans loup_live.py)

                env_action, risk_scale = map_agent_action_to_env_action(
                    a=a,
                    pos=pos,
                    cfg=cfg,
                    device=device,
                    state_tensor=st,
                    policy_long=policy_long,
                    policy_short=policy_short,
                    epoch=epoch,
                )
                test_env.set_risk_scale(risk_scale)

                ns, r, done, _, info = test_env.step(env_action)
                s = ns

            if test_env.idx > 0:
                final_close = test_env.data.close[test_env.idx - 1]
            else:
                final_close = test_env.data.close[-1]

            latent = 0.0
            if test_env.position != 0 and test_env.current_size > 0 and test_env.entry_price > 0:
                latent = (
                    test_env.position *
                    (final_close - test_env.entry_price) *
                    test_env.current_size *
                    test_env.cfg.leverage
                )

            final_equity = test_env.capital + latent

            dd_ep = info["drawdown"]
            all_dd.append(dd_ep)
            all_trades.extend(test_env.trades_pnl)
            all_equity.append(final_equity - cfg.initial_capital)

    test_profit = float(sum(all_equity))
    test_num_trades = len(all_trades)
    test_winrate = (
        float(np.mean([p > 0 for p in all_trades]))
        if test_num_trades > 0 else 0.0
    )
    test_max_dd = float(max(all_dd) if all_dd else 0.0)

    wins_test   = [p for p in all_trades if p > 0]
    losses_test = [p for p in all_trades if p <= 0]
    avg_win_test     = float(np.mean(wins_test))   if wins_test   else 0.0
    avg_loss_test    = float(np.mean(losses_test)) if losses_test else 0.0
    num_loss_test    = len(losses_test)
    total_profit_test = float(sum(wins_test))
    total_loss_test   = float(sum(losses_test))

    print(
        f"[{cfg.side.upper()}{suffix}][TEST] "
        f"NetPNL={test_profit:+9.2f}$  "
        f"Trades={test_num_trades}  "
        f"Win={test_winrate:2.0%}  Loss={1-test_winrate:2.0%}  "
        f"TotalW={total_profit_test:+8.2f}$  TotalL={total_loss_test:+8.2f}$  "
        f"AvgW={avg_win_test:+.3f}$  AvgL={avg_loss_test:+.3f}$  "
        f"NbLoss={num_loss_test}  "
        f"MaxDD={test_max_dd:.3f}"
    )

    print(f"[{cfg.side.upper()}{suffix}] Fin du split.")


# ======================================================================
# TRAINING SIMPLE (split 70/15/15)
# ======================================================================

def run_training_full(cfg: PPOConfig):
    df = load_mt5_data(cfg)

    if os.path.exists(NORM_STATS_PATH):
        stats = load_global_norm_stats()
    else:
        stats = compute_and_save_global_norm_stats(df, FEATURE_COLS)

    train_data, val_data, test_data = create_datasets(df, FEATURE_COLS, stats)
    run_training_on_split(train_data, val_data, test_data, stats, cfg, suffix="")


# ======================================================================
# WALK-FORWARD
# ======================================================================

def run_walkforward(
    cfg_base: PPOConfig,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    max_folds: int = 1
):
    assert train_frac + val_frac + test_frac <= 1.0 + 1e-6, "Les fractions ne peuvent pas dépasser 1.0"

    df_full = load_mt5_data(cfg_base)
    n = len(df_full)

    if os.path.exists(NORM_STATS_PATH):
        stats = load_global_norm_stats()
    else:
        stats = compute_and_save_global_norm_stats(df_full, FEATURE_COLS)

    train_len = int(n * train_frac)
    val_len   = int(n * val_frac)
    test_len  = int(n * test_frac)
    window_len = train_len + val_len + test_len

    if train_len <= 0 or val_len <= 0 or test_len <= 0 or window_len > n:
        raise ValueError("Longueurs de segments invalides (train/val/test) pour le walk-forward.")

    step = test_len

    print(f"\n=== WALK-FORWARD {cfg_base.side.upper()} ===")
    print(f"Total bars={n}, window_len={window_len}, "
          f"train={train_len}, val={val_len}, test={test_len}, step={step}")

    fold = 0
    start = 0
    while start + window_len <= n and fold < max_folds:
        fold += 1
        print(f"\n--- Fold {fold} : indices [{start} : {start + window_len}) ---")

        train_data, val_data, test_data = create_datasets_from_slices(
            df_full, FEATURE_COLS,
            start=start,
            train_len=train_len,
            val_len=val_len,
            test_len=test_len,
            stats=stats
        )

        cfg_fold = PPOConfig(**cfg_base.__dict__)
        cfg_fold.model_prefix = f"{cfg_base.model_prefix}_wf{fold}"

        suffix = f"_wf{fold}"
        run_training_on_split(train_data, val_data, test_data, stats, cfg_fold, suffix=suffix)

        start += step

    print(f"\n=== FIN WALK-FORWARD {cfg_base.side.upper()} (folds={fold}) ===")


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    cfg_base = PPOConfig()

    # LONG
    # cfg_long = PPOConfig(**cfg_base.__dict__)
    # cfg_long.side = "long"
    # cfg_long.model_prefix = "saintv2_loup_long"
    # # 0.55/0.15/0.10 → window=80% des données → 3 folds possibles avec step=10%
    # run_walkforward(cfg_long, train_frac=0.55, val_frac=0.15, test_frac=0.10, max_folds=3)

    # SHORT
    cfg_short = PPOConfig(**cfg_base.__dict__)
    cfg_short.side = "short"
    cfg_short.model_prefix = "saintv2_loup_short"
    # mêmes proportions que LONG pour cohérence des comparaisons
    run_walkforward(cfg_short, train_frac=0.55, val_frac=0.15, test_frac=0.10, max_folds=3)

    # CLOSE : après entraînement LONG/SHORT
    # cfg_close = PPOConfig(**cfg_base.__dict__)
    # cfg_close.side = "close"
    # cfg_close.model_prefix = "saintv2_loup_close"
    # run_training_full(cfg_close)
