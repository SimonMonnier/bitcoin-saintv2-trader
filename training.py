# ======================================================================
# PPO + SAINTv2 — SCALPING BTCUSD M1 (SINGLE-HEAD + ACTION MASK + H1)
# Version "Loup Ω" LONG / SHORT / CLOSE
# ======================================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
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
# LOG HELPERS — ANSI couleurs pour les prints d'entraînement
# ============================================================

try:
    import colorama
    colorama.just_fix_windows_console()
except ImportError:
    pass


class _C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[32m"
    RED     = "\033[31m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    GREY    = "\033[90m"
    WHITE   = "\033[97m"


def _col(text: str, color: str) -> str:
    return f"{color}{text}{_C.RESET}"


def _money(x: float, width: int = 9) -> str:
    """+1234.56$ ou -123.45$ coloré selon signe."""
    s = f"{x:+.2f}$".rjust(width)
    if x > 0:
        return _col(s, _C.GREEN)
    if x < 0:
        return _col(s, _C.RED)
    return _col(s, _C.GREY)


def _pct(x: float, width: int = 5) -> str:
    return f"{x*100:>{width-1}.1f}%"


def _split_by_side(trades_pnl, trades_side):
    """Renvoie (pnl_long, pnl_short) avec chacun (wins, losses, total_pnl)."""
    long_w  = [p for p, s in zip(trades_pnl, trades_side) if s == 1 and p > 0]
    long_l  = [p for p, s in zip(trades_pnl, trades_side) if s == 1 and p <= 0]
    short_w = [p for p, s in zip(trades_pnl, trades_side) if s == -1 and p > 0]
    short_l = [p for p, s in zip(trades_pnl, trades_side) if s == -1 and p <= 0]
    return {
        "long":  {"wins": long_w,  "losses": long_l,  "pnl": float(sum(long_w + long_l))},
        "short": {"wins": short_w, "losses": short_l, "pnl": float(sum(short_w + short_l))},
    }

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

# 0:BUY  1:SELL  2:HOLD
N_ACTIONS = 3
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
    # Fenêtre temporelle (UTC). Si date_to est None → maintenant.
    # Les barres en-dehors de [date_from, date_to] sont ignorées.
    # 2022-01-01 inclut bear Luna/FTX/2022 → équilibre haussier/baissier
    date_from: datetime = field(default_factory=lambda: datetime(2022, 1, 1))
    date_to: Optional[datetime] = None
    # n_bars est gardé en fallback uniquement si copy_rates_range échoue.
    # 2_500_000 bougies M1 ≈ 4.7 ans (avec weekends crypto = 24/7)
    n_bars: int = 2_500_000
    lookback: int = 25

    # PPO Training
    epochs: int = 240               # +50% vs 160 pour cosine LR plus douce
    episodes_per_epoch: int = 4     # +1 pour plus de diversité par epoch
    episode_length: int = 4000      # 2333 → 4000 (~2.8j de M1) : trajectoires plus longues
    updates_per_epoch: int = 2
    tp_shrink: float = 0.7

    batch_size: int = 256
    gamma: float = 0.97
    lambda_gae: float = 0.95
    clip_eps: float = 0.18
    lr: float = 3e-4
    target_kl: float = 0.03
    value_coef: float = 0.5
    # (B) Entropy plus fort : pousse la policy à explorer SHORT
    entropy_coef: float = 0.25
    # (A) Clip plus serré : empêche les gradient explosions fold 2
    max_grad_norm: float = 0.5

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

    # Spécialisation d'agent (mode "close" supprimé)
    # "both"  → BUY + SELL + HOLD
    # "long"  → seulement BUY1 / HOLD
    # "short" → seulement SELL1 / HOLD
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

def _fetch_paginated(symbol: str, timeframe: int,
                     date_from: datetime, date_to: datetime,
                     chunk: int = 100_000) -> Optional[np.ndarray]:
    """
    Récupère les bougies entre date_from et date_to en paginant par chunks
    de fin → début (copy_rates_from accepte un date_to + count).
    MT5 limite copy_rates_range quand le cache local est trop court.
    Cette fonction "remonte" pas à pas pour tirer l'historique manquant.
    """
    all_chunks = []
    cursor = date_to
    seen_oldest = None
    safety_iter = 0
    while safety_iter < 200:  # garde-fou : max 200 × chunk = 20M bougies
        safety_iter += 1
        rates = mt5.copy_rates_from(symbol, timeframe, cursor, chunk)
        if rates is None or len(rates) == 0:
            break

        # Filtrer ce qui est avant date_from (fin de la pagination)
        oldest_ts = int(rates[0]["time"])
        oldest_dt = datetime.utcfromtimestamp(oldest_ts)

        # On garde tout ce chunk pour l'instant, le filtrage final se fait après
        all_chunks.append(rates)

        # Critère d'arrêt 1 : on a dépassé date_from
        if oldest_dt <= date_from:
            break
        # Critère d'arrêt 2 : on n'avance plus
        if seen_oldest is not None and oldest_ts >= seen_oldest:
            break
        seen_oldest = oldest_ts

        # On positionne le curseur juste AVANT la bougie la plus ancienne
        cursor = oldest_dt - pd.Timedelta(seconds=1)

    if not all_chunks:
        return None

    # Concat + dédoublon + filtre date_from
    rates_all = np.concatenate(all_chunks)
    rates_all = np.unique(rates_all)  # numpy structuré : trie + dédoublonne
    ts_from = int(date_from.timestamp())
    ts_to   = int(date_to.timestamp())
    rates_all = rates_all[(rates_all["time"] >= ts_from) & (rates_all["time"] <= ts_to)]
    return rates_all


def load_mt5_data(cfg: PPOConfig) -> pd.DataFrame:
    print("Connexion MT5…")
    if not mt5.initialize():
        raise RuntimeError("Erreur MT5.init()")

    date_from = cfg.date_from
    date_to = cfg.date_to or datetime.now()
    print(
        f"[DATA] Plage demandée : {date_from:%Y-%m-%d %H:%M} → "
        f"{date_to:%Y-%m-%d %H:%M}"
    )

    # Force MT5 à charger le symbol dans le Market Watch
    mt5.symbol_select(cfg.symbol, True)

    # 1) Tentative directe avec copy_rates_range (rapide quand cache OK)
    rates_m1 = mt5.copy_rates_range(cfg.symbol, cfg.timeframe, date_from, date_to)
    rates_h1 = mt5.copy_rates_range(cfg.symbol, cfg.htf_timeframe, date_from, date_to)

    n_m1_target = int((date_to - date_from).total_seconds() // 60 * 0.7)  # 70% (weekends, gaps)

    # 2) Si insuffisant → pagination par copy_rates_from
    if rates_m1 is None or len(rates_m1) < n_m1_target:
        nb = 0 if rates_m1 is None else len(rates_m1)
        print(f"[DATA] copy_rates_range M1 insuffisant ({nb:,} bougies), pagination en cours…")
        rates_m1 = _fetch_paginated(cfg.symbol, cfg.timeframe, date_from, date_to, chunk=100_000)

    if rates_h1 is None or len(rates_h1) < 100:
        nb = 0 if rates_h1 is None else len(rates_h1)
        print(f"[DATA] copy_rates_range H1 insuffisant ({nb:,} bougies), pagination en cours…")
        rates_h1 = _fetch_paginated(cfg.symbol, cfg.htf_timeframe, date_from, date_to, chunk=20_000)

    # 3) Dernier fallback : copy_rates_from_pos avec n_bars (peut être petit)
    if rates_m1 is None or len(rates_m1) == 0:
        print(f"[DATA] Pagination M1 vide, dernier fallback copy_rates_from_pos(0, {cfg.n_bars}).")
        rates_m1 = mt5.copy_rates_from_pos(cfg.symbol, cfg.timeframe, 0, cfg.n_bars)
    if rates_h1 is None or len(rates_h1) == 0:
        n_h1 = max(cfg.n_bars // 5, 5000)
        print(f"[DATA] Pagination H1 vide, dernier fallback copy_rates_from_pos(0, {n_h1}).")
        rates_h1 = mt5.copy_rates_from_pos(cfg.symbol, cfg.htf_timeframe, 0, n_h1)

    mt5.shutdown()

    if rates_m1 is None or rates_h1 is None:
        raise RuntimeError("MT5 n'a renvoyé aucune donnée M1 ou H1")

    print(f"[DATA] M1 bougies brutes : {len(rates_m1):,}  |  H1 : {len(rates_h1):,}")

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
        # Side (+1 long, -1 short) du trade fermé — index aligné avec trades_pnl
        self.trades_side: List[int] = []

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

        # Plus de mode "close" : fermeture uniquement par SL/TP/break-even/trailing
        manual_close = False

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
                self.trades_side.append(int(self.position))

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
    # 0=BUY  1=SELL  2=HOLD
    mask = torch.zeros(N_ACTIONS, dtype=torch.bool, device=device)

    if pos != 0:
        mask[2] = True  # En position : seulement HOLD (fermeture par SL/TP/trailing)
        return mask

    if side == "both":
        mask[0] = True   # BUY
        mask[1] = True   # SELL
        mask[2] = True   # HOLD
    elif side == "long":
        mask[0] = True
        mask[2] = True
    elif side == "short":
        mask[1] = True
        mask[2] = True
    else:
        mask[0] = True
        mask[1] = True
        mask[2] = True

    return mask


def build_action_mask_from_positions(positions: torch.Tensor, side: str) -> torch.Tensor:
    device = positions.device
    B = positions.shape[0]
    mask = torch.zeros(B, N_ACTIONS, dtype=torch.bool, device=device)

    flat = (positions == 0)
    inpos = ~flat

    # 0=BUY  1=SELL  2=HOLD
    if flat.any():
        if side == "both":
            mask[flat, 0] = True
            mask[flat, 1] = True
            mask[flat, 2] = True
        elif side == "long":
            mask[flat, 0] = True
            mask[flat, 2] = True
        elif side == "short":
            mask[flat, 1] = True
            mask[flat, 2] = True
        else:
            mask[flat, 0] = True
            mask[flat, 1] = True
            mask[flat, 2] = True

    if inpos.any():
        mask[inpos, 2] = True  # HOLD = nouvelle index 2

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

    # Mode "close" supprimé : seuls les sides "both" / "long" / "short" sont supportés.
    # Modes both / long / short — espace agent réduit à 3 actions
    #   a=0 → BUY     (env_action=0)
    #   a=1 → SELL    (env_action=1)
    #   a=2 → HOLD    (env_action=2)
    if pos == 0:
        if a == 0:    # BUY
            env_action = 0
            risk_scale = 1.0
        elif a == 1:  # SELL
            env_action = 1
            risk_scale = 1.0
        else:         # HOLD ou tout autre
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
        # Split LONG / SHORT côté training
        "train_long_w", "train_long_l", "train_long_pnl",
        "train_short_w", "train_short_l", "train_short_pnl",
        "val_pnl", "val_trades", "val_win", "val_loss",
        "val_totalW", "val_totalL", "val_avgW", "val_avgL", "val_nbL",
        "val_pf", "val_dd",
        # Split LONG / SHORT côté validation
        "val_long_w", "val_long_l", "val_long_pnl",
        "val_short_w", "val_short_l", "val_short_pnl",
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

    # Mode "close" supprimé : pas de modèles gelés à charger.
    policy_long = None
    policy_short = None

    best_val_profit = -1e9
    best_metric = -1e9
    best_state = None
    epochs_no_improve = 0
    # patience proportionnelle au nombre d'epochs (60% du total)
    patience = max(100, int(cfg.epochs * 0.6))
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
        epoch_trades_side: List[int] = []  # +1 long, -1 short — index aligné avec epoch_trades_pnl

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
                            # Plus de niveaux x1.8 : on force uniquement BUY1 (0) ou SELL1 (1)
                            if cfg.side == "long":
                                chosen_action = 0
                            elif cfg.side == "short":
                                chosen_action = 1
                            else:
                                # (C) Biais SHORT pour contrer la dérive long-only :
                                # marché 2022-2026 globalement haussier → les SELL forcés
                                # perdent souvent, ce qui a fait apprendre au modèle à ne plus
                                # shorter. On compense en sur-représentant les SELL au curriculum.
                                #   5/7 SELL (1) / 2/7 BUY (0)
                                chosen_action = random.choice([1, 1, 1, 1, 1, 0, 0])
                    # =======================================================

                    if force_opening and chosen_action is not None:
                        agent_action = torch.tensor(chosen_action, device=device, dtype=torch.long)
                        logprob = dist.log_prob(agent_action).squeeze()
                    else:
                        agent_action = dist.sample()
                        # Filtre confiance : entrée uniquement si prob >= seuil
                        # (HOLD = action 2 dans la convention 3-actions)
                        if pos == 0 and int(agent_action.item()) != 2:
                            probs = torch.softmax(logits_masked, dim=-1)
                            if probs[int(agent_action.item())].item() < CONF_THRESHOLD:
                                agent_action = torch.tensor(2, device=device, dtype=torch.long)
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
            epoch_trades_side.extend(env.trades_side)

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
                print(f"  {_col('⚠ early-stop KL', _C.YELLOW)}  KL={np.mean(epoch_kl):.4f}")
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
        val_trades_side: List[int] = []

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
                val_trades_side.extend(val_env.trades_side)

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

        # --------- Split LONG / SHORT ---------
        train_split = _split_by_side(epoch_trades_pnl, epoch_trades_side)
        val_split   = _split_by_side(val_trades,       val_trades_side)

        train_long_w  = len(train_split["long"]["wins"])
        train_long_l  = len(train_split["long"]["losses"])
        train_short_w = len(train_split["short"]["wins"])
        train_short_l = len(train_split["short"]["losses"])
        train_long_pnl  = train_split["long"]["pnl"]
        train_short_pnl = train_split["short"]["pnl"]

        val_long_w  = len(val_split["long"]["wins"])
        val_long_l  = len(val_split["long"]["losses"])
        val_short_w = len(val_split["short"]["wins"])
        val_short_l = len(val_split["short"]["losses"])
        val_long_pnl  = val_split["long"]["pnl"]
        val_short_pnl = val_split["short"]["pnl"]

        # --------- Couleurs / styles ---------
        tag       = _col(f"[{cfg.side.upper()}{suffix}]", _C.MAGENTA + _C.BOLD)
        epoch_str = _col(f"EPOCH {epoch:03d}", _C.CYAN + _C.BOLD)

        # Couleur du Sortino30 selon valeur
        s30 = recent_metric
        if s30 >= 0.05:
            s30_col = _C.GREEN + _C.BOLD
        elif s30 >= 0.0:
            s30_col = _C.GREEN
        elif s30 >= -0.05:
            s30_col = _C.YELLOW
        else:
            s30_col = _C.RED

        # Couleur du DD
        def _dd_col(d):
            if d < 0.2:   return _C.GREEN
            if d < 0.5:   return _C.YELLOW
            return _C.RED

        # ----- Ligne 1 : TRAIN -----
        print(
            f"{tag} {epoch_str}  "
            f"{_col('TRAIN', _C.WHITE + _C.BOLD)}  "
            f"PNL {_money(profit_epoch, width=10)}  "
            f"trades={num_trades_epoch:>4d}  "
            f"WR {_pct(winrate_epoch)}  "
            f"PF {profit_factor_train:>4.2f}  "
            f"DD {_col(_pct(max_dd_epoch), _dd_col(max_dd_epoch))}  "
            f"{_col('L', _C.GREEN)}({_col(str(train_long_w), _C.GREEN)}W/"
            f"{_col(str(train_long_l), _C.RED)}L) {_money(train_long_pnl, width=9)}  "
            f"{_col('S', _C.BLUE)}({_col(str(train_short_w), _C.GREEN)}W/"
            f"{_col(str(train_short_l), _C.RED)}L) {_money(train_short_pnl, width=9)}"
        )

        # ----- Ligne 2 : VAL -----
        print(
            f"{tag} {epoch_str}  "
            f"{_col('VAL  ', _C.WHITE + _C.BOLD)}  "
            f"PNL {_money(val_profit, width=10)}  "
            f"trades={val_num_trades:>4d}  "
            f"WR {_pct(val_winrate)}  "
            f"PF {profit_factor_val:>4.2f}  "
            f"DD {_col(_pct(val_max_dd), _dd_col(val_max_dd))}  "
            f"{_col('L', _C.GREEN)}({_col(str(val_long_w), _C.GREEN)}W/"
            f"{_col(str(val_long_l), _C.RED)}L) {_money(val_long_pnl, width=9)}  "
            f"{_col('S', _C.BLUE)}({_col(str(val_short_w), _C.GREEN)}W/"
            f"{_col(str(val_short_l), _C.RED)}L) {_money(val_short_pnl, width=9)}"
        )

        # ----- Ligne 3 : METRICS PPO -----
        print(
            f"{tag} {epoch_str}  "
            f"{_col('META ', _C.GREY + _C.BOLD)}  "
            f"Sortino {metric:>+6.3f}  "
            f"{_col(f'Sortino30 {s30:>+6.3f}', s30_col)}  "
            f"AvgW {_money(avg_win_train, width=8)}  AvgL {_money(avg_loss_train, width=8)}  "
            f"ActorL {np.mean(epoch_actor_loss):>+7.4f}  "
            f"CriticL {np.mean(epoch_critic_loss):>7.4f}  "
            f"H {np.mean(epoch_entropy):>5.3f}  "
            f"KL {np.mean(epoch_kl):>+6.4f}  "
            f"ENV [{_col(f'B {buy_ratio:>4.1%}', _C.GREEN)} "
            f"{_col(f'S {sell_ratio:>4.1%}', _C.BLUE)} "
            f"{_col(f'H {hold_ratio:>4.1%}', _C.GREY)}]"
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
                "train_long_w": train_long_w,
                "train_long_l": train_long_l,
                "train_long_pnl": round(train_long_pnl, 4),
                "train_short_w": train_short_w,
                "train_short_l": train_short_l,
                "train_short_pnl": round(train_short_pnl, 4),
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
                "val_long_w": val_long_w,
                "val_long_l": val_long_l,
                "val_long_pnl": round(val_long_pnl, 4),
                "val_short_w": val_short_w,
                "val_short_l": val_short_l,
                "val_short_pnl": round(val_short_pnl, 4),
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
            print(
                f"  {_col('★', _C.YELLOW + _C.BOLD)} "
                f"{_col(f'NEW BEST PROFIT', _C.YELLOW + _C.BOLD)}  "
                f"ValPNL={_money(best_val_profit, width=10)}  trades={val_num_trades}"
            )

        # Best réel pour live : Sortino30 + min trades en validation
        if val_num_trades >= cfg.min_val_trades_save and recent_metric > best_metric:
            best_metric = recent_metric
            best_state = policy.state_dict().copy()
            torch.save(best_state, best_path)
            epochs_no_improve = 0
            print(
                f"  {_col('★', _C.MAGENTA + _C.BOLD)} "
                f"{_col(f'NEW BEST SORTINO30', _C.MAGENTA + _C.BOLD)}  "
                f"Sortino30={recent_metric:+.3f}  trades={val_num_trades}"
            )
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

    # =======================================================
    # PIPELINE DUEL : un seul modèle qui décide BUY ou SELL
    # =======================================================
    # side="both"  → le masque d'actions autorise BUY1, SELL1, BUY1.8,
    #                SELL1.8 et HOLD quand le modèle est flat.
    # Le curriculum force aléatoirement BUY ou SELL au tirage (équiprobable
    # entre 0,1,2,3) pour amorcer les deux directions.
    # Résultat : un unique fichier .pth qui remplace LONG + SHORT,
    # exactement comme en live "duel".
    print("\n" + "=" * 70)
    print("  ENTRAÎNEMENT DUEL (long + short combinés)")
    print("=" * 70)
    cfg_duel = PPOConfig(**cfg_base.__dict__)
    cfg_duel.side = "both"
    cfg_duel.model_prefix = "saintv2_loup_duel"
    run_walkforward(cfg_duel, train_frac=0.55, val_frac=0.15, test_frac=0.10, max_folds=3)

    print("\n" + "=" * 70)
    print("  PIPELINE TERMINÉ : modèle duel entraîné.")
    print("  Fichiers générés :")
    print("    best_saintv2_loup_duel_both_wf1.pth")
    print("    bestprofit_saintv2_loup_duel_both_wf1.pth")
    print("=" * 70)

    # ---------------------------------------------------------
    # Alternative : entraîner LONG et SHORT séparément
    # (à dé-commenter si tu veux des modèles spécialisés au lieu du duel)
    # ---------------------------------------------------------
    # cfg_long = PPOConfig(**cfg_base.__dict__)
    # cfg_long.side = "long"
    # cfg_long.model_prefix = "saintv2_loup_long"
    # run_walkforward(cfg_long, train_frac=0.55, val_frac=0.15, test_frac=0.10, max_folds=3)
    #
    # cfg_short = PPOConfig(**cfg_base.__dict__)
    # cfg_short.side = "short"
    # cfg_short.model_prefix = "saintv2_loup_short"
    # run_walkforward(cfg_short, train_frac=0.55, val_frac=0.15, test_frac=0.10, max_folds=3)

    # (Mode "close" retiré : l'agent ne ferme plus manuellement, uniquement via SL/TP/trailing)
