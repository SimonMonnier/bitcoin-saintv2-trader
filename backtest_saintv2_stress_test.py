import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import time as _time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# LOG HELPERS — couleurs ANSI + formattage uniforme
# ============================================================

# Activation des codes ANSI sur Windows (PowerShell/cmd modernes les supportent)
try:
    import colorama
    colorama.just_fix_windows_console()
except ImportError:
    pass

class C:
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

def _c(text: str, color: str) -> str:
    return f"{color}{text}{C.RESET}"

def fmt_money(x: float, width: int = 10) -> str:
    s = f"{x:+.2f}$"
    s = s.rjust(width)
    return _c(s, C.GREEN if x > 0 else (C.RED if x < 0 else C.GREY))

def fmt_pct(x: float, width: int = 6) -> str:
    return f"{x*100:>{width-1}.1f}%"

def fmt_time(t) -> str:
    if isinstance(t, pd.Timestamp):
        return t.strftime("%Y-%m-%d %H:%M")
    return str(t)

def hr(char: str = "─", n: int = 78) -> str:
    return _c(char * n, C.GREY)

def banner(title: str, color: str = C.CYAN) -> str:
    line = "═" * 78
    return f"{_c(line, color)}\n  {_c(title, color + C.BOLD)}\n{_c(line, color)}"

# ============================================================
# BACKTEST ALIGNÉ AVEC training.py + loup_live.py (Loup Ω)
# ------------------------------------------------------------
#   - Pas de filtre de confiance (pure argmax comme en live)
#   - FEATURE_COLS / N_POS_FEATURES identiques au training
#   - Architecture SAINTv2 identique (d_model=80, blocks=2, heads=4)
#   - Modes side : "long" / "short" / "duel"
# ============================================================

# 0:BUY  1:SELL  2:HOLD
N_ACTIONS = 3
MASK_VALUE = -1e4  # même valeur que le training / live

# Stats de normalisation (doit correspondre à ton training)
NORM_STATS_PATH = "norm_stats_ohlc_indics.npz"

# Modèles pré-entraînés (best PROFIT ici) — mêmes noms que ton training
# Pattern training : f"bestprofit_{cfg.model_prefix}_{cfg.side}{suffix}.pth"
#   model_prefix LONG  = "saintv2_loup_long"   side="long"  suffix="_wf1"
#   model_prefix SHORT = "saintv2_loup_short"  side="short" suffix="_wf1"
BEST_MODEL_LONG_PATH = "bestprofit_saintv2_loup_long_wf1_long_wf1.pth"
BEST_MODEL_SHORT_PATH = "bestprofit_saintv2_loup_short_wf1_short_wf1.pth"
# Modèle unifié (training side="both")
BEST_MODEL_DUEL_PATH = "bestprofit_saintv2_loup_duel_wf1_both_wf1.pth"


@dataclass
class LiveConfig:
    symbol: str = "BTCUSD"
    timeframe: int = mt5.TIMEFRAME_M1
    htf_timeframe: int = mt5.TIMEFRAME_H1  # identique au training / live

    lookback: int = 25

    # nombre de bougies à charger pour le backtest
    n_bars_m1: int = 800_000
    n_bars_h1: int = 200_000

    # même params que l'env / live
    tp_shrink: float = 0.7

    # trading
    initial_capital: float = 1000.0
    position_size: float = 0.01    # utilisé uniquement si dynamic_volume = False
    leverage: float = 6.0
    fee_rate: float = 0.0004
    atr_sl_mult: float = 1.2
    atr_tp_mult: float = 2.4

    # Volume dynamique : aligné avec loup_live.compute_dynamic_volume
    dynamic_volume: bool = True
    max_lot: float = 100.0

    # valeurs "moyennes" utilisées comme base avant stress
    spread_bps: float = 0.0002    # 20 bps de spread "moyen"
    slippage_bps: float = 0.0000  # on laisse à 0, le stress-test gère le slippage random

    # ======= BREAK-EVEN + TRAILING (en ATR) - comme en live =======
    breakeven_atr_mult: float = 1.0       # mouvement favorable pour BE
    trailing_start_atr_mult: float = 1.5  # mouvement favorable pour activer le trailing
    trailing_dist_atr_mult: float = 1.0   # distance du trailing en ATR

    # Backtest : marge mini en points si broker renvoie 0
    backtest_min_extra_points: int = 100

    # ======= Seuil de confiance minimal pour ouvrir un trade =======
    # 0.90 = aligné avec CONF_THRESHOLD (training) et loup_live.min_confidence.
    # Mettre 0.0 pour pure argmax si tu veux mesurer la policy brute.
    min_confidence: float = 0.90

    # device
    force_cpu: bool = False

    # mode d’agent :
    #   "both"  : modèle unifié duel (1 seul .pth, décide BUY/SELL/HOLD)
    #   "duel"  : long vs short (2 modèles séparés, arbitrage par max prob)
    #   "long"  : long only     (utilise bestprofit_long)
    #   "short" : short only    (utilise bestprofit_short)
    side: str = "both"

    # fréquence d'affichage de progression (en nombre de bougies M1)
    progress_interval_bars: int = 1440  # ~ 1 jour

    date_from: datetime = datetime(2026, 1, 1)
    date_to: Optional[datetime] = None


@dataclass
class BTState:
    capital: float
    equity: float
    position: int = 0          # 0, +1, -1
    volume: float = 0.0
    entry_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    entry_index: int = -1
    entry_atr: float = 0.0     # ATR au moment de l'entrée (pour unrealized_atr)
    last_risk_scale: float = 1.0
    trades_pnl: List[float] = field(default_factory=list)
    trades_meta: List[Dict] = field(default_factory=list)
    max_equity: float = 0.0


# ============================================================
# STRESS-TEST V3 (INSTITUTIONNEL)
# ============================================================

@dataclass
class StressConfig:
    enable: bool = True

    # slippage random ±20 bps
    max_slippage_bps: float = 0.0002

    # bruit des ticks (gaussien, appliqué sur OHLC)
    tick_noise_std: float = 0.00005  # 5 bps std

    # distorsion de l’ATR (erreur d’estimation)
    atr_distortion: float = 0.10  # ±10 %

    # randomisation TP/SL ±10 %
    tp_sl_random: float = 0.10

    # micro-gaps (saut de prix sur une bougie)
    micro_gap_prob: float = 0.0005
    micro_gap_jump_std: float = 0.0005  # ~5 bps de gap moyen

    # news spikes simulés (élargissement brutal de la range)
    news_spike_prob: float = 0.0005

    # trous aléatoires de 1–3 minutes (on saute complètement des barres)
    hole_prob: float = 0.0005


def safe_normalize(X, stats, clip_sigma=5.0):
    z = (X - stats["mean"]) / (stats["std"] + 1e-8)
    # On clip tout à ±5σ → le modèle reste dans sa zone de confort
    z = np.clip(z, -clip_sigma, clip_sigma)
    return z

# ============================================================
# INDICATEURS — IDENTIQUES TRAINING / LIVE ("Loup Ω")
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


FEATURE_COLS_M1 = [
    "open", "high", "low", "close",
    "rsi_14",
    "returns", "vol_20", "range_norm",
    "mom_5", "rsi_ok", "high_vol_regime",
]

FEATURE_COLS_H1 = [
    "close_h1",
    "rsi_14_h1",
    "returns_h1", "vol_20_h1", "range_norm_h1",
]

FEATURE_COLS = FEATURE_COLS_M1 + FEATURE_COLS_H1
N_BASE_FEATURES = len(FEATURE_COLS)

# Embedding de position IDENTIQUE training/live :
#   - position (-1, 0, 1)
#   - unrealized_atr  (PnL latent normalisé par l'ATR d'entrée)
#   - bars_held_norm  (durée détention / scalping_max_holding, capée à 3)
#   - last_risk_scale
N_POS_FEATURES = 4
OBS_N_FEATURES = N_BASE_FEATURES + N_POS_FEATURES


# ============================================================
# MODELE SAINTv2 — IDENTIQUE TRAINING / LIVE
# ============================================================

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


# ============================================================
# UTILS NORMA / DEVICE
# ============================================================

def get_device(cfg: LiveConfig):
    if cfg.force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        print("CUDA détecté — utilisation GPU.")
        return torch.device("cuda")
    print("Pas de CUDA — utilisation CPU.")
    return torch.device("cpu")


def load_norm_stats(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stats de normalisation introuvables : {path}")
    data = np.load(path)
    return {"mean": data["mean"], "std": data["std"]}


def normalize_features(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    mean, std = stats["mean"], stats["std"]
    std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std


# ============================================================
# DATA M1 + H1 POUR BACKTEST (MT5)
# ============================================================

def _fetch_paginated(symbol: str, timeframe: int,
                     date_from: datetime, date_to: datetime,
                     chunk: int = 100_000) -> Optional[np.ndarray]:
    """Pagination MT5 (identique à training._fetch_paginated)."""
    all_chunks = []
    cursor = date_to
    seen_oldest = None
    safety_iter = 0
    while safety_iter < 200:
        safety_iter += 1
        rates = mt5.copy_rates_from(symbol, timeframe, cursor, chunk)
        if rates is None or len(rates) == 0:
            break
        oldest_ts = int(rates[0]["time"])
        oldest_dt = datetime.utcfromtimestamp(oldest_ts)
        all_chunks.append(rates)
        if oldest_dt <= date_from:
            break
        if seen_oldest is not None and oldest_ts >= seen_oldest:
            break
        seen_oldest = oldest_ts
        cursor = oldest_dt - pd.Timedelta(seconds=1)
    if not all_chunks:
        return None
    rates_all = np.concatenate(all_chunks)
    rates_all = np.unique(rates_all)
    ts_from = int(date_from.timestamp())
    ts_to   = int(date_to.timestamp())
    rates_all = rates_all[(rates_all["time"] >= ts_from) & (rates_all["time"] <= ts_to)]
    return rates_all


def fetch_ohlc_with_indicators(cfg: LiveConfig) -> pd.DataFrame:
    utc_from = cfg.date_from
    utc_to = cfg.date_to or datetime.now()

    mt5.symbol_select(cfg.symbol, True)

    print(f"  → fetch M1 {utc_from:%Y-%m-%d} → {utc_to:%Y-%m-%d}…")
    rates_m1 = mt5.copy_rates_range(cfg.symbol, cfg.timeframe, utc_from, utc_to)
    rates_h1 = mt5.copy_rates_range(cfg.symbol, cfg.htf_timeframe, utc_from, utc_to)

    n_m1_target = int((utc_to - utc_from).total_seconds() // 60 * 0.7)
    n_h1_target = int((utc_to - utc_from).total_seconds() // 3600 * 0.7)

    if rates_m1 is None or len(rates_m1) < n_m1_target:
        nb = 0 if rates_m1 is None else len(rates_m1)
        print(f"  ⚠ copy_rates_range M1 insuffisant ({nb:,} / {n_m1_target:,}), pagination…")
        rates_m1 = _fetch_paginated(cfg.symbol, cfg.timeframe, utc_from, utc_to, chunk=100_000)

    if rates_h1 is None or len(rates_h1) < n_h1_target:
        nb = 0 if rates_h1 is None else len(rates_h1)
        print(f"  ⚠ copy_rates_range H1 insuffisant ({nb:,} / {n_h1_target:,}), pagination…")
        rates_h1 = _fetch_paginated(cfg.symbol, cfg.htf_timeframe, utc_from, utc_to, chunk=20_000)

    if rates_m1 is None or rates_h1 is None:
        raise RuntimeError("MT5 n'a renvoyé aucune donnée M1 ou H1.")

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
    return merged


def build_live_obs(
    df_merged: pd.DataFrame,
    stats: Dict[str, np.ndarray],
    cfg: LiveConfig,
    pos: int,
    entry_price: float,
    entry_atr: float,
    bars_in_position: int,
    last_risk_scale: float,
) -> Optional[np.ndarray]:
    """
    Construit l'observation IDENTIQUE à training/loup_live :
      base features (M1+H1 normalisés et clip ±5σ)
      + 4 features de position : [pos, unrealized_atr, bars_held_norm, risk_scale]
    """
    if len(df_merged) < cfg.lookback + 1:
        return None

    X = df_merged[FEATURE_COLS].values.astype(np.float32)
    X_norm = safe_normalize(X, stats, clip_sigma=5.0)

    base = X_norm[-cfg.lookback:]  # (T, N_BASE_FEATURES)

    current_price = float(df_merged["close"].iloc[-1]) if len(df_merged) > 0 else 0.0

    # PnL latent en unités d'ATR (scale-invariant, typiquement [-5, 5])
    if pos != 0 and entry_atr > 1e-8 and entry_price > 0.0:
        unrealized_atr = float(pos * (current_price - entry_price) / entry_atr)
    else:
        unrealized_atr = 0.0

    # Durée détention normalisée (cap 3 = overtime), max_holding aligné training (=12)
    scalping_max_holding = 12
    bars_held_norm = float(min(bars_in_position / max(scalping_max_holding, 1), 3.0))

    pos_feature = float(pos)
    risk_feature = float(last_risk_scale)

    extra_vec = np.array(
        [pos_feature, unrealized_atr, bars_held_norm, risk_feature],
        dtype=np.float32
    )
    extra_block = np.repeat(extra_vec[None, :], cfg.lookback, axis=0)

    obs = np.concatenate([base, extra_block], axis=-1).astype(np.float32)
    return obs


# ============================================================
# STRESS-TEST : UTILITAIRES PRIX / EXECUTION
# ============================================================

def apply_price_stress_to_bar(
    high: float,
    low: float,
    close: float,
    prev_close: Optional[float],
    time_i: datetime,
    stress: StressConfig
) -> Tuple[float, float, float]:
    """
    Applique :
      - bruit des ticks
      - micro-gaps
      - news spikes
    sur (high, low, close) d'une bougie.
    """
    if not stress.enable:
        return high, low, close

    # 1) bruit de ticks (gaussien, même facteur sur OHLC)
    noise = np.random.normal(0.0, stress.tick_noise_std)
    factor = 1.0 + noise
    high *= factor
    low *= factor
    close *= factor

    # 2) micro-gap (petit saut de prix)
    if np.random.rand() < stress.micro_gap_prob and prev_close is not None:
        gap_direction = 1 if np.random.rand() < 0.5 else -1
        gap = 1.0 + gap_direction * np.random.normal(0.0, stress.micro_gap_jump_std)
        high *= gap
        low *= gap
        close *= gap

    # 3) news spike (élargissement violent de la range)
    if np.random.rand() < stress.news_spike_prob:
        direction = 1 if np.random.rand() < 0.5 else -1
        spike_mag = np.random.uniform(0.002, 0.01)  # 0.2% à 1%
        if direction > 0:
            # spike haussier
            high = max(high, close * (1.0 + spike_mag))
        else:
            # spike baissier
            low = min(low, close * (1.0 - spike_mag))

    # cohérence OHLC
    lo = min(low, close, high)
    hi = max(low, close, high)
    return hi, lo, close


def compute_execution_price(
    side: int,
    close_price: float,
    time_i: datetime,
    cfg: LiveConfig,
    stress: StressConfig
) -> float:
    """
    Calcule un prix d'exécution avec :
      - spread variable selon l'heure
      - slippage random ±20 bps
    """
    # spread horaire de base
    hour = time_i.hour
    base_spread = cfg.spread_bps

    # nuit et heures creuses : spread plus large
    if hour < 6 or hour >= 22:
        base_spread *= 2.0
    # session US très liquide : spread un peu plus serré
    elif 13 <= hour <= 18:
        base_spread *= 0.8

    # jitter aléatoire du spread
    if stress.enable:
        jitter = np.random.normal(0.0, base_spread * 0.3)
        base_spread = max(base_spread + jitter, 0.0)

    half_spread = base_spread / 2.0

    # slippage random
    slippage = 0.0
    if stress.enable:
        slippage = np.random.uniform(-stress.max_slippage_bps, stress.max_slippage_bps)

    # côté client : on paie le spread + slippage
    if side == 1:  # LONG
        exec_bps = half_spread + slippage
        price = close_price * (1.0 + exec_bps)
    else:          # SHORT
        exec_bps = half_spread + slippage
        price = close_price * (1.0 - exec_bps)

    return float(price)


# ============================================================
# LOGIQUE SL/TP + BREAK-EVEN / TRAILING + ACTION MASK
# ============================================================

def compute_entry_atr(df_closed: pd.DataFrame) -> float:
    if "atr_14" not in df_closed.columns or len(df_closed) == 0:
        return 0.0
    atr = float(df_closed["atr_14"].iloc[-1])
    return max(atr, 0.0)


def compute_dynamic_volume(equity: float, max_lot: float = 100.0) -> float:
    """IDENTIQUE à loup_live.compute_dynamic_volume : paliers 1000$, cap 100.00."""
    if equity <= 2000.0:
        tier = 1
    else:
        tier = int((equity - 1.0) // 1000.0)
    lot = 0.01 * tier
    lot = min(lot, max_lot)
    return round(lot, 2)


def compute_sl_tp(
    cfg: LiveConfig,
    entry_price: float,
    side: int,
    entry_atr: float,
    stress: Optional[StressConfig] = None
):
    fallback = 0.0015 * entry_price
    eff_atr = max(entry_atr, fallback, 1e-8)

    # distorsion ATR (erreur d'estimation)
    if stress is not None and stress.enable:
        distort = np.random.uniform(1.0 - stress.atr_distortion,
                                    1.0 + stress.atr_distortion)
        eff_atr *= distort

    sl_dist = cfg.atr_sl_mult * eff_atr
    tp_dist = cfg.atr_tp_mult * eff_atr * cfg.tp_shrink

    if side == 1:
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    else:
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist

    # randomisation TP/SL ±10 %
    if stress is not None and stress.enable:
        sl_factor = np.random.uniform(1.0 - stress.tp_sl_random,
                                      1.0 + stress.tp_sl_random)
        tp_factor = np.random.uniform(1.0 - stress.tp_sl_random,
                                      1.0 + stress.tp_sl_random)

        # on applique le facteur sur la distance à l'entry (pour garder le sens)
        sl = entry_price + (sl - entry_price) * sl_factor
        tp = entry_price + (tp - entry_price) * tp_factor

    sl = max(sl, 1e-8)
    tp = max(tp, 1e-8)
    return sl, tp


def update_sl_be_trailing_backtest(
    cfg: LiveConfig,
    df_closed: pd.DataFrame,
    state: BTState,
    min_price_dist: float
):
    """
    Version backtest de update_sl_be_trailing_live :
      - break-even + trailing
      - clamp du SL pour respecter une distance mini vis-à-vis du prix courant
        (approximation du trade_stops_level broker).
    """
    if state.position == 0:
        return
    if len(df_closed) == 0:
        return

    row = df_closed.iloc[-1]
    high_bar = float(row["high"])
    low_bar = float(row["low"])
    close_bar = float(row["close"])

    atr = compute_entry_atr(df_closed)
    if atr <= 0:
        return

    side = state.position  # +1 LONG, -1 SHORT
    entry_price = float(state.entry_price)
    current_sl = float(state.sl) if state.sl is not None else 0.0

    if side == 1:
        favorable_move = high_bar - entry_price
    else:
        favorable_move = entry_price - low_bar

    if favorable_move <= 0:
        return

    candidate_sl = None
    reason = ""

    # ----------------- BREAK-EVEN -----------------
    be_trigger = cfg.breakeven_atr_mult * atr

    if favorable_move >= be_trigger:
        be_sl = entry_price
        if side == 1:
            if current_sl < be_sl or current_sl == 0.0:
                candidate_sl = be_sl
                reason = "BREAKEVEN_LONG"
        else:
            if current_sl > be_sl or current_sl == 0.0:
                candidate_sl = be_sl
                reason = "BREAKEVEN_SHORT"

    # ----------------- TRAILING STOP -----------------
    trail_trigger = cfg.trailing_start_atr_mult * atr
    trail_dist = cfg.trailing_dist_atr_mult * atr

    if favorable_move >= trail_trigger:
        if side == 1:
            trail_sl = high_bar - trail_dist
            trail_sl = max(trail_sl, entry_price)
            if candidate_sl is None or trail_sl > candidate_sl:
                candidate_sl = trail_sl
                reason = "TRAIL_LONG"
        else:
            trail_sl = low_bar + trail_dist
            trail_sl = min(trail_sl, entry_price)
            if candidate_sl is None or trail_sl < candidate_sl:
                candidate_sl = trail_sl
                reason = "TRAIL_SHORT"

    if candidate_sl is None:
        return

    # Simu contrainte broker : distance mini
    if side == 1:
        max_sl_allowed = close_bar - min_price_dist
        if candidate_sl > max_sl_allowed:
            candidate_sl = max_sl_allowed  # clamp silencieux (sinon spam)

        if current_sl > 0 and candidate_sl <= current_sl + 1e-8:
            return  # non amélioré, silent

    else:
        min_sl_allowed = close_bar + min_price_dist
        if candidate_sl < min_sl_allowed:
            candidate_sl = min_sl_allowed

        if current_sl > 0 and candidate_sl >= current_sl - 1e-8:
            return

    # Icône selon raison
    icon = "↗" if "TRAIL" in reason else "⊜"  # break-even = égalité
    print(
        f"  {_c(icon, C.YELLOW)} {_c(reason, C.YELLOW)}  "
        f"SL {_c(f'{current_sl:>8.2f}', C.GREY)} → {_c(f'{candidate_sl:.2f}', C.WHITE)}"
    )
    state.sl = candidate_sl


def build_mask_from_pos_scalar(pos: int, device, side: str) -> torch.Tensor:
    mask = torch.zeros(N_ACTIONS, dtype=torch.bool, device=device)

    # 0=BUY  1=SELL  2=HOLD
    if pos != 0:
        mask[2] = True
        return mask

    if side == "long":
        mask[0] = True
        mask[2] = True
    elif side == "short":
        mask[1] = True
        mask[2] = True
    else:  # "both" / "duel"
        mask[0] = True
        mask[1] = True
        mask[2] = True

    return mask


# ============================================================
# BACKTEST PRINCIPAL
# ============================================================

def run_backtest(cfg: LiveConfig):
    print(banner("🐺  LOUP Ω — BACKTEST STRESS-TEST"))
    print(f"  {_c('Symbole', C.GREY):<20} {C.BOLD}{cfg.symbol}{C.RESET}  ({cfg.timeframe=}, HTF={cfg.htf_timeframe})")
    print(f"  {_c('Période', C.GREY):<20} {cfg.date_from}  →  {cfg.date_to or 'maintenant'}")
    print(f"  {_c('Mode side', C.GREY):<20} {_c(cfg.side.upper(), C.MAGENTA + C.BOLD)}")
    print(f"  {_c('Capital initial', C.GREY):<20} {cfg.initial_capital:.2f}$  |  lot={cfg.position_size}  |  lev=x{cfg.leverage:.0f}")
    print(f"  {_c('Min confidence', C.GREY):<20} {cfg.min_confidence:.2f}  ({'argmax pur' if cfg.min_confidence <= 0.0 else 'filtré'})")
    print(hr())

    print(f"{_c('→', C.CYAN)} Connexion MT5…")
    if not mt5.initialize():
        raise RuntimeError("Erreur MT5.initialize() pour le backtest.")

    info = mt5.symbol_info(cfg.symbol)
    if info is None:
        print(f"  {_c('⚠', C.YELLOW)} symbol_info({cfg.symbol}) introuvable — valeurs par défaut.")
        point = 0.01
        broker_stops_points = 0
    else:
        point = info.point
        broker_stops_points = getattr(info, "trade_stops_level", 0) or 0

    min_points = max(broker_stops_points, cfg.backtest_min_extra_points)
    min_price_dist = min_points * point

    print(
        f"  {_c('Broker', C.GREY):<20} point={point:.8f}  "
        f"stops_lvl={broker_stops_points}pts  "
        f"min_dist={min_price_dist:.5f}"
    )

    print(f"{_c('→', C.CYAN)} Téléchargement OHLC M1+H1…")
    try:
        df = fetch_ohlc_with_indicators(cfg)
    finally:
        mt5.shutdown()

    print(f"  {_c('✓', C.GREEN)} {len(df):,} bougies M1 fusionnées avec H1.")

    if len(df) < cfg.lookback + 10:
        raise RuntimeError("Pas assez de données pour lancer le backtest.")

    device = get_device(cfg)
    stats = load_norm_stats(NORM_STATS_PATH)

    # Chargement des modèles
    policy_long = None
    policy_short = None
    policy_duel = None

    def _build_policy_bt():
        return SAINTPolicySingleHead(
            n_features=OBS_N_FEATURES,
            d_model=80,
            num_blocks=2,
            heads=4,
            dropout=0.05,
            ff_mult=2,
            max_len=cfg.lookback,
            n_actions=N_ACTIONS
        ).to(device)

    if cfg.side == "both":
        if not os.path.exists(BEST_MODEL_DUEL_PATH):
            raise FileNotFoundError(f"Modèle DUEL introuvable : {BEST_MODEL_DUEL_PATH}")
        policy_duel = _build_policy_bt()
        policy_duel.load_state_dict(torch.load(BEST_MODEL_DUEL_PATH, map_location=device))
        policy_duel.eval()
        print(f"  {_c('✓', C.GREEN)} Modèle DUEL  : {_c(BEST_MODEL_DUEL_PATH, C.CYAN)}")

    if cfg.side in ("duel", "long"):
        if not os.path.exists(BEST_MODEL_LONG_PATH):
            raise FileNotFoundError(f"Modèle LONG introuvable : {BEST_MODEL_LONG_PATH}")
        policy_long = SAINTPolicySingleHead(
            n_features=OBS_N_FEATURES,
            d_model=80,
            num_blocks=2,
            heads=4,
            dropout=0.05,
            ff_mult=2,
            max_len=cfg.lookback,
            n_actions=N_ACTIONS
        ).to(device)
        policy_long.load_state_dict(torch.load(BEST_MODEL_LONG_PATH, map_location=device))
        policy_long.eval()
        print(f"  {_c('✓', C.GREEN)} Modèle LONG  : {_c(BEST_MODEL_LONG_PATH, C.CYAN)}")

    if cfg.side in ("duel", "short"):
        if not os.path.exists(BEST_MODEL_SHORT_PATH):
            raise FileNotFoundError(f"Modèle SHORT introuvable : {BEST_MODEL_SHORT_PATH}")
        policy_short = SAINTPolicySingleHead(
            n_features=OBS_N_FEATURES,
            d_model=80,
            num_blocks=2,
            heads=4,
            dropout=0.05,
            ff_mult=2,
            max_len=cfg.lookback,
            n_actions=N_ACTIONS
        ).to(device)
        policy_short.load_state_dict(torch.load(BEST_MODEL_SHORT_PATH, map_location=device))
        policy_short.eval()
        print(f"  {_c('✓', C.GREEN)} Modèle SHORT : {_c(BEST_MODEL_SHORT_PATH, C.CYAN)}")

    print(hr())
    print(f"{_c('▶ DÉMARRAGE BOUCLE BACKTEST', C.CYAN + C.BOLD)}")
    print(hr())
    bt_t0 = _time.time()

    # État backtest
    state = BTState(
        capital=cfg.initial_capital,
        equity=cfg.initial_capital,
        max_equity=cfg.initial_capital
    )
    max_dd = 0.0

    n = len(df)
    start_index = cfg.lookback

    # Stress-test V3
    stress = StressConfig(enable=True)
    gap_hole_remaining = 0  # nb de barres à sauter (trous 1–3 minutes)

    # IMPORTANT :
    # - On démarre à lookback+1 pour pouvoir construire l'obs sur df[:i]
    #   (i bougies, dont au moins lookback)
    for i in range(start_index + 1, n - 1):
        row = df.iloc[i]
        time_i = row["time"]
        time_str = time_i.strftime("%Y-%m-%d %H:%M:%S") if isinstance(time_i, pd.Timestamp) else str(time_i)

        # Prix "bruts"
        close_raw = float(row["close"])
        high_raw = float(row["high"])
        low_raw = float(row["low"])
        prev_close = float(df.iloc[i - 1]["close"]) if i > 0 else None

        # Application du bruit / micro-gaps / news spikes
        high_bar, low_bar, close_bar = apply_price_stress_to_bar(
            high_raw, low_raw, close_raw, prev_close, time_i, stress
        )

        # Trous aléatoires de 1–3 minutes (on saute complètement la logique de cette bougie)
        if stress.enable:
            if gap_hole_remaining > 0:
                gap_hole_remaining -= 1
                continue
            if np.random.rand() < stress.hole_prob:
                gap_hole_remaining = np.random.randint(1, 4)  # 1 à 3 minutes
                gap_hole_remaining -= 1
                continue

        closed_this_bar = False

        # 0) BREAK-EVEN + TRAILING (sur les barres < i)
        if state.position != 0 and i > state.entry_index:
            df_prev_closed = df.iloc[:i].reset_index(drop=True)
            update_sl_be_trailing_backtest(cfg, df_prev_closed, state, min_price_dist)

        # 1) SL / TP sur la barre i (avec prix "stressés")
        if state.position != 0 and i > state.entry_index:
            exit_price = None
            exit_reason = None

            if state.position == 1:
                if low_bar <= state.sl:
                    exit_price = state.sl
                    exit_reason = "SL"
                elif high_bar >= state.tp:
                    exit_price = state.tp
                    exit_reason = "TP"
            elif state.position == -1:
                if high_bar >= state.sl:
                    exit_price = state.sl
                    exit_reason = "SL"
                elif low_bar <= state.tp:
                    exit_price = state.tp
                    exit_reason = "TP"

            if exit_price is not None:
                pnl = (
                    state.position *
                    (exit_price - state.entry_price) *
                    state.volume *
                    cfg.leverage
                )
                fee = cfg.fee_rate * exit_price * state.volume
                realized = pnl - fee

                state.capital += realized
                state.trades_pnl.append(realized)
                closed_this_bar = True

                side_txt = "LONG " if state.position == 1 else "SHORT"
                side_col = C.GREEN if state.position == 1 else C.RED
                reason_col = C.RED if exit_reason == "SL" else C.GREEN
                # Durée du trade en barres
                hold_bars = i - state.entry_index
                print(
                    f"  {_c('✗', C.RED if realized < 0 else C.GREEN)} "
                    f"{_c(fmt_time(time_i), C.GREY)}  "
                    f"{_c(side_txt, side_col)} "
                    f"{_c(exit_reason, reason_col)}  "
                    f"@ {exit_price:>9.2f}  "
                    f"PnL {fmt_money(realized)}  "
                    f"Cap {fmt_money(state.capital, width=11)}  "
                    f"{_c(f'({hold_bars}b)', C.GREY)}"
                )

                # Trade-by-trade meta (avant reset)
                state.trades_meta.append({
                    "exit_time": fmt_time(time_i),
                    "entry_idx": int(state.entry_index),
                    "exit_idx": int(i),
                    "side": int(state.position),
                    "entry_price": float(state.entry_price),
                    "exit_price": float(exit_price),
                    "pnl": float(realized),
                    "hit_sl": (exit_reason == "SL"),
                    "hit_tp": (exit_reason == "TP"),
                    "hold_bars": int(hold_bars),
                })

                state.position = 0
                state.volume = 0.0
                state.entry_price = 0.0
                state.sl = 0.0
                state.tp = 0.0
                state.entry_index = -1
                state.entry_atr = 0.0
                state.last_risk_scale = 1.0
                # NE PLUS TOUCHER max_equity ICI

        # 2) Equity & drawdown (avec close "stressé")
        if state.position != 0:
            latent = (
                state.position *
                (close_bar - state.entry_price) *
                state.volume *
                cfg.leverage
            )
        else:
            latent = 0.0

        state.equity = state.capital + latent

        # max_equity suit les plus hauts historiques de l'equity
        state.max_equity = max(state.max_equity, state.equity)

        dd = 0.0
        if state.max_equity > 0:
            dd = (state.max_equity - state.equity) / state.max_equity
        max_dd = max(max_dd, dd)

        # 3) Obs sur df[0..i-1] (PAS de fuite sur la bougie i)
        df_closed_for_obs = df.iloc[:i].reset_index(drop=True)
        bars_in_pos = (i - state.entry_index) if (state.position != 0 and state.entry_index >= 0) else 0
        obs = build_live_obs(
            df_closed_for_obs, stats, cfg,
            pos=state.position,
            entry_price=state.entry_price,
            entry_atr=state.entry_atr,
            bars_in_position=bars_in_pos,
            last_risk_scale=state.last_risk_scale
        )
        if obs is None:
            continue

        # 4) Décision d'ENTRÉE si FLAT — pure argmax, comme loup_live.py
        #    Le seuil min_confidence est optionnel (0.0 par défaut = comme live).
        if state.position == 0 and not closed_this_bar:
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                thr = cfg.min_confidence  # 0.0 = pas de filtre

                # On garde des refs externes au bloc pour le log post-ouverture
                prob_long_open = None
                prob_short_open = None
                probs_long = None
                probs_short = None
                probs_duel = None

                if cfg.side == "both":
                    if policy_duel is None:
                        a = 2  # HOLD
                    else:
                        logits_d, _ = policy_duel(s)
                        logits_d = logits_d[0]
                        mask_d = build_mask_from_pos_scalar(0, device, "both")
                        logits_d_m = logits_d.masked_fill(~mask_d, MASK_VALUE)
                        probs_duel = torch.softmax(logits_d_m, dim=-1)

                        a_duel = int(torch.argmax(probs_duel).item())
                        p_duel = float(probs_duel[a_duel].item())

                        # argmax pur sur 3 actions (BUY/SELL/HOLD)
                        if a_duel in (0, 1) and p_duel >= thr:
                            a = a_duel
                        else:
                            a = 2  # HOLD

                elif cfg.side == "duel":
                    if policy_long is None or policy_short is None:
                        a = 2  # HOLD
                    else:
                        # ----- LONG -----
                        logits_long, _ = policy_long(s)
                        logits_long = logits_long[0]
                        mask_long = build_mask_from_pos_scalar(0, device, "long")
                        logits_long_m = logits_long.masked_fill(~mask_long, MASK_VALUE)
                        prob_long_open = torch.softmax(logits_long_m, dim=-1)

                        # ----- SHORT -----
                        logits_short, _ = policy_short(s)
                        logits_short = logits_short[0]
                        mask_short = build_mask_from_pos_scalar(0, device, "short")
                        logits_short_m = logits_short.masked_fill(~mask_short, MASK_VALUE)
                        prob_short_open = torch.softmax(logits_short_m, dim=-1)

                        # argmax sur les 3 actions (HOLD inclus)
                        a_long  = int(torch.argmax(prob_long_open ).item())
                        a_short = int(torch.argmax(prob_short_open).item())
                        p_long  = float(prob_long_open [a_long ].item())
                        p_short = float(prob_short_open[a_short].item())

                        # On compare les deux camps. Si l'un veut HOLD, on regarde l'autre.
                        long_wants_entry  = a_long  == 0 and p_long  >= thr
                        short_wants_entry = a_short == 1 and p_short >= thr

                        if long_wants_entry and short_wants_entry:
                            # arbitrage : on prend le plus confiant
                            a = a_long if p_long >= p_short else a_short
                        elif long_wants_entry:
                            a = a_long
                        elif short_wants_entry:
                            a = a_short
                        else:
                            a = 2  # HOLD  # HOLD

                elif cfg.side == "long":
                    if policy_long is None:
                        a = 2  # HOLD
                    else:
                        logits_long, _ = policy_long(s)
                        logits_long = logits_long[0]
                        mask_long = build_mask_from_pos_scalar(0, device, "long")
                        logits_long_m = logits_long.masked_fill(~mask_long, MASK_VALUE)
                        probs_long = torch.softmax(logits_long_m, dim=-1)

                        a_long = int(torch.argmax(probs_long).item())
                        p_long = float(probs_long[a_long].item())

                        # argmax pur + filtre optionnel
                        if a_long == 0 and p_long >= thr:
                            a = a_long
                        else:
                            a = 2  # HOLD  # HOLD (soit le model l'a choisi, soit thr non atteint)

                elif cfg.side == "short":
                    if policy_short is None:
                        a = 2  # HOLD
                    else:
                        logits_short, _ = policy_short(s)
                        logits_short = logits_short[0]
                        mask_short = build_mask_from_pos_scalar(0, device, "short")
                        logits_short_m = logits_short.masked_fill(~mask_short, MASK_VALUE)
                        probs_short = torch.softmax(logits_short_m, dim=-1)

                        a_short = int(torch.argmax(probs_short).item())
                        p_short = float(probs_short[a_short].item())

                        if a_short == 1 and p_short >= thr:
                            a = a_short
                        else:
                            a = 2  # HOLD
                else:
                    a = 2  # HOLD

            # mapping vers env_action + risk_scale (3 actions : 0=BUY, 1=SELL, 2=HOLD)
            if a == 0:    # BUY
                env_action = 0
                risk_scale = 1.0
            elif a == 1:  # SELL
                env_action = 1
                risk_scale = 1.0
            else:         # HOLD (a == 2)
                env_action = 2
                risk_scale = 1.0

            # Sécurité explicite : on refuse toute ouverture si on n'est pas flat
            if state.position != 0:
                continue

            # ouverture de position si BUY/SELL (entrée sur la bougie i, mais sans la voir dans l'obs)
            if env_action in (0, 1):
                side = 1 if env_action == 0 else -1

                entry_price = compute_execution_price(
                    side=side,
                    close_price=close_bar,
                    time_i=time_i,
                    cfg=cfg,
                    stress=stress
                )

                # Volume dynamique selon equity (aligné avec loup_live)
                if getattr(cfg, "dynamic_volume", False):
                    base_volume = compute_dynamic_volume(state.equity, getattr(cfg, "max_lot", 100.0))
                else:
                    base_volume = float(cfg.position_size)
                volume = round(base_volume * (risk_scale if risk_scale > 0 else 1.0), 2)
                entry_atr = compute_entry_atr(df_closed_for_obs)
                # fallback ATR si broker renvoie 0 (aligné avec env training)
                effective_entry_atr = max(entry_atr, 0.0015 * entry_price, 1e-8)
                sl, tp = compute_sl_tp(cfg, entry_price, side, entry_atr, stress)

                state.position = side
                state.volume = volume
                state.entry_price = entry_price
                state.sl = sl
                state.tp = tp
                state.entry_index = i
                state.entry_atr = float(effective_entry_atr)
                state.last_risk_scale = risk_scale

                side_txt = "LONG " if side == 1 else "SHORT"
                side_col = C.GREEN if side == 1 else C.RED
                action_name = {0: "BUY", 1: "SELL"}.get(a, "HOLD")

                # Probabilités compactes sur une ligne
                def _fmt_probs(p, picked):
                    names = ["BUY", "SELL", "HOLD"]
                    parts = []
                    for k, nm in enumerate(names):
                        v = p[k].item()
                        col = C.WHITE + C.BOLD if k == picked else C.GREY
                        parts.append(f"{_c(nm, col)}={_c(f'{v:.2f}', col)}")
                    return " ".join(parts)

                if cfg.side == "both" and probs_duel is not None:
                    print(f"    {_c('probas D', C.GREY)} {_fmt_probs(probs_duel, a)}")
                elif cfg.side == "duel" and prob_long_open is not None and prob_short_open is not None:
                    print(
                        f"    {_c('probas L', C.GREY)} "
                        f"{_fmt_probs(prob_long_open, a if a == 0 else -1)}"
                    )
                    print(
                        f"    {_c('probas S', C.GREY)} "
                        f"{_fmt_probs(prob_short_open, a if a == 1 else -1)}"
                    )
                elif cfg.side == "long" and probs_long is not None:
                    print(f"    {_c('probas  ', C.GREY)} {_fmt_probs(probs_long, a)}")
                elif cfg.side == "short" and probs_short is not None:
                    print(f"    {_c('probas  ', C.GREY)} {_fmt_probs(probs_short, a)}")

                rs_txt = f"x{risk_scale:.1f}" if risk_scale != 1.0 else "x1.0"
                print(
                    f"  {_c('▶', C.CYAN)} "
                    f"{_c(fmt_time(time_i), C.GREY)}  "
                    f"{_c(side_txt, side_col + C.BOLD)} {_c(action_name, side_col)}  "
                    f"@ {entry_price:>9.2f}  "
                    f"SL {sl:>9.2f}  TP {tp:>9.2f}  "
                    f"vol={volume:.4f} {_c(rs_txt, C.YELLOW)}"
                )

        # 5) Log de progression périodique
        if ((i - (start_index + 1)) % cfg.progress_interval_bars == 0) or (i == n - 2):
            pnl_total = state.equity - cfg.initial_capital
            nb = len(state.trades_pnl)
            wins = sum(1 for p in state.trades_pnl if p > 0)
            losses = nb - wins
            wr = (wins / nb) if nb > 0 else 0.0
            tot_w = sum(p for p in state.trades_pnl if p > 0)
            tot_l = sum(-p for p in state.trades_pnl if p < 0)
            pf = (tot_w / tot_l) if tot_l > 1e-8 else 0.0
            progress = (i - start_index) / (n - start_index) * 100
            elapsed = _time.time() - bt_t0

            # Couleur DD selon gravité
            dd_pct = max_dd * 100
            dd_col = C.GREEN if dd_pct < 20 else (C.YELLOW if dd_pct < 50 else C.RED)

            print(
                f"\n{hr('·')}\n"
                f"  {_c('⏱', C.MAGENTA)} {_c(fmt_time(time_i), C.GREY)}  "
                f"{_c(f'[{progress:5.1f}%]', C.MAGENTA)}  "
                f"{_c(f'{elapsed:.0f}s', C.GREY)}\n"
                f"    {_c('Equity', C.GREY):<14} {fmt_money(state.equity, width=11)}   "
                f"{_c('PnL', C.GREY)} {fmt_money(pnl_total, width=11)}\n"
                f"    {_c('Trades', C.GREY):<14} {nb:<4d} ({_c(str(wins), C.GREEN)}W / "
                f"{_c(str(losses), C.RED)}L)   "
                f"{_c('WR', C.GREY)} {fmt_pct(wr)}   "
                f"{_c('PF', C.GREY)} {pf:.2f}   "
                f"{_c('DDmax', C.GREY)} {_c(f'{dd_pct:.1f}%', dd_col)}\n"
                f"{hr('·')}"
            )

    # ========================================================
    # RÉSUMÉ FINAL
    # ========================================================
    pnl_total = state.equity - cfg.initial_capital
    nb_trades = len(state.trades_pnl)
    wins = [p for p in state.trades_pnl if p > 0]
    losses = [p for p in state.trades_pnl if p <= 0]
    nb_wins = len(wins)
    nb_losses = len(losses)
    winrate = (nb_wins / nb_trades) if nb_trades > 0 else 0.0
    avg_pnl = float(np.mean(state.trades_pnl)) if nb_trades > 0 else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    max_profit = max(state.trades_pnl) if nb_trades > 0 else 0.0
    max_loss = min(state.trades_pnl) if nb_trades > 0 else 0.0
    total_w = sum(wins)
    total_l = sum(-p for p in losses)
    pf = (total_w / total_l) if total_l > 1e-8 else 0.0
    roi = (pnl_total / cfg.initial_capital) * 100 if cfg.initial_capital > 0 else 0.0
    elapsed = _time.time() - bt_t0

    # Verdict simple selon Sortino approximé : PF*WR
    score = pf * winrate
    if pnl_total > 0 and pf >= 1.2 and winrate >= 0.40:
        verdict, vcol = "✓ ROBUSTE", C.GREEN
    elif pnl_total > 0:
        verdict, vcol = "~ ACCEPTABLE", C.YELLOW
    else:
        verdict, vcol = "✗ NON RENTABLE", C.RED

    pnl_col = C.GREEN if pnl_total > 0 else C.RED
    dd_col = C.GREEN if max_dd*100 < 20 else (C.YELLOW if max_dd*100 < 50 else C.RED)

    # Export CSV trade-by-trade
    import csv as _csv
    trades_csv = f"backtest_trades_{cfg.side}.csv"
    _fields = ["exit_time", "entry_idx", "exit_idx", "side", "entry_price",
               "exit_price", "pnl", "hit_sl", "hit_tp", "hold_bars"]
    with open(trades_csv, "w", newline="", encoding="utf-8") as _f:
        _w = _csv.DictWriter(_f, fieldnames=_fields)
        _w.writeheader()
        for tm in state.trades_meta:
            _w.writerow({
                "exit_time": tm["exit_time"],
                "entry_idx": tm["entry_idx"], "exit_idx": tm["exit_idx"],
                "side": tm["side"],
                "entry_price": round(tm["entry_price"], 4),
                "exit_price": round(tm["exit_price"], 4),
                "pnl": round(tm["pnl"], 4),
                "hit_sl": int(tm["hit_sl"]), "hit_tp": int(tm["hit_tp"]),
                "hold_bars": tm["hold_bars"],
            })
    print(f"  {_c('✓', C.GREEN)} Trade-by-trade CSV : {_c(trades_csv, C.CYAN)} ({len(state.trades_meta)} trades)")

    print("\n" + banner("📊  RÉSULTATS BACKTEST", C.MAGENTA))
    print(f"  {_c('Verdict', C.GREY):<28} {_c(verdict, vcol + C.BOLD)}")
    print(f"  {_c('Mode side', C.GREY):<28} {_c(cfg.side.upper(), C.MAGENTA)}")
    print(f"  {_c('Durée run', C.GREY):<28} {elapsed:.1f}s")
    print(hr())
    print(f"  {_c('Capital initial', C.GREY):<28} {cfg.initial_capital:>10.2f}$")
    print(f"  {_c('Capital final', C.GREY):<28} {state.equity:>10.2f}$")
    print(f"  {_c('PnL total', C.GREY):<28} {_c(f'{pnl_total:>+10.2f}$', pnl_col + C.BOLD)}  ({_c(f'{roi:+.1f}%', pnl_col)})")
    print(f"  {_c('Max drawdown', C.GREY):<28} {_c(f'{max_dd*100:>10.1f}%', dd_col)}")
    print(hr())
    print(f"  {_c('Nb trades', C.GREY):<28} {nb_trades}")
    print(f"  {_c('Wins / Losses', C.GREY):<28} {_c(str(nb_wins), C.GREEN)} / {_c(str(nb_losses), C.RED)}")
    print(f"  {_c('Winrate', C.GREY):<28} {fmt_pct(winrate)}")
    print(f"  {_c('Profit Factor', C.GREY):<28} {pf:.2f}")
    print(f"  {_c('Score (PF×WR)', C.GREY):<28} {score:.3f}")
    print(hr())
    print(f"  {_c('PnL moyen / trade', C.GREY):<28} {fmt_money(avg_pnl)}")
    print(f"  {_c('Avg gain (W)', C.GREY):<28} {fmt_money(avg_win)}")
    print(f"  {_c('Avg perte (L)', C.GREY):<28} {fmt_money(avg_loss)}")
    print(f"  {_c('Meilleur trade', C.GREY):<28} {fmt_money(max_profit)}")
    print(f"  {_c('Pire trade', C.GREY):<28} {fmt_money(max_loss)}")
    print(_c("═" * 78, C.MAGENTA))


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Backtest du modèle BESTPROFIT DUEL (par défaut)
    # ---------------------------------------------------------
    #   side="both"  → 1 modèle unifié (bestprofit_saintv2_loup_duel_both_wf1)
    #   side="duel"  → 2 modèles séparés long+short
    #   side="long"  → bestprofit_long uniquement
    #   side="short" → bestprofit_short uniquement
    # min_confidence=0.0 → pure argmax, comportement identique au live.
    #     Mettre 0.5–0.9 pour stresser la sélectivité a posteriori.
    cfg = LiveConfig(
        side="both",
        min_confidence=0.0,
        n_bars_m1=200_000,
        n_bars_h1=50_000,
        date_from=datetime(2026, 1, 1),
    )
    run_backtest(cfg)
