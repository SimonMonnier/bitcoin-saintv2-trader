"""Execution on MT5 BID candles. Prices returned are executable quotes."""


def execution_quote(bid: float, side: int, spread_bps: float) -> float:
    """BUY pays ASK; SELL receives BID. Spread is relative to the BID."""
    return float(bid) * (1.0 + max(float(spread_bps), 0.0) / 10000.0 if side == 1 else 1.0)
