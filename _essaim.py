"""Les deux en essaim : que faut-il, et que coute 100 % de l'equite ?

L'or ne tient qu'une position a 3 % de budget parce que son lot MINIMUM
risque 2.81 % d'un compte de 1 000 EUR — la taille du contrat, pas un
reglage. Deux facons d'obtenir un essaim des deux cotes : monter le budget,
ou monter le capital. On mesure les deux sur l'equite continue, alignee.
"""
import numpy as np, pandas as pd
import training as T, instruments as I
from saint_core import FEATURE_COLS
from alignement import jeux_alignes

base = T.PPOConfig()
jeux, commun = jeux_alignes()
print(f"{len(commun):,} barres communes, {commun[0]:%Y-%m-%d} -> {commun[-1]:%Y-%m-%d}\n")

def essai(budget, capital, n=4000, depart=60_000):
    pf = T.Portefeuille(capital)
    envs = {}
    for sym in ("BTCUSD", "XAUUSD"):
        c = I.config_instrument(base, sym)
        c.budget_risque = budget
        c.initial_capital = capital
        df = jeux[sym]
        tr = int(len(df) * 0.55)
        st = T.compute_and_save_global_norm_stats(df.iloc[:tr], FEATURE_COLS, path=None)
        d = T.MarketData(df.iloc[:tr].reset_index(drop=True), FEATURE_COLS, st)
        e = T.BTCTradingEnvDiscrete(d, c)
        e.portefeuille = pf
        T.reset_au_depart(e, depart)
        e.end_idx = depart + n + 5
        envs[sym] = e
    pf.capital = float(capital)
    rng = np.random.default_rng(11)
    eq, nb, no, fin = [], [], [], None
    for _ in range(n):
        for sym in ("BTCUSD", "XAUUSD"):
            u = rng.random()
            _, _, dn, _, _ = envs[sym].step(0 if u < .35 else (1 if u < .7 else 2))
            if dn and fin is None:
                fin = f"{sym}"
        eq.append(pf.equity())
        nb.append(envs["BTCUSD"].n_positions); no.append(envs["XAUUSD"].n_positions)
        if fin: break
    eq = np.array(eq)
    pic = np.maximum.accumulate(eq)
    creux = float((1 - eq/np.maximum(pic, 1e-9)).max())
    return (int(np.median(nb)), int(np.median(no)), max(nb), max(no),
            eq[-1]/capital - 1, creux, len(eq), fin)

print(f"{'budget':>8} {'capital':>9} {'pos BTC':>9} {'pos OR':>8} "
      f"{'gain':>8} {'creux':>8} {'barres':>8} {'fin':>9}")
print("-" * 72)
for budget, capital in ((0.03, 1000), (0.10, 1000), (0.30, 1000),
                        (1.00, 1000), (0.03, 25000)):
    mb, mo, xb, xo, g, cx, nn, fin = essai(budget, capital)
    print(f"{100*budget:>7.0f}% {capital:>8,}$ {mb:>4}/{xb:<4} {mo:>3}/{xo:<4} "
          f"{100*g:>+7.0f}% {100*cx:>7.0f}% {nn:>8,} {fin or 'ok':>9}")
print("\n'pos' = mediane/max. 'fin' = instrument dont l'episode s'est arrete.")
