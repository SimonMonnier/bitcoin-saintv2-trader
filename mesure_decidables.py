"""Combien de barres redeviennent des DECISIONS selon le nombre de positions tolerees.

C'est le chiffre qui justifie ou non le refactor : l'acteur ne recoit de
gradient que sur les barres ou il peut entrer. Aujourd'hui il en recoit ~500
par epoch sur 483 840 barres collectees, parce qu'il est en position presque
tout le temps.
"""
import numpy as np
import cibles as C
import training as T

cfg = T.PPOConfig()
df = T.load_mt5_data(cfg)
n = len(df)
b = int(n * 0.55)   # fold 1 : train [0, 0.55n), validation ensuite
dv = df.iloc[b:b + int(n * 0.15)].reset_index(drop=True)
del df

PAS = 3
idx = np.arange(cfg.lookback, len(dv) - C.BORNE_DEFAUT - 2, PAS)
ra, rv, da, dvv = C.rendements(dv, idx, cfg, durees=True)
ok = np.isfinite(da)
idx, da = idx[ok], da[ok]
print(f"{len(idx):,} barres candidates, duree mediane {np.median(da)*5/60:.1f} h\n")

# La politique entre des qu'elle le peut : c'est le regime observe (elle est
# plate une seule barre entre deux trades). On simule donc "entrer des que
# possible", ce qui est le pire cas pour le nombre de decisions.
print(f"{'K':>4} {'decisions':>11} {'part des barres':>17} {'x vs K=1':>10} "
      f"{'trades':>8}")
print("-" * 56)
base = None
for K in (1, 2, 3, 4, 6, 8, 12, 20):
    fins = []
    dec = 0
    trades = 0
    for i in range(len(idx)):
        t = idx[i]
        fins = [f for f in fins if f > t]
        if len(fins) < K:
            dec += 1                    # l'acteur peut decider ici
            fins.append(t + da[i])      # ... et il entre, regime observe
            trades += 1
    if base is None:
        base = dec
    print(f"{K:>4} {dec:>11,} {100*dec/len(idx):>16.1f}% {dec/base:>9.1f}x "
          f"{trades:>8,}")

print("\nLECTURE. 'decisions' est le nombre de barres ou l'acteur recoit du")
print("gradient. A K=1 il est quasi nul parce que la position bloque tout ;")
print("le plafond configure (max_decisions_per_epoch) vaut 40 000 et n'est")
print("jamais atteint. La colonne 'x vs K=1' est le facteur par lequel le")
print("signal d'apprentissage de la politique serait multiplie.")
