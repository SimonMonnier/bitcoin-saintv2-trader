"""Profil de la VRAIE boucle d'entrainement, a petite echelle."""
import cProfile, io as _io, pstats, sys, time
import training as T

cfg = T.PPOConfig()
cfg.epochs = 1
cfg.episodes_per_epoch = 4          # 21 fois moins que la production
cfg.model_prefix = "profil_tmp"
cfg.n_moyenne_poids = 1
t0 = time.perf_counter()
pr = cProfile.Profile(); pr.enable()
try:
    T.run_walkforward(cfg, train_frac=0.55, val_frac=0.15, test_frac=0.10,
                      max_folds=1)
except SystemExit:
    pass
except Exception as e:
    print(f"[arret] {type(e).__name__}: {e}")
pr.disable()
print(f"\nDUREE TOTALE {time.perf_counter()-t0:.1f} s pour "
      f"{cfg.episodes_per_epoch} episodes (production : 84)")
sio = _io.StringIO()
pstats.Stats(pr, stream=sio).sort_stats("tottime").print_stats(14)
print("\nOU PASSE LE TEMPS")
for l in sio.getvalue().split("\n")[4:22]:
    print("  " + l[:130])
