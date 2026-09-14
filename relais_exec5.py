"""Attend que exec4 atteigne l'epoch 10, puis enchaine sur exec5.

exec4 mesure une chose : scalping_max_holding 120 -> 30 et le retrait de la
sortie par le temps. exec5 en mesure une autre : ls_ratio_top remplacee par
taker_1m_ma5. Les enchainer au lieu de les superposer est ce qui permettra
d'attribuer un resultat a une cause.

Lecture seule tant que l'epoch 10 n'est pas ecrite. Fermer cette fenetre
n'arrete rien.

    python relais_exec5.py <pid_exec4>
"""

import os
import re
import sys
import time
import shutil
import signal
import subprocess
import datetime as dt

LOG = "training_btc.log"
CIBLE = re.compile(r"EPOCH 010  VAL")
ARCHIVES = "runs_archives"
PAUSE = 60


def vivant(pid: int) -> bool:
    """Le processus tourne-t-il encore ?

    SURTOUT PAS os.kill(pid, 0). Sous POSIX c'est le test de vie standard ;
    sous Windows os.kill appelle TerminateProcess(handle, sig) et le signal
    devient le CODE DE SORTIE. os.kill(pid, 0) ne teste donc rien : il TUE le
    processus avec le code 0, ce qui ressemble a un arret propre dans les
    journaux. C'est ainsi qu'un veilleur en lecture seule a supprime un
    entrainement a l'epoch 7, sans la moindre trace d'erreur.

    tasklist est lent mais ne touche a rien.
    """
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                       capture_output=True, text=True)
    return str(pid) in r.stdout


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python relais_exec5.py <pid_exec4>")
        return 2
    pid = int(sys.argv[1])
    print(f"[RELAIS] surveillance de exec4 (pid {pid}) — cible : epoch 10")

    while True:
        if not vivant(pid):
            print("[RELAIS] exec4 s'est arrete avant l'epoch 10. "
                  "Aucun enchainement automatique : a verifier a la main.")
            return 1
        try:
            txt = open(LOG, encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            txt = ""
        if CIBLE.search(txt):
            print(f"[RELAIS] epoch 10 detectee a {dt.datetime.now():%H:%M:%S}")
            break
        time.sleep(PAUSE)

    # Laisser l'epoch finir d'ecrire sa ligne META et ses checkpoints.
    print("[RELAIS] pause de 90 s pour laisser l'epoch se cloturer proprement")
    time.sleep(90)

    print(f"[RELAIS] arret de exec4 (pid {pid})")
    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    time.sleep(5)

    os.makedirs(ARCHIVES, exist_ok=True)
    horo = dt.datetime.now().strftime("%Y%m%d_%H%M")
    shutil.copy(LOG, os.path.join(ARCHIVES, f"training_exec4_{horo}.log"))
    for f in os.listdir("."):
        if f.startswith("run_") and "exec4" in f and f.endswith(".json"):
            shutil.copy(f, ARCHIVES)
    print(f"[RELAIS] journal exec4 archive dans {ARCHIVES}/")

    print("[RELAIS] lancement de exec5")
    with open(LOG, "w", encoding="utf-8") as fh:
        p = subprocess.Popen([sys.executable, "-u", "training.py"],
                             stdout=fh, stderr=subprocess.STDOUT)
    print(f"[RELAIS] exec5 demarre, pid {p.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
