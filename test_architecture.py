"""Tests de regression de l'architecture SAINT.

Le test 1 est le plus important du fichier. Il verifie qu'AUCUNE information ne
circule entre les echantillons d'un lot : sans cette propriete, un modele
entraine avec des lots de 128 se comporte autrement en live, ou il decide sur
une observation a la fois. C'est la raison pour laquelle l'intersample
attention de SAINT — l'operation qui donne son nom au papier — est
volontairement absente ici.

    python test_architecture.py

Sortie attendue : "N/N OK".
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from saint_core import SAINTPolicySingleHead, OBS_N_FEATURES, N_ACTIONS

_ok = _total = 0


def verifie(nom, condition, detail=""):
    global _ok, _total
    _total += 1
    if condition:
        _ok += 1
        print(f"  OK    {nom}")
    else:
        print(f"  ECHEC {nom}   {detail}")


def _modele(blocs=2, graine=0):
    torch.manual_seed(graine)
    return SAINTPolicySingleHead(n_features=OBS_N_FEATURES, d_model=80,
                                 num_blocks=blocs, heads=4, ff_mult=2).eval()


def test_independance_au_lot():
    """LE test. Un lot de 8 doit donner exactement 8 passes de 1.

    Si ce test echoue, le modele n'est pas deployable : les poids appris sous
    un lot de 128 produiraient d'autres decisions en live, ou le lot vaut 1.
    """
    p = _modele()
    x = torch.randn(8, 25, OBS_N_FEATURES)
    with torch.no_grad():
        lot, v_lot = p(x)
        seul = torch.stack([p(x[i:i + 1])[0][0] for i in range(8)])
        v_seul = torch.stack([p(x[i:i + 1])[1][0] for i in range(8)])
    e_l = (lot - seul).abs().max().item()
    e_v = (v_lot - v_seul).abs().max().item()
    verifie(f"logits : lot de 8 == 8 passes de 1 (ecart {e_l:.1e})", e_l < 1e-4,
            "de l'information circule entre echantillons")
    verifie(f"value  : lot de 8 == 8 passes de 1 (ecart {e_v:.1e})", e_v < 1e-4)


def test_ordre_du_lot_indifferent():
    """Permuter le lot doit permuter la sortie, sans rien changer d'autre."""
    p = _modele()
    x = torch.randn(6, 25, OBS_N_FEATURES)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        direct = p(x)[0][perm]
        permute = p(x[perm])[0]
    ecart = (direct - permute).abs().max().item()
    verifie(f"sortie invariante a l'ordre du lot (ecart {ecart:.1e})", ecart < 1e-4)


def test_formes():
    p = _modele()
    with torch.no_grad():
        logits, value = p(torch.randn(8, 25, OBS_N_FEATURES))
    verifie("formes de sortie",
            logits.shape == (8, N_ACTIONS) and value.shape == (8,),
            f"{tuple(logits.shape)} {tuple(value.shape)}")


def test_acteur_non_pique_a_init():
    """Init orthogonale gain 0.01 sur l'acteur (Engstrom et al. 2020).

    Un acteur initialise a gain 1 sort des logits deja marques : la politique
    demarre piquee sur une preference arbitraire, et les premiers gradients
    servent a la defaire au lieu d'apprendre.
    """
    p = _modele()
    with torch.no_grad():
        probs = torch.softmax(p(torch.randn(64, 25, OBS_N_FEATURES))[0], -1)
    etendue = (probs.max(-1).values - probs.min(-1).values).mean().item()
    verifie(f"politique quasi uniforme a l'init (etendue {etendue:.4f})",
            etendue < 0.05)


def test_tous_les_gradients():
    """Aucun parametre mort : LayerScale a 1e-4 ne doit pas couper le gradient."""
    p = _modele()
    p.train()
    logits, value = p(torch.randn(8, 25, OBS_N_FEATURES))
    (logits.pow(2).mean() + value.pow(2).mean()).backward()
    morts = [n for n, q in p.named_parameters()
             if q.grad is None or q.grad.abs().sum().item() == 0.0]
    verifie(f"tous les parametres recoivent du gradient", not morts,
            f"{len(morts)} morts : {morts[:5]}")


def test_lookback_variable():
    """RoPE se recalcule : le lookback n'est pas fige dans les poids."""
    p = _modele()
    try:
        with torch.no_grad():
            for T in (10, 25, 54):
                assert p(torch.randn(2, T, OBS_N_FEATURES))[0].shape == (2, N_ACTIONS)
        verifie("lookback variable 10 / 25 / 54", True)
    except Exception as e:
        verifie("lookback variable 10 / 25 / 54", False, str(e)[:80])


def test_export_onnx():
    """L'EA MQL5 consomme un ONNX : toute brique exotique doit s'exporter."""
    import io
    p = _modele()
    try:
        buf = io.BytesIO()
        torch.onnx.export(p, torch.randn(1, 25, OBS_N_FEATURES), buf,
                          input_names=["obs"], output_names=["logits", "value"],
                          opset_version=17)
        verifie(f"export ONNX ({buf.tell() // 1024} Ko)", buf.tell() > 0)
    except Exception as e:
        verifie("export ONNX", False, f"{type(e).__name__}: {str(e)[:90]}")


def test_deterministe_en_eval():
    """Dropout et profondeur stochastique doivent etre inertes en eval()."""
    p = _modele()
    x = torch.randn(4, 25, OBS_N_FEATURES)
    with torch.no_grad():
        a, b = p(x)[0], p(x)[0]
    verifie("eval() est deterministe", torch.equal(a, b))


if __name__ == "__main__":
    print("Architecture SAINT — tests de regression\n")
    test_independance_au_lot()
    test_ordre_du_lot_indifferent()
    test_formes()
    test_acteur_non_pique_a_init()
    test_tous_les_gradients()
    test_lookback_variable()
    test_deterministe_en_eval()
    test_export_onnx()
    print(f"\n{_ok}/{_total} OK")
    sys.exit(0 if _ok == _total else 1)
