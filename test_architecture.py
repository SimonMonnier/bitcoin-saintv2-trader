"""Tests de regression de l'architecture SAINT.

L'INDEPENDANCE AU LOT est la propriete centrale verifiee ici, deux fois : sans
memoire, puis avec. Un modele entraine par lots de 128 doit produire exactement
les memes sorties en live, ou il decide sur une observation a la fois.

C'est ce qui a dicte la forme de l'intersample attention retenue. La version
litterale de SAINT fait regarder a chaque ligne les autres lignes DU LOT : a
lot 1 elle degenere, et les poids se comportent autrement en production. Ici
les « autres echantillons » sont une banque figee tiree de la fenetre de train
et rangee dans le checkpoint — meme capacite, meme comportement partout.

    python test_architecture.py

Sortie attendue : "N/N OK".
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from saint_core import (SAINTPolicySingleHead, OBS_N_FEATURES, N_ACTIONS,
                        build_policy)

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


# ============================================================
#  INTERSAMPLE ATTENTION — forme deployable
# ============================================================

def _avec_memoire(K=64, graine=0):
    torch.manual_seed(graine)
    p = SAINTPolicySingleHead(n_features=OBS_N_FEATURES, d_model=80,
                              num_blocks=2, heads=4, ff_mult=2, n_ref=K).eval()
    p.definit_banque(torch.randn(K, 25, OBS_N_FEATURES))
    return p


def test_memoire_independante_du_lot():
    """LE test, avec la memoire active.

    La banque est la meme pour tous les echantillons, donc l'attention
    inter-echantillons ne fait circuler AUCUNE information entre les lignes du
    lot courant. C'est ce qui la distingue de la version litterale de SAINT, et
    ce qui la rend deployable a lot 1.
    """
    p = _avec_memoire()
    x = torch.randn(8, 25, OBS_N_FEATURES)
    with torch.no_grad():
        lot = p(x)[0]
        seul = torch.stack([p(x[i:i + 1])[0][0] for i in range(8)])
    e = (lot - seul).abs().max().item()
    verifie(f"memoire active : lot de 8 == 8 passes de 1 (ecart {e:.1e})", e < 1e-4,
            "la memoire fait fuir de l'information entre echantillons")


def test_memoire_agit():
    """Une memoire qui ne change rien serait pire qu'absente : elle couterait
    des parametres et donnerait l'illusion d'un mecanisme."""
    p = _avec_memoire()
    x = torch.randn(8, 25, OBS_N_FEATURES)
    with torch.no_grad():
        avant = p(x)[0].clone()
        sauv = p.memoire.bank_repr.clone()
        p.memoire.bank_repr = torch.randn_like(sauv) * 3.0
        apres = p(x)[0]
        p.memoire.bank_repr = sauv
    d = (avant - apres).abs().max().item()
    verifie(f"changer la banque change la sortie ({d:.1e})", d > 1e-6,
            "memoire inerte")


def test_memoire_survit_au_checkpoint():
    """La banque doit voyager avec les poids, et l'architecture etre DEDUITE
    du fichier — pas supposee par une constante a tenir synchronisee."""
    import io as _io
    p = _avec_memoire()
    x = torch.randn(4, 25, OBS_N_FEATURES)
    buf = _io.BytesIO()
    torch.save(p.state_dict(), buf)
    buf.seek(0)
    etat = torch.load(buf, weights_only=True)
    q = build_policy(torch.device("cpu"), lookback=25, state_dict=etat)
    q.load_state_dict(etat, strict=True)
    q.eval()
    with torch.no_grad():
        e = (p(x)[0] - q(x)[0]).abs().max().item()
    verifie(f"checkpoint -> live : sortie identique ({e:.1e})", e < 1e-6)
    verifie("build_policy deduit n_ref du checkpoint",
            q.memoire is not None and q.memoire.n_ref == p.memoire.n_ref,
            f"{q.memoire.n_ref if q.memoire else None} vs {p.memoire.n_ref}")


def test_memoire_inerte_sans_banque():
    """Avant que la banque soit figee, le modele doit tourner sans memoire —
    pas interroger des zeros et apprendre a s'y fier."""
    torch.manual_seed(0)
    p = SAINTPolicySingleHead(n_features=OBS_N_FEATURES, d_model=80,
                              num_blocks=2, heads=4, ff_mult=2, n_ref=64).eval()
    try:
        with torch.no_grad():
            o = p(torch.randn(4, 25, OBS_N_FEATURES))[0]
        verifie("banque non figee : inerte, pas d'erreur", o.shape == (4, N_ACTIONS))
    except Exception as e:
        verifie("banque non figee : inerte, pas d'erreur", False, str(e)[:80])


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
    test_memoire_independante_du_lot()
    test_memoire_agit()
    test_memoire_survit_au_checkpoint()
    test_memoire_inerte_sans_banque()
    print(f"\n{_ok}/{_total} OK")
    sys.exit(0 if _ok == _total else 1)
