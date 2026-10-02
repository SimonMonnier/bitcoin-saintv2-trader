"""Tests CPU courts du jeu continu, de ses observations et du masque PPO."""
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import torch

import jeu_kairos as J

torch.set_num_threads(1)


class PolitiqueFixe:
    def __init__(self, cfg, achats=(), logits_stops=None):
        self.cfg, self.achats = cfg, set(achats)
        self.logits_stops = (list(range(len(cfg.sl_atr))) if logits_stops is None
                             else list(logits_stops))
        self.appels = []

    def eval(self):
        return self

    def jeu(self, x):
        n = len(x)
        instants = x[:, -1, 0].cpu().numpy().astype(int)
        self.appels.extend((int(t), x[k, -1, -5:].cpu().numpy().copy())
                           for k, t in enumerate(instants))
        le = torch.full((n, 3), -100., device=x.device)
        le[:, J.ATTENDRE] = 0
        le[:, J.ACHETER] = torch.tensor([100. if t in self.achats else -100.
                                       for t in instants], device=x.device)
        ltp = torch.zeros((n, 2, len(self.cfg.tp_atr)), device=x.device)
        lsl = torch.tensor(self.logits_stops, dtype=torch.float32,
                           device=x.device).expand(n, 2, -1)
        return le, torch.zeros(n, device=x.device), ltp, lsl


def config(**kw):
    base = dict(jeu_version=2, lookback=1, n_expert=0, horizon_max=4,
                tp_atr=(1.,), sl_atr=(1.,), jetons=10, positions_max=1,
                porte_rang_expert=0., marches=(), capital=1000., risque_pct=1.,
                lot_min=.01, pas_lot=.01, contrat=1., levier=500., vie_R=3.,
                glissement_entree_bps=0., glissement_sortie_bps=0.,
                swap_achat_bps_jour=0., swap_vente_bps_jour=0.,
                ppo_epochs=1, minibatch=1024)
    return replace(J.JeuConfig(), **(base | kw))


def donnees(cfg, n=80):
    X = np.arange(n, dtype=np.float32)[:, None]
    shape = (n, 2, len(cfg.tp_atr), len(cfg.sl_atr))
    return X, np.ones(shape, np.float32), np.full(shape, 4, np.int16), np.zeros(shape, np.int8)


class RolloutTests(unittest.TestCase):
    def lance(self, cfg, jours, achats=(), collecte=False, R=None, D=None, S=None,
              prix=None, atr=None, marge=None, rangs=None, fin_valide=80,
              logits_stops=None):
        X, r, d, s = donnees(cfg)
        p = PolitiqueFixe(cfg, achats, logits_stops)
        out = J.joue(p, np.asarray(jours, dtype=int).reshape(-1, 2), X,
                     r if R is None else R, d if D is None else D,
                     s if S is None else S, fin_valide, cfg, "cpu", False,
                     collecte=collecte, prix=prix, atr=atr, marge=marge, rangs=rangs)
        return p, X, out

    def test_une_position_conservee_a_minuit(self):
        c = config()
        p, _, (scores, coups, _) = self.lance(c, [[0, 10], [10, 20]], [9, 10, 13])
        self.assertEqual([x[1] for x in coups], [9, 13])
        np.testing.assert_array_equal(scores, [1., 1.])
        for premier, suivant in zip(coups, coups[1:]):
            self.assertGreaterEqual(suivant[1], premier[1] + premier[6])

    def test_observation_ne_voit_pas_issue_future(self):
        c = config(positions_max=2)
        _, r, _, _ = donnees(c)
        r[0] = 2.; r[1] = -1.
        p, _, _ = self.lance(c, [[0, 10]], [0, 1], R=r)
        obs = dict(p.appels)
        self.assertEqual(obs[1][1], 0.)
        self.assertEqual(obs[1][4], .5)
        self.assertAlmostEqual(obs[4][1], 2 / 3, places=6)
        self.assertAlmostEqual(obs[5][1], 1 / 3, places=6)

    def test_perte_de_la_veille_arrete_le_jour_de_resolution(self):
        c = config()
        _, r, _, _ = donnees(c)
        r[9] = -3.
        p, _, (_, coups, _) = self.lance(c, [[0, 10], [10, 20], [20, 30]],
                                        [9, 13, 20], R=r)
        self.assertEqual([x[1] for x in coups], [9, 20])
        self.assertEqual(dict(p.appels)[20][1], 0)

    def test_stops_non_financables_masques_et_rejoues_par_ppo(self):
        c = config(sl_atr=(1., 2., 4., 8.))
        prix, atr = np.full(80, 50000.), np.full(80, 200.)
        m = J.fraction_marge(prix, atr, c)
        p, X, (scores, coups, tr) = self.lance(
            c, [[0, 10]], [0, 4, 8], collecte=True, marge=m,
            logits_stops=[0., 1., 2., 100.])
        self.assertEqual([x[4] for x in coups], [2, 2, 2])
        for v in tr[0]:
            masque = ((v[2] >> (np.arange(4) + 2)) & 1).astype(bool)
            np.testing.assert_array_equal(masque, [True, True, True, False])
            logits = torch.tensor([0., 1., 2., 100.]).masked_fill(
                ~torch.as_tensor(masque), J._NEG)
            attendu = torch.log_softmax(logits, -1)[v[5]].item()
            self.assertAlmostEqual(v[8], attendu, places=7)
        self.assertTrue(tr[0][-1][12])
        lignes = J.avantages(tr, c)
        with patch.object(J, "_pas") as optimisation:
            bilan = J.maj_ppo(p, {}, lignes, X, c, "cpu", np.random.default_rng(3))
        self.assertEqual(optimisation.call_count, 1)
        self.assertEqual(bilan["kl"], 0.)
        probs = torch.softmax(torch.tensor([0., 1., 2.]), -1)
        h = float(-(probs * probs.log()).sum()) / 2
        self.assertAlmostEqual(bilan["Hb"], h, places=6)
        self.assertEqual(sum(scores), sum(x[5] for x in coups))

    def test_tous_stops_refuses_attente_finie(self):
        c = config()
        _, _, (scores, coups, tr) = self.lance(
            c, [[0, 10]], range(10), collecte=True,
            prix=np.full(80, 100000.), atr=np.full(80, 2000.))
        self.assertFalse(coups)
        self.assertEqual(len(tr[0]), 10)
        self.assertTrue(all(v[2] == 0 and v[3] == J.ATTENDRE for v in tr[0]))
        self.assertTrue(tr[0][-1][12])
        self.assertEqual(scores[0], 0.)

    def test_collecte_eval_meme_jour_memes_actions(self):
        c = config(sl_atr=(1., 2., 4., 8.))
        kw = dict(prix=np.full(80, 50000.), atr=np.full(80, 200.))
        _, _, ev = self.lance(c, [[0, 10]], [0, 4, 8], **kw)
        _, _, co = self.lance(c, [[0, 10]], [0, 4, 8], collecte=True, **kw)
        self.assertEqual(ev[1], co[1])
        np.testing.assert_array_equal(ev[0], co[0])

    def test_segments_distincts_ne_partagent_pas_le_solde(self):
        c = config(sl_atr=(1., 8.))
        _, r, _, _ = donnees(c)
        r[9] = 100.
        limites = np.r_[np.full(30, 30), np.full(50, 80)]
        _, _, (_, coups, _) = self.lance(
            c, [[0, 10], [30, 40]], [9, 30], R=r, fin_valide=limites,
            prix=np.full(80, 50000.), atr=np.full(80, 200.), logits_stops=[0., 100.])
        self.assertEqual(sorted((v[1], v[4]) for v in coups), [(9, 0), (30, 0)])

    def test_scores_sont_somme_des_trades_attribues(self):
        c = config()
        _, _, (scores, coups, _) = self.lance(c, [[10, 20], [0, 10]], [0, 4, 8, 12, 16])
        for g in range(len(scores)):
            self.assertEqual(scores[g], sum(x[5] for x in coups if x[0] == g))

    def test_porte_vide_et_episodes_vides_terminent(self):
        c = config(porte_rang_expert=.9)
        p, _, (_, coups, _) = self.lance(c, [[0, 10], [10, 20]], range(20),
                                        rangs=np.zeros((80, 2)))
        self.assertFalse(coups)
        self.assertFalse(p.appels)
        _, _, (scores, coups, tr) = self.lance(c, [], collecte=True)
        self.assertEqual(len(scores), 0)
        self.assertEqual(coups, [])
        self.assertEqual(tr, [])

    def test_duree_transition_inclut_attente_place_libre(self):
        c = config(positions_max=2)
        _, _, (_, _, tr) = self.lance(c, [[0, 10]], [0, 1, 4], collecte=True)
        for a, b in zip(tr[0], tr[0][1:]):
            self.assertEqual(a[11], b[0] - a[0])

    def test_cloture_dans_trou_ne_consomme_pas_vie_jour_suivant(self):
        c = config()
        _, r, _, _ = donnees(c)
        r[9] = -3.
        # La perte se realise a 13 dans un jour absent, pas dans [20,30).
        _, _, (_, coups, _) = self.lance(c, [[0, 10], [20, 30]], [9, 20], R=r)
        self.assertEqual([v[1] for v in coups], [9, 20])


if __name__ == "__main__":
    unittest.main()
