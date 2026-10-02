"""Regressions du replay : chronologie, risque, positions et equity. CPU seul."""
import unittest
from types import SimpleNamespace

import numpy as np

from jeu_compte import compte_live, masque_stops, taille_position


def config(**kw):
    d = dict(capital=1000., risque_pct=1., lot_min=.01, pas_lot=.01,
             contrat=1., levier=500., positions_max=1, sl_atr=(1.,),
             glissement_entree_bps=0., glissement_sortie_bps=0.,
             swap_achat_bps_jour=0., swap_vente_bps_jour=0., minutes_par_barre=5)
    return SimpleNamespace(**(d | kw))


class CompteTests(unittest.TestCase):
    def replay(self, t, r, du, cfg=None, **kw):
        return compte_live(np.asarray(t), np.zeros(len(t), dtype=int),
                           np.asarray(r), np.asarray(du), np.full(30, 100.),
                           np.full(30, 1.), cfg or config(), **kw)

    def test_lot_minimum_ne_force_jamais_le_risque(self):
        c = config()
        self.assertEqual(taille_position(1000, 1100, 50000, c), 0)
        self.assertEqual(taille_position(1000, 1000, 50000, c), .01)
        self.assertEqual(taille_position(1000, 499, 50000, c), .02)
        np.testing.assert_array_equal(masque_stops(np.array([499, 1100]), 50000, 1000, c), [True, False])
        b = compte_live([0], [0], [1.], [1], np.full(3, 50000.), np.full(3, 1100.), c)
        self.assertEqual(b["live_risque_refuses"], 1)
        self.assertEqual(b["live_pris"], 0)

    def test_marge_reduit_lot_sans_remonter_au_minimum(self):
        c = config(levier=1)
        self.assertAlmostEqual(taille_position(1000, 1, 100, c, 2), .02)
        self.assertEqual(taille_position(1000, 1, 100, c, .5), 0)

    def test_position_ouverte_la_veille_bloque_le_lendemain(self):
        b = self.replay([9, 10], [1., 1.], [4, 4])
        self.assertEqual(b["live_pris"], 1)
        self.assertEqual(b["live_positions"], 1)
        self.assertEqual(b["live_total"], 10)

    def test_sortie_intrabar_ne_libere_pas_la_place_au_meme_open(self):
        b = self.replay([0, 1], [1., 1.], [2, 1], sorties=[0, 0])
        self.assertEqual(b["live_pris"], 1)
        self.assertEqual(b["live_positions"], 1)

    def test_timeout_a_open_libere_place_et_solde(self):
        # Le premier trade decide a 0 entre a 1, sort au timeout open[2].
        # Le deuxieme entre a ce meme open[2], avec le nouveau solde 1010.
        b = self.replay([0, 1], [1., 1.], [2, 1], sorties=[2, 0])
        self.assertEqual(b["live_pris"], 2)
        self.assertAlmostEqual(b["live_total"], 20.1)

    def test_timeout_n_impute_pas_la_cloture_de_sa_barre(self):
        close = np.full(6, 100.)
        close[3] = 1.  # Apres le timeout open[3] : plus d'exposition.
        b = compte_live([0], [0], [0.], [3], close, np.ones(6), config(),
                        sens=[0], open_=np.full(6, 100.), spread=np.zeros(6), sorties=[2])
        self.assertEqual(b["live_equity_dd_dollars"], 0)

    def test_dd_realise_respecte_ordre_des_sorties(self):
        b = self.replay([0, 1, 2], [-2., 1., -2.], [6, 1, 1],
                        cfg=config(positions_max=3))
        # Gain de 10, perte de 20.20 (taille sur 1010), puis perte de 20.
        self.assertAlmostEqual(b["live_dd_realise_dollars"], -40.2)
        self.assertAlmostEqual(b["live_total"], -30.2)

    def test_equity_latente_negative_malgre_trade_gagnant(self):
        close = np.full(6, 100.)
        close[1], close[2] = 99.5, 100.2
        b = compte_live([0], [0], [1.], [3], close, np.ones(6), config(),
                        sens=[0], open_=np.full(6, 100.), spread=np.zeros(6), sorties=[0])
        self.assertEqual(b["live_total"], 10)
        self.assertEqual(b["live_dd_realise_dollars"], 0)
        self.assertAlmostEqual(b["live_equity_dd_dollars"], -5)
        self.assertEqual(b["live_equity_mode"], "liquidation_aux_clotures_bougies")

    def test_equity_prix_entree_open_spread_glissements_swap(self):
        c = config(glissement_entree_bps=1, glissement_sortie_bps=2,
                   swap_achat_bps_jour=10, minutes_par_barre=1440)
        op = np.full(5, 100.); op[1] = 101.
        spread = np.full(5, 3.)
        b = compte_live([0], [0], [1.], [2], np.full(5, 100.), np.ones(5), c,
                        sens=[0], open_=op, spread=spread, sorties=[0])
        p0 = 101 * (1 + .0003 + .0001)
        attendu = 10 * (100 * (1 - .0002) - p0 - .001 * p0)
        self.assertAlmostEqual(b["live_equity_dd_dollars"], attendu)
        self.assertEqual(b["live_total"], 10)  # r inclut deja les couts.

    def test_pas_de_fausse_equity_sans_metadonnees(self):
        b = self.replay([0], [1.], [2])
        self.assertFalse(b["live_equity_disponible"])
        self.assertNotIn("live_equity_dd_pct", b)
        self.assertEqual(b["live_dd_type"], "solde_realise")


if __name__ == "__main__":
    unittest.main()
