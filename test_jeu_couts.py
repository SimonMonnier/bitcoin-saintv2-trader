"""Regression CPU de la rampe : pas de modele, GPU ni donnees du run.

Les fonctions de simulation sont chargees depuis leur AST pour verifier le
code de production sans importer le reseau ni lancer l'entrainement.
"""
import ast
from dataclasses import dataclass, replace
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


@dataclass
class Config:
    horizon_max: int = 2
    tp_atr: tuple = (1.,)
    sl_atr: tuple = (1.,)
    glissement_entree_bps: float = 0.
    glissement_sortie_bps: float = 0.
    swap_achat_bps_jour: float = 0.
    swap_vente_bps_jour: float = 0.
    minutes_par_barre: int = 5


class CoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arbre = ast.parse(Path(__file__).with_name("jeu_kairos.py").read_text(encoding="utf-8-sig"))
        cls.ns = {"np": np, "JeuConfig": Config}
        noms = {"table_coups", "table_a_cout"}
        fonctions = [n for n in cls.arbre.body if isinstance(n, ast.FunctionDef) and n.name in noms]
        if len(fonctions) != len(noms):
            raise AssertionError("fonctions de simulation absentes")
        exec(compile(ast.Module(body=fonctions, type_ignores=[]), "jeu_kairos.py", "exec"), cls.ns)

    def trajectoire(self):
        o = np.full(4, 100.)
        h = np.array([100., 101.10, 100., 100.])
        l = np.array([100., 100., 98.8, 100.])
        return o, h, l, np.full(4, 20.), np.ones(4)

    def test_changement_de_sortie_ne_cree_pas_de_gain_interpole(self):
        args = self.trajectoire()
        cfg = Config()
        calcule = lambda f: self.ns["table_coups"](*args, cfg, f)
        zero, plein = calcule(0.), calcule(1.)
        key = (0, 0, 0, 0)
        self.assertEqual(tuple(a[key] for a in zero), (1., 1, 0))
        self.assertEqual(tuple(a[key] for a in plein), (-1., 2, 1))
        actuel = self.ns["table_a_cout"](.25, zero, plein, calcule)
        self.assertEqual(tuple(a[key] for a in actuel), (1., 1, 0))
        # Ancien code : +0.5 R, duree 1 et etiquette STOP, aucun vrai trade.
        ancien_r = .75 * zero[0][key] + .25 * plein[0][key]
        self.assertEqual(ancien_r, .5)
        self.assertNotEqual(actuel[0][key], ancien_r)
        self.assertNotEqual(actuel[2][key], plein[2][key])

    def test_extremites_reutilisees_sans_recalcul(self):
        zero, plein = (object(), object(), object()), (object(), object(), object())

        def interdit(frac):
            raise AssertionError("les extremites ne doivent pas etre recalculees")

        self.assertIs(self.ns["table_a_cout"](0., zero, plein, interdit), zero)
        self.assertIs(self.ns["table_a_cout"](1., zero, plein, interdit), plein)
        for invalide in (-.1, 1.1, np.nan, np.inf):
            with self.subTest(invalide=invalide), self.assertRaises(ValueError):
                self.ns["table_a_cout"](invalide, zero, plein, interdit)

    def test_tous_couples_durees_et_sorties_au_meme_cout(self):
        args = self.trajectoire()
        cfg = Config(tp_atr=(1., 2., 4., 8.), sl_atr=(1., 2., 4., 8.),
                     swap_achat_bps_jour=5., swap_vente_bps_jour=-1.,
                     glissement_entree_bps=.5, glissement_sortie_bps=1.)
        calcule = lambda f: self.ns["table_coups"](*args, cfg, f)
        zero, plein = calcule(0.), calcule(1.)
        for frac in (0., .1, .25, .5, .9, 1.):
            actuel = self.ns["table_a_cout"](frac, zero, plein, calcule)
            attendu = calcule(frac)
            for a, e in zip(actuel, attendu):
                self.assertEqual(a.shape, (4, 2, 4, 4))
                np.testing.assert_array_equal(a, e)

    def test_multi_marche_preserve_frontieres_et_swap(self):
        # Charge les callbacks reels des deux chemins multi-marches.
        for nom in ("main_multi_blocs", "main_multi"):
            with self.subTest(chemin=nom):
                main = next(n for n in self.arbre.body if isinstance(n, ast.FunctionDef) and n.name == nom)
                callbacks = [n for n in main.body if isinstance(n, ast.FunctionDef)
                             and n.name in ("_cfg_m", "_calcule_table")]
                self.assertEqual(len(callbacks), 2)
                o = np.r_[np.full(8, 100.), np.full(8, 10000.)]
                cfg = Config()
                contexte = dict(self.ns, cfg=cfg, replace=replace, o=o, h=o, l=o,
                                sp=np.zeros(16), atr=np.ones(16), blocs=[("A", 0, 8), ("B", 8, 16)],
                                _plus=["swap_achat_bps_jour", "swap_vente_bps_jour"],
                                d=pd.DataFrame({"swap_achat_bps_jour": [0.]*8 + [20.]*8,
                                                "swap_vente_bps_jour": [0.]*8 + [-10.]*8}))
                exec(compile(ast.Module(body=callbacks, type_ignores=[]), "jeu_kairos.py", "exec"), contexte)
                calcule = contexte["_calcule_table"]
                zero, plein = calcule(0.), calcule(1.)
                r, duree, sortie = self.ns["table_a_cout"](.25, zero, plein, calcule)
                self.assertTrue(np.isnan(r[5:8]).all())
                self.assertTrue(np.isnan(r[13:16]).all())
                self.assertEqual(r[0, 0, 0, 0], 0.)
                self.assertLess(r[8, 0, 0, 0], 0.)  # swap acheteur propre au marche B
                self.assertGreater(r[8, 1, 0, 0], 0.)  # credit vendeur de B
                self.assertEqual(duree[8, 0, 0, 0], 3)
                self.assertEqual(sortie[8, 0, 0, 0], 2)


if __name__ == "__main__":
    unittest.main()
