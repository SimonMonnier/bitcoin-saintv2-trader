# -*- coding: utf-8 -*-
"""Tests CPU du pipeline/rejeu; petits experts, aucun cache de marche reel."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

import jeu_artifacts as A


class TestArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(52)
        cls.d = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=2200, freq="5min"),
                              "x": rng.normal(size=2200).astype(np.float32),
                              "z": rng.normal(size=2200).astype(np.float32)})
        cls.cols = ["x", "z"]
        cls.X = cls.d[cls.cols].to_numpy(np.float32)
        cls.sm = {"mean": cls.X.mean(0), "std": cls.X.std(0)}
        cls.Xn = np.clip((cls.X - cls.sm["mean"]) / (cls.sm["std"] + 1e-8), -5, 5)
        cls.models = []
        for sign in (1, -1):
            m = lgb.LGBMRegressor(n_estimators=3, num_leaves=4, min_child_samples=3,
                                  n_jobs=1, verbose=-1, random_state=0)
            m.fit(cls.Xn[:96], sign * cls.Xn[:96, 0])
            cls.models.append(m)
        cls.pred_final = np.stack([m.predict(cls.Xn) for m in cls.models], axis=1).astype(np.float32)
        cls.pred = cls.pred_final.copy()
        # Imite des predictions OOF differentes de l'expert final avant le test.
        cls.pred[:2000] += rng.normal(0, 0.1, (2000, 2)).astype(np.float32)
        import jeu_kairos as J
        cls.FE = J.features_expert(cls.pred, fenetre=2000)
        cls.se = {"mean": cls.FE.astype(np.float64).mean(0),
                  "std": cls.FE.astype(np.float64).std(0)}
        cls.cfg = {"minutes_par_barre": 5, "n_expert": 4,
                   "expert_ratio_min": 0.0, "jeu_version": 2}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "pipeline_run_bloc1.json"

    def save(self):
        return A.sauvegarde_pipeline_bloc(
            self.path, cfg=self.cfg, donnees=self.d, colonnes=self.cols,
            normalisation_marche=self.sm, normalisation_expert=self.se,
            modeles_expert=self.models, predictions_expert=self.pred, meta={"bloc": 1})

    def test_legacy_defaults_do_not_replace_explicit_historical_settings(self):
        old = A.normalise_config_historique({"n_expert": 4})
        self.assertEqual((old["expert_ratio_min"], old["jeu_version"], old["expert_mode"]),
                         (0.0, 1, "moyenne"))
        explicit = A.normalise_config_historique({"expert_ratio_min": 1.0, "jeu_version": 2})
        self.assertEqual((explicit["expert_ratio_min"], explicit["jeu_version"]), (1.0, 2))

    def test_roundtrip_preserves_oof_and_exact_float_normalization(self):
        self.save()
        p = A.charge_pipeline_bloc(self.path, donnees=self.d, colonnes=self.cols)
        with patch.object(lgb.LGBMRegressor, "fit", side_effect=AssertionError("Aucun fit en replay")):
            Xk, rangs, pred = A.predit_pipeline_bloc(p, self.d)
        expected_fe = np.clip((self.FE - self.se["mean"]) / (self.se["std"] + 1e-8),
                              -5, 5).astype(np.float32)
        np.testing.assert_array_equal(Xk, np.concatenate([self.Xn, expected_fe], axis=1))
        np.testing.assert_array_equal(rangs, self.FE[:, 2:4])
        np.testing.assert_array_equal(pred, self.pred)
        self.assertFalse(np.array_equal(pred, self.pred_final))
        self.assertEqual(p["_normalisation_marche"]["mean"].dtype, np.float32)
        self.assertEqual(p["_normalisation_expert"]["mean"].dtype, np.float64)

    def test_exported_boosters_reproduce_final_experts_without_fit(self):
        self.save()
        p = A.charge_pipeline_bloc(self.path, donnees=self.d)
        models = A.charge_modeles_expert(p)
        got = np.stack([m.predict(self.Xn, num_threads=1) for m in models], axis=1).astype(np.float32)
        np.testing.assert_array_equal(got, self.pred_final)

    def test_changed_values_row_order_or_types_rejected(self):
        self.save()
        changed = self.d.copy()
        changed.loc[7, "x"] = np.float32(changed.loc[7, "x"] + 1)
        for wrong in (changed, self.d.iloc[::-1], self.d.astype({"x": "float64"})):
            with self.subTest(types=wrong.dtypes.to_dict()):
                with self.assertRaisesRegex(ValueError, "empreinte"):
                    A.charge_pipeline_bloc(self.path, donnees=wrong)

    def test_index_is_not_part_of_dataset_identity(self):
        self.save()
        same = self.d.set_axis(np.arange(len(self.d)) + 100)
        A.charge_pipeline_bloc(self.path, donnees=same)

    def test_changed_feature_order_rejected(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "colonnes"):
            A.charge_pipeline_bloc(self.path, donnees=self.d, colonnes=self.cols[::-1])

    def test_corrupted_sidecar_rejected_before_replay(self):
        info = self.save()
        with (self.path.parent / info["predictions"]["fichier"]).open("ab") as fh:
            fh.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "corrompu"):
            A.charge_pipeline_bloc(self.path, donnees=self.d)

    def test_pipeline_never_overwrites_existing_artifacts(self):
        self.save()
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.save()
        self.assertEqual(self.path.read_bytes(), before)

    def test_replays_do_not_overwrite_original_or_each_other(self):
        original = Path(self.tmp.name) / "test_run_bloc1.json"
        original.write_text('{"score": 123}', encoding="utf-8")
        first = A.ecrit_resultat_rejeu("run", 1, {"score": 2}, dossier=self.tmp.name)
        second = A.ecrit_resultat_rejeu("run", 1, {"score": 3}, dossier=self.tmp.name)
        self.assertNotEqual(first, second)
        self.assertEqual(json.loads(original.read_text(encoding="utf-8"))["score"], 123)
        self.assertEqual(json.loads(first.read_text(encoding="utf-8"))["bilan"]["score"], 2)

    def test_checkpoint_references_exact_pipeline_and_fold(self):
        from rejoue_test_bloc import verifie_lien_pipeline
        self.save()
        state = {"bloc": 1, "pipeline": {"fichier": self.path.name,
                                           "sha256": A.empreinte_fichier(self.path)}}
        verifie_lien_pipeline(state, self.path, 1, 2)
        with self.assertRaisesRegex(ValueError, "autre bloc"):
            verifie_lien_pipeline(state, self.path, 2, 2)
        with self.assertRaisesRegex(ValueError, "sans reference"):
            verifie_lien_pipeline({"bloc": 1}, self.path, 1, 2)
        # Un checkpoint ancien peut etre reconstruit explicitement sans lien.
        verifie_lien_pipeline({"bloc": 1}, self.path, 1, 1)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(" ")
        with self.assertRaisesRegex(ValueError, "ne correspond pas"):
            verifie_lien_pipeline(state, self.path, 1, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
