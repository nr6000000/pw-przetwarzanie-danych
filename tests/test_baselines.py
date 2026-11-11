"""Baseline metrics, split isolation and a small command-line run."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib

import runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/baselines.json"
TRAIN = [
    ("Business", "Company profits rise as sales recover", "<p>Revenue climbed &amp; profit margins improved.</p>"),
    ("Sci/Tech", "Researchers unveil faster computer chip", "The processor uses less power than earlier models."),
    ("Sports", "United claim victory in cup final", "A late goal secured the team's first trophy."),
    ("World", "Leaders agree on new peace treaty", "Negotiators signed an agreement after months of talks."),
]
VAL = [
    ("Business", "Retail shares rise after strong earnings", "Investors welcomed higher sales and profits."),
    ("Sci/Tech", "New satellite sends first images", "Engineers received detailed pictures of the coast."),
    ("Sports", "City secure league title", "The champions celebrated with fans after the final match."),
    ("World", "Brussels hosts talks on trade pact", "Delegates met to discuss a new international agreement."),
]


class BaselineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def write_split(self, name, rows):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["category", "title", "description"])
            writer.writerows(rows)
        return runner.load_split(path)

    def test_b0_uses_training_majority_and_reports_known_metrics(self):
        train = self.write_split("train.csv", [TRAIN[0], TRAIN[3],
            ("Business", "Banks report higher quarterly earnings", "Lending income increased during the quarter.")])
        val = self.write_split("val.csv", [VAL[3], VAL[0],
            ("World", "Aid shipments reach flooded towns", "Relief teams delivered food and medical supplies.")])
        model = runner.build_model("b0", self.config)
        model.fit(train.texts, train.labels)

        metrics, predictions = runner.evaluate(model, val, ["Business", "World"])

        self.assertEqual(predictions, ["Business"] * 3)
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 0.25)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [2, 0]])

    def test_b1_preserves_html_and_does_not_learn_validation_words(self):
        train = self.write_split("train.csv", TRAIN)
        val = self.write_split("val.csv", VAL)
        model = runner.build_model("b1", self.config)
        model.fit(train.texts, train.labels)
        vocabulary = dict(model.named_steps["tfidf"].vocabulary_)

        runner.evaluate(model, val, sorted(set(train.labels)))

        self.assertEqual(train.texts[0], TRAIN[0][1] + "\n" + TRAIN[0][2])
        self.assertIn("amp", vocabulary)
        self.assertNotIn("brussels", model.named_steps["tfidf"].vocabulary_)
        self.assertEqual(model.named_steps["tfidf"].vocabulary_, vocabulary)

    def test_overlapping_article_is_rejected_ignoring_case_and_spaces(self):
        train = self.write_split("train.csv", TRAIN)
        val = self.write_split("val.csv", [
            ("World", "  LEADERS   AGREE ON NEW PEACE TREATY  ", TRAIN[3][2].upper())])

        with self.assertRaises(ValueError):
            runner.validate_splits({"train": train, "val": val})

    def test_cli_saves_both_models_and_three_seed_reports_without_test_files(self):
        self.write_split("train.csv", TRAIN)
        val = self.write_split("val.csv", VAL)
        output_root = self.root / "results"
        completed = subprocess.run(
            [sys.executable, str(ROOT / "baselines.py"), "--data-dir", str(self.root),
             "--output-dir", str(output_root), "--config", str(CONFIG_PATH)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        folder, = output_root.iterdir()
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(set(manifest["data"]), {"train", "val"})
        self.assertEqual(manifest["seeds"], self.config["seeds"])
        aggregate = json.loads((folder / "aggregate.json").read_text(encoding="utf-8"))
        self.assertEqual(aggregate["b0"]["val"]["n_runs"], 1)
        self.assertEqual(aggregate["b1"]["val"]["n_runs"], 3)
        with (folder / "summary.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4)
        for name in ["b0", *(f"b1-seed-{seed}" for seed in self.config["seeds"])]:
            run = folder / name
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(set(metrics["splits"]), {"val"})
            self.assertTrue((run / "confusion-val.csv").is_file())
            with (run / "predictions-val.csv").open(encoding="utf-8", newline="") as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual([row["record_id"] for row in predictions], val.record_ids)
            self.assertEqual([row["y_true"] for row in predictions], val.labels)
            self.assertEqual([int(row["row_index"]) for row in predictions], list(range(len(VAL))))
            model = joblib.load(run / "model.joblib")
            self.assertEqual(model.predict(val.texts).tolist(), [row["y_pred"] for row in predictions])
            if name == "b0":
                self.assertEqual(metrics["selected_class"], "Business")
                self.assertAlmostEqual(metrics["splits"]["val"]["macro_f1"], 0.1)


if __name__ == "__main__":
    unittest.main()
