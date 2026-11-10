"""Grouped AG News preparation without changing source texts or consulting test labels."""

import csv
from pathlib import Path
import tempfile
import unittest

import prepare_ag_news


def write_source(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def read_output(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class PreparationTests(unittest.TestCase):
    def test_full_text_and_near_duplicates_group_but_shared_titles_do_not(self):
        title = "Central bank announces new interest rates"
        description = (
            "Officials announced their decision after a meeting that lasted several hours. "
            "The governor said the bank would continue to monitor inflation and employment "
            "before making further changes. Analysts had expected borrowing costs to remain "
            "steady while consumers adjusted to rising prices. Business leaders welcomed "
            "the statement and said stable policy would help companies plan their investments "
            "over the coming year. Financial markets reacted calmly to the news."
        )
        rows = [
            ["3", title, description],
            ["3", "  CENTRAL  BANK ANNOUNCES NEW INTEREST RATES ", description.upper()],
            ["3", title, description + " Today."],
            ["1", title, "Rescue crews delivered food to isolated villages after severe flooding damaged roads."],
        ]

        groups, near_pairs = prepare_ag_news.article_groups(rows)

        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[0], groups[2])
        self.assertNotEqual(groups[0], groups[3])
        self.assertGreater(near_pairs, 0)

    def test_preparation_is_repeatable_preserves_test_and_ignores_test_labels(self):
        topics = [
            ("1", ["Peace talks resume in Geneva", "Government opens new border crossing",
                   "Diplomats sign regional trade agreement", "Parliament approves foreign aid package"],
             "Officials discussed international policy with delegates from neighbouring countries."),
            ("2", ["City win league championship", "United reach national cup final",
                   "Marathon champion sets course record", "Tennis star advances to semifinals"],
             "Players and coaches celebrated their performance with supporters after the match."),
            ("3", ["Banks report stronger quarterly profits", "Retail sales rise over summer",
                   "Oil prices fall as supplies recover", "Airline announces new investment plan"],
             "Company executives said revenue increased while investors welcomed the financial results."),
            ("4", ["New satellite sends first images", "Researchers develop faster computer chip",
                   "Software update improves network security", "Scientists reveal battery technology breakthrough"],
             "The research team presented technical results at an international science conference."),
        ]
        train = [[label, title, description] for label, titles, description in topics for title in titles]
        linked = ["1", "Flood warnings issued across the valley", "Emergency teams prepared shelters for displaced families."]
        conflict = ["1", "Museum opens new archaeology exhibition", "Visitors can see recently discovered ancient objects."]
        train += [linked, conflict, ["3", conflict[1], conflict[2]]]
        test = [["2", *linked[1:]],
                ["2", "National squad returns from tournament", "Fans greeted the players at the airport."],
                ["3", "Factory announces expansion", "The company plans to hire more workers."],
                ["4", "Spacecraft reaches distant asteroid", "Engineers confirmed a successful landing."]]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_source(root / "train.csv", train)
            write_source(root / "test.csv", test)
            first = prepare_ag_news.prepare(root, root / "first", max_records=16)
            write_source(root / "test.csv", [["4", *row[1:]] for row in test])
            prepare_ag_news.prepare(root, root / "second", max_records=16)

            self.assertEqual(first["train_rows_linked_to_test"], 1)
            self.assertEqual(first["train_rows_in_remaining_label_conflicts"], 2)
            self.assertEqual(sum(first["outputs"][name]["n_records"] for name in ("train", "val")), 16)
            train_rows = read_output(root / "first/train.csv")
            val_rows = read_output(root / "first/val.csv")
            self.assertFalse({row["group_id"] for row in train_rows} & {row["group_id"] for row in val_rows})
            for name in ("train", "val"):
                self.assertEqual((root / f"first/{name}.csv").read_bytes(), (root / f"second/{name}.csv").read_bytes())
            for row in train_rows + val_rows:
                original = train[int(row["record_id"].rsplit("/", 1)[1])]
                self.assertEqual((row["title"], row["description"]), tuple(original[1:]))
            actual_test = read_output(root / "first/test.csv")
            self.assertEqual(len(actual_test), len(test))
            for index, row in enumerate(actual_test):
                self.assertEqual(row["record_id"], f"ag-news/test/{index}")
                self.assertEqual((row["title"], row["description"]), tuple(test[index][1:]))
                self.assertEqual(row["category"], prepare_ag_news.CLASSES[test[index][0]])


if __name__ == "__main__":
    unittest.main()
