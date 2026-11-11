import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import time
import uuid
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import DictionaryCleaner, FreqWordCleaner, HtmlCleaner, StopwordCleaner


@dataclass
class Split:
    path: Path
    sha256: str
    texts: list[str]
    labels: list[str]
    record_ids: list[str]


def load_split(path: Path) -> Split:
    texts, labels, record_ids = [], [], []
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"category", "title", "description"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns are {sorted(required)}")
        for index, row in enumerate(reader):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: malformed CSV record {index}")
            if not row["category"].strip() or not row["title"].strip():
                raise ValueError(f"{path}: empty category or title in record {index}")
            text = row["title"] + "\n" + row["description"]
            record_ids.append(row.get("record_id") or hashlib.sha256(text_key(text).encode("utf-8")).hexdigest())
            texts.append(text)
            labels.append(row["category"])
    if not texts:
        raise ValueError(f"{path}: empty split")
    return Split(path, digest, texts, labels, record_ids)


def text_key(text: str) -> str:
    return " ".join(text.casefold().split())


def validate_splits(splits: dict[str, Split]) -> list[str]:
    if "train" not in splits:
        raise ValueError("The train split is required")
    labels = sorted(set(splits["train"].labels))
    if len(labels) < 2:
        raise ValueError("Training requires at least two classes")
    seen_ids, seen_texts = {}, {}
    for name, split in splits.items():
        unknown = set(split.labels) - set(labels)
        if unknown:
            raise ValueError(f"{name}: classes absent from train: {sorted(unknown)}")
        for index, (record_id, text) in enumerate(zip(split.record_ids, split.texts)):
            if record_id in seen_ids:
                raise ValueError(
                    f"Repeated record ID: {seen_ids[record_id]} and {name}[{index}]. "
                    "Prepare disjoint splits before training."
                )
            seen_ids[record_id] = f"{name}[{index}]"
            key = text_key(text)
            if key in seen_texts and seen_texts[key] != name:
                raise ValueError(f"Duplicate article across {seen_texts[key]} and {name}: {record_id}")
            seen_texts[key] = name
    return labels

def build_model(config: dict, seed: int):
    pipeline = []
    for step in config["pipeline"]:
        if step["name"] == "dummy":
            # sklearn orders string classes alphabetically and argmax breaks ties first.
            model = DummyClassifier(strategy="most_frequent")
        elif step["name"] == "tfidf":
            args = {
                **step["args"], 
                "ngram_range": tuple(step["args"]["ngram_range"])
            }
            model = TfidfVectorizer(**args)
        elif step["name"] == "linear_svc":
            model = LinearSVC(**step["args"], random_state=seed)
        elif step["name"] == "html_cleaner":
            model = HtmlCleaner(**step["args"])
        elif step["name"] == "dictionary_cleaner":
            model = DictionaryCleaner(**step["args"])
        elif step["name"] == "stopword_cleaner":
            model = StopwordCleaner(**step["args"])
        elif step["name"] == "freqword_cleaner":
            model = FreqWordCleaner(**step["args"])
        else:
            raise ValueError(f"Unknown model: {step["name"]}")

        pipeline.append((step["name"], model))
    
    return Pipeline(pipeline)

def evaluate(model, split: Split, labels: list[str]) -> tuple[dict, list[str]]:
    started = time.perf_counter()
    predictions = model.predict(split.texts).tolist()
    inference_seconds = time.perf_counter() - started
    return {
        "n_records": len(split.labels),
        "accuracy": float(accuracy_score(split.labels, predictions)),
        "macro_f1": float(f1_score(
            split.labels, predictions, labels=labels, average="macro", zero_division=0,
        )),
        "label_order": labels,
        "confusion_matrix": confusion_matrix(split.labels, predictions, labels=labels).tolist(),
        "inference_seconds": inference_seconds,
        "records_per_second": len(predictions) / max(inference_seconds, 1e-12),
    }, predictions


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    # if not args.models or len(set(args.models)) != len(args.models):
    #     raise ValueError("Provide at least one model, without duplicates")
    
    # seeds = args.seeds if args.seeds is not None else config["seeds"]
    # if not seeds or len(set(seeds)) != len(seeds):
    #     raise ValueError("Provide at least one seed, without duplicates")
    
    names = ["train", "val"] + (["test"] if args.evaluate_test else [])
    splits = {name: load_split(args.data_dir / f"{name}.csv") for name in names}
    labels = validate_splits(splits)
    train = splits["train"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = args.output_dir / f"{timestamp}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    manifest = {
        "created_at_utc": timestamp,
        "status": "incomplete",
        "seeds": args.seeds,

        "features": "original title + newline + original description",
        "record_id": "source record_id, or sha256(normalised title + description); row_index is zero-based",
        "duplicate_check": "record IDs and normalised full text across splits; see preparation manifest for near duplicates",
        "label_order": labels,
        "b0_tie_break": "alphabetically first of the most frequent training classes",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            **{name.replace("-", "_"): version(name)
               for name in ("scikit-learn", "joblib", "numpy", "scipy")},
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
        },
        "data": {
            name: {"path": str(split.path.resolve()), "sha256": split.sha256,
                   "n_records": len(split.labels), "class_counts": dict(Counter(split.labels))}
            for name, split in splits.items()
        },
    }
    preparation = args.data_dir / "manifest.json"
    if preparation.is_file():
        manifest["preparation_manifest"] = {
            "path": str(preparation.resolve()),
            "sha256": hashlib.sha256(preparation.read_bytes()).hexdigest(),
        }
    write_json(folder / "manifest.json", manifest)

    summary = []
    for model_config_path in args.models:
        name = model_config_path.stem
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        deterministic = model_config["deterministic"]

        for seed in [0] if deterministic else args.seeds:
            model = build_model(model_config, seed)
    
            model_folder = folder / (name if deterministic else f"{name}-seed-{seed}")
            model_folder.mkdir()
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(train.texts, train.labels)
            fit_seconds = time.perf_counter() - started
            result = {"model": name, "seed": seed, "fit_seconds": fit_seconds, "splits": {}}
            # if name == "b0":
            #     result["selected_class"] = str(model.predict([""])[0])
            #     result["train_class_counts"] = dict(Counter(train.labels))
            # else:
            #     result["vocabulary_size"] = len(model.named_steps["tfidf"].vocabulary_)
            for split_name in names[1:]:
                split = splits[split_name]
                metrics, predictions = evaluate(model, split, labels)
                result["splits"][split_name] = metrics
                write_csv(
                    model_folder / f"predictions-{split_name}.csv",
                    ["record_id", "row_index", "y_true", "y_pred"],
                    zip(split.record_ids, range(len(predictions)), split.labels, predictions),
                )
                write_csv(
                    model_folder / f"confusion-{split_name}.csv", ["true/predicted", *labels],
                    ((label, *row) for label, row in zip(labels, metrics["confusion_matrix"])),
                )
                summary.append({"model": name, "seed": seed, "split": split_name,
                                "fit_seconds": fit_seconds,
                                **{key: metrics[key] for key in (
                                    "n_records", "accuracy", "macro_f1", "inference_seconds",
                                )}})
                print(f"{name:2} seed={str(seed):4} {split_name:11} "
                      f"macro-F1={metrics['macro_f1']:.4f} accuracy={metrics['accuracy']:.4f}")
            joblib.dump(model, model_folder / "model.joblib")
            write_json(model_folder / "metrics.json", result)
    write_csv(folder / "summary.csv", summary[0], (row.values() for row in summary))
    groups = defaultdict(list)
    for row in summary:
        groups[row["model"], row["split"]].append(row)
    aggregates = {}
    for (name, split_name), rows in groups.items():
        stats = {"n_runs": len(rows)}
        for metric in ("macro_f1", "accuracy"):
            values = [row[metric] for row in rows]
            stats[metric] = {
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            }
        aggregates.setdefault(name, {})[split_name] = stats
    write_json(folder / "aggregate.json", aggregates)
    manifest["status"] = "complete"
    write_json(folder / "manifest.json", manifest)
    print(f"Results: {folder}")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("splits/"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/"))
    parser.add_argument("--models", type=Path, nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[502561219, 805712426, 54137036],
                        help="B1 seeds; defaults to the configuration (B0 is fitted once)")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="also evaluate the official test.csv after freezing experiments")
    args = parser.parse_args()

    try:
        run(args)
    except (OSError, ValueError, ConvergenceWarning) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
