"""Prepare at most 20,000 grouped training/validation records and the full AG News test."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re


CLASSES = {"1": "World", "2": "Sports", "3": "Business", "4": "Sci/Tech"}
WORDS = re.compile(r"(?u)\b\w+\b")


def text_key(row):
    return tuple(" ".join(field.casefold().split()) for field in row[1:])


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if any(len(row) != 3 or row[0] not in CLASSES or not row[1].strip() for row in rows):
        raise ValueError(f"{path}: expected class index, nonempty title and description columns")
    if not rows:
        raise ValueError(f"{path}: empty input")
    return rows


def article_groups(rows):
    """Group normalized full texts and pairs with five-word-shingle Jaccard >= 0.9."""
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left, right):
        left, right = find(left), find(right)
        parent[max(left, right)] = min(left, right)

    exact, shingles, frequencies = {}, [], Counter()
    for index, row in enumerate(rows):
        key = text_key(row)
        if key in exact:
            join(index, exact[key])
        else:
            exact[key] = index
        words = WORDS.findall((row[1] + "\n" + row[2]).casefold())
        grams = {hashlib.blake2b("\x1f".join(words[i:i + 5]).encode(), digest_size=16).digest()
                 for i in range(len(words) - 4)}
        shingles.append(grams)
        frequencies.update(grams)
    # A shared global token order and Jaccard prefix filter avoid an all-pairs scan.
    postings, near_pairs = defaultdict(list), 0
    for index in sorted(range(len(rows)), key=lambda i: (len(shingles[i]), i)):
        grams = shingles[index]
        minimum = (9 * len(grams) + 9) // 10
        prefix = sorted(grams, key=lambda word: (frequencies[word], word))[:len(grams) - minimum + 1]
        candidates = {other for word in prefix for other in postings[word] if len(shingles[other]) >= minimum}
        for other in candidates:
            intersection = len(grams & shingles[other])
            if 10 * intersection >= 9 * (len(grams) + len(shingles[other]) - intersection):
                join(index, other)
                near_pairs += 1
        for word in prefix:
            postings[word].append(index)
    return [find(index) for index in range(len(rows))], near_pairs


def prepare(input_dir, output_dir, max_records=20_000, val_fraction=0.15, seed=42):
    if not 4 <= max_records <= 20_000 or not 0 < val_fraction < 1:
        raise ValueError("Use 4 <= max-records <= 20000 and 0 < val-fraction < 1")
    paths = {name: input_dir / f"{name}.csv" for name in ("train", "test")}
    train, test = (read_rows(paths[name]) for name in ("train", "test"))
    if set(row[0] for row in train) != set(CLASSES):
        raise ValueError("Official training input must contain all four AG News classes")
    rows = train + test
    group_ids, near_pairs = article_groups(rows)
    held_out_groups = set(group_ids[len(train):])
    groups = defaultdict(list)
    for index, row in enumerate(train):
        groups[group_ids[index]].append(index)
    linked_to_test, conflicts = [], []
    eligible = defaultdict(list)
    for group_id, indices in groups.items():
        if group_id in held_out_groups:
            linked_to_test.extend(indices)
        elif len({train[index][0] for index in indices}) > 1:
            conflicts.extend(indices)
        else:
            eligible[train[indices[0]][0]].append(indices)
    rng = random.Random(seed)
    selected = {"train": [], "val": [], "test": list(range(len(train), len(rows)))}
    for label in CLASSES:
        candidates = eligible[label]
        rng.shuffle(candidates)
        chosen, count = [], 0
        for group in candidates:
            if count + len(group) <= max_records // 4:
                chosen.append(group)
                count += len(group)
        if len(chosen) < 2:
            raise ValueError(f"Class {CLASSES[label]} needs at least two eligible article groups")
        val_target, val_count = max(1, round(count * val_fraction)), 0
        for position, group in enumerate(chosen):
            destination = "val" if position < len(chosen) - 1 and val_count + len(group) <= val_target else "train"
            selected[destination].extend(group)
            if destination == "val":
                val_count += len(group)
        if not val_count:
            raise ValueError(f"Class {CLASSES[label]} has no group small enough for validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, indices in selected.items():
        indices.sort()
        path = output_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["record_id", "group_id", "category", "title", "description"])
            for index in indices:
                original = index if index < len(train) else index - len(train)
                source = "train" if index < len(train) else "test"
                label, title, description = rows[index]
                writer.writerow([f"ag-news/{source}/{original}", f"ag-news/group/{group_ids[index]}",
                                 CLASSES[label], title, description])
        outputs[name] = {"n_records": len(indices), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "class_counts": dict(Counter(CLASSES[rows[index][0]] for index in indices))}
    manifest = {
        "dataset": "AG News CSV version 3, updated 2015-09-09; recovered from tracked old_approach/ag_news_csv",
        "source_reference": "https://github.com/zhangxiangxiao/Crepe",
        "source_download_verified": False,
        "terms": "Local dataset readme permits research/non-commercial uses; code licences do not license the articles.",
        "seed": seed, "max_train_val_records": max_records, "validation_fraction_target": val_fraction,
        "text": "Original title and description; no HTML, stopword or dictionary processing",
        "grouping": "Normalized full (title, description) OR five-word-shingle Jaccard >= 0.9; transitive components",
        "near_duplicate_method": "Unicode word tokens after casefold; 128-bit BLAKE2b shingles; prefix join with full candidate verification; HTML kept",
        "limitations": "Near check omits texts shorter than five tokens and can miss paraphrases/reordered updates; same titles alone are not identities",
        "test_policy": "All official test rows retained in original order; linked training groups excluded using text only, never test labels",
        "record_id": "ag-news/{source split}/{zero-based original CSV row}",
        "sources": {name: {"path": str(path.resolve()), "n_records": len(train) if name == "train" else len(test),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in paths.items()},
        "near_pairs": near_pairs, "train_rows_linked_to_test": len(linked_to_test),
        "train_rows_in_remaining_label_conflicts": len(conflicts), "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("old_approach/ag_news_csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("splits/ag_news"))
    parser.add_argument("--max-records", type=int, default=20_000)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        manifest = prepare(args.input_dir, args.output_dir, args.max_records, args.val_fraction, args.seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps({name: result["n_records"] for name, result in manifest["outputs"].items()}))


if __name__ == "__main__":
    main()
