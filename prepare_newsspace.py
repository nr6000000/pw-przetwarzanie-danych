r"""Convert the original AG's corpus dump (newsSpace.bz2, or the extracted newsSpace file) into the project's CSVs.

Choose one or more outputs (file names in brackets are the defaults):

  --raw [raw_newsspace.csv]   the whole dump, both eras and all 9 columns, NULL kept as \N.
                              newsspace_analysis.ipynb reads this file.
  --full [newsspace_era_b.csv]
                              era B only, with the columns
                                  source, category, title, description, date, time, rank, split
  --train [splits]            train.csv, val.csv, test.csv and test-future.csv in this folder, for modelling:
                              the classes in CLASSES only, cleaned, and the training set balanced (see below).

Columns:
  * date / time are pubdate split into YYYY-MM-DD and HH:MM:SS
  * an empty cell means NULL (description is NULL or empty in the dump)
  * split: rows from 2008 (1 Jan - 20 Feb 2008) are 'test-future', a later test set for checking how models
    generalise to newer articles. Earlier rows are 'train' / 'val' / 'test', about 70 / 15 / 15, chosen by a hash of
    the normalised title: about a third of the titles repeat, and this keeps every copy of an article in the same
    split.

--train cleaning and balancing:
  * only rows of the classes in CLASSES
  * titles that appear under more than one of these classes are dropped (the same article with different labels)
  * every article is kept once (titles compared ignoring case and spacing); the first copy in the dump wins, so
    test-future has no article that already appeared before 2008
  * train.csv is balanced: the same number of articles per class, the size of the smallest class but at most
    MAX_PER_CLASS.
  * val.csv, test.csv and test-future.csv keep EVAL_SHARE (half) of each class's clean articles, so they keep
    their natural class mix; use macro-F1.
  * all sampling uses fixed seeds (one generator per split), so the files are identical on every run
  * the four files have the --full columns without split

Era B: the dump holds two crawls separated by a 15-month gap with no data at all. The older one (era A,
Aug 2004 - Feb 2005) has no date for ~69% of its rows and is only roughly in order, so it is dropped. Every row
dated from ERA_B_START on is kept: era B, May 2006 - Feb 2008, 686,376 rows, all dated.
See newsspace_analysis.ipynb for the analysis behind this.

The dump is a MySQL text dump (SELECT ... INTO OUTFILE defaults):
  * one record per row, columns separated by TAB, rows ended by LF
  * NULL is written as \N; an unknown pubdate is the zero date 0000-00-00 00:00:00
  * inside a field, \ , TAB and LF are escaped with a backslash (a backslash followed by a real TAB / LF,
    or by another backslash)
  * encoding is mixed: mostly UTF-8, with some Latin-1/cp1252 rows

Usage (the dump comes first, because each flag may take a file name):
    python prepare_newsspace.py [newsSpace.bz2] [--raw [CSV]] [--full [CSV]] [--train [DIR]]
"""
import argparse
import bz2
import collections
import contextlib
import csv
import hashlib
import os
import random
import re

COLUMNS = ("source", "url", "title", "image", "category", "description", "rank", "pubdate", "video")
OUTPUT_COLUMNS = ("source", "category", "title", "description", "date", "time", "rank", "split")
ERA_B_START = "2006-01-01"   # any date inside the Feb 2005 - May 2006 gap separates the eras
FUTURE_START = "2008-01-01"  # era-B rows from this date on form the test-future set
SPLITS = ((0.70, "train"), (0.85, "val"), (1.00, "test"))   # cumulative shares for the rows before FUTURE_START
SPLIT_NAMES = ("train", "val", "test", "test-future")

CLASSES = ("Sci/Tech", "World", "Business", "Entertainment", "Sports")   # classes kept by --train
MAX_PER_CLASS = 20_000       # training-set limit per class
EVAL_SHARE = 0.5             # share of each class's clean val / test / test-future articles that is kept
SEED = 42                    # for all sampling in --train

NULL = "\\N"
_ESC = {b"0": b"\0", b"b": b"\b", b"n": b"\n", b"r": b"\r", b"t": b"\t", b"Z": b"\x1a"}
_ESC_RE = re.compile(rb"\\(.)", re.S)


def _ends_escaped(b):
    """True if b ends with an odd number of backslashes."""
    return (len(b) - len(b.rstrip(b"\\"))) % 2 == 1


def _records(f):
    """Yield raw records (bytes, without the final LF); LF preceded by an odd number of backslashes does not end a record."""
    buf = []
    for line in f:
        buf.append(line)
        if line.endswith(b"\n") and not _ends_escaped(line[:-1]):
            yield b"".join(buf)[:-1]
            buf = []
    if buf:
        yield b"".join(buf)


def _split_fields(rec):
    """Split on TABs that are not escaped."""
    fields, cur = [], None
    for part in rec.split(b"\t"):
        cur = part if cur is None else cur + b"\t" + part
        if not _ends_escaped(cur):
            fields.append(cur)
            cur = None
    if cur is not None:
        fields.append(cur)
    return fields


def _unescape(b):
    if b == b"\\N":
        return None
    if b"\\" in b:
        b = _ESC_RE.sub(lambda m: _ESC.get(m.group(1), m.group(1)), b)
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return b.decode("cp1252")
        except UnicodeDecodeError:
            return b.decode("latin-1")


def read_dump(path):
    """Yield each record of the dump as a list of 9 decoded strings (None for NULL), in file order."""
    opener = bz2.open if str(path).endswith(".bz2") else open
    with opener(path, "rb") as f:
        for rec in _records(f):
            fields = _split_fields(rec)
            if len(fields) != len(COLUMNS):
                raise ValueError("expected %d fields, got %d: %r" % (len(COLUMNS), len(fields), rec[:200]))
            yield [_unescape(x) for x in fields]


def title_key(title):
    """Title compared ignoring case and spacing: two rows with the same key are the same article."""
    return " ".join(title.split()).lower()


def split_of(title, date):
    """'test-future' from FUTURE_START on; before that train / val / test by a stable hash of the title key."""
    if date >= FUTURE_START:
        return "test-future"
    u = int.from_bytes(hashlib.md5(title_key(title).encode("utf-8")).digest()[:8], "big") / 2 ** 64
    return next(name for limit, name in SPLITS if u < limit)


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_training_files(pool, folder):
    """pool: (key, split, row) for every era-B row of CLASSES, in dump order. Clean, balance and write four CSVs."""
    labels = collections.defaultdict(set)
    for key, _, row in pool:
        labels[key].add(row[1])
    clean = {s: {c: [] for c in CLASSES} for s in SPLIT_NAMES}
    seen = set()
    for i, (key, split, row) in enumerate(pool):
        if len(labels[key]) == 1 and key not in seen:
            seen.add(key)
            clean[split][row[1]].append((i, row))

    n = min(MAX_PER_CLASS, min(len(clean["train"][c]) for c in CLASSES))
    sizes = {s: (lambda k: int(k * EVAL_SHARE)) for s in SPLIT_NAMES}
    sizes["train"] = lambda k: n
    for s, size in sizes.items():
        rng = random.Random("%d-%s" % (SEED, s))   # own seeded generator per split: changing one leaves the others
        clean[s] = {c: rng.sample(clean[s][c], size(len(clean[s][c]))) for c in CLASSES}

    os.makedirs(folder, exist_ok=True)
    print("--train: %d articles per class in train.csv (limit %d); val, test and test-future keep %.0f%% "
          "of each class" % (n, MAX_PER_CLASS, 100 * EVAL_SHARE))
    print("    %-14s" % "" + "".join("%13s" % s for s in SPLIT_NAMES))
    for c in CLASSES:
        print("    %-14s" % c + "".join("%13d" % len(clean[s][c]) for s in SPLIT_NAMES))
    for s in SPLIT_NAMES:
        rows = sorted(r for c in CLASSES for r in clean[s][c])          # back to dump order
        path = os.path.join(folder, s + ".csv")
        write_csv(path, OUTPUT_COLUMNS[:-1], (row for _, row in rows))
        print("    wrote %d rows to %s" % (len(rows), path))


def main():
    ap = argparse.ArgumentParser(description="Convert the newsSpace dump to the project's CSVs.",
                                 epilog="Put the dump before the flags, because each flag may take a file name.")
    ap.add_argument("source", nargs="?", default="newsSpace.bz2",
                    help="the original dump: newsSpace.bz2 or the extracted newsSpace file (default: %(default)s)")
    ap.add_argument("--raw", nargs="?", const="raw_newsspace.csv", metavar="CSV",
                    help="write the whole dump (both eras, all columns, NULL as \\N) (default: %(const)s)")
    ap.add_argument("--full", nargs="?", const="newsspace_era_b.csv", metavar="CSV",
                    help="write era B with the selected columns and the split column (default: %(const)s)")
    ap.add_argument("--train", nargs="?", const="splits", metavar="DIR",
                    help="write cleaned train / val / test / test-future CSVs, train balanced, the others halved "
                         "(default: %(const)s)")
    args = ap.parse_args()
    if not (args.raw or args.full or args.train):
        ap.error("choose at least one output: --raw, --full or --train")

    total = kept = 0
    first_kept = None
    split_counts = collections.Counter()
    pool = []
    with contextlib.ExitStack() as stack:
        raw = full = None
        if args.raw:
            raw = csv.writer(stack.enter_context(open(args.raw, "w", encoding="utf-8", newline="")))
            raw.writerow(COLUMNS)
        if args.full:
            full = csv.writer(stack.enter_context(open(args.full, "w", encoding="utf-8", newline="")))
            full.writerow(OUTPUT_COLUMNS)

        for row in read_dump(args.source):
            if raw:
                raw.writerow([NULL if v is None else v for v in row])
            r = dict(zip(COLUMNS, row))
            pubdate = r["pubdate"] or ""          # zero date '0000-...' sorts before ERA_B_START
            if pubdate >= ERA_B_START:
                if first_kept is None:
                    first_kept = total
                date, _, time = pubdate.partition(" ")
                split = split_of(r["title"], date)
                out = [r["source"], r["category"], r["title"], r["description"] or "", date, time, r["rank"] or ""]
                if full:
                    full.writerow(out + [split])
                if args.train and r["category"] in CLASSES:
                    pool.append((title_key(r["title"]), split, out))
                split_counts[split] += 1
                kept += 1
            elif first_kept is not None:
                raise ValueError("row %d is undated or older than era B, but era B started at row %d"
                                 % (total, first_kept))
            total += 1

    print("read %d rows from %s; era B is rows %d-%d (%d rows)" % (total, args.source, first_kept, total - 1, kept))
    before = kept - split_counts["test-future"]
    print("split: " + ", ".join("%s %d (%.1f%% of %s)" % (s, split_counts[s], 100 * split_counts[s] / n, of)
                                 for s, n, of in (("train", before, "pre-2008"), ("val", before, "pre-2008"),
                                                  ("test", before, "pre-2008"), ("test-future", kept, "era B"))))
    if args.raw:
        print("--raw: wrote all %d rows to %s" % (total, args.raw))
    if args.full:
        print("--full: wrote %d era-B rows to %s" % (kept, args.full))
    if args.train:
        write_training_files(pool, args.train)


if __name__ == "__main__":
    main()
