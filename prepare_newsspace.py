r"""Convert the original AG's corpus dump (newsSpace.bz2, or the extracted newsSpace file) into the project's CSV
format, keeping era B only.

Output (default newsspace_era_b.csv): UTF-8 CSV with a header row
    source, title, description, date, time, rank
  * date / time are pubdate split into YYYY-MM-DD and HH:MM:SS
  * an empty cell means NULL (description is NULL or empty in the dump)

Era B: the dump holds two crawls separated by a 15-month gap with no data at all. The older one (era A,
Aug 2004 - Feb 2005) has no date for ~69% of its rows and is only roughly in order, so it is dropped. Every row
dated from ERA_B_START on is kept: era B, May 2006 - Feb 2008, 686,376 rows, all dated.
See newsspace_analysis.ipynb for the analysis behind this.

With --raw the whole dump (both eras, all 9 columns) is also written as CSV, with NULL kept as \N.
newsspace_analysis.ipynb reads that file.

The dump is a MySQL text dump (SELECT ... INTO OUTFILE defaults):
  * one record per row, columns separated by TAB, rows ended by LF
  * NULL is written as \N; an unknown pubdate is the zero date 0000-00-00 00:00:00
  * inside a field, \ , TAB and LF are escaped with a backslash (a backslash followed by a real TAB / LF,
    or by another backslash)
  * encoding is mixed: mostly UTF-8, with some Latin-1/cp1252 rows

Usage:
    python prepare_newsspace.py [newsSpace.bz2] [-o newsspace_era_b.csv] [--raw raw_newsspace.csv]
"""
import argparse
import bz2
import contextlib
import csv
import re

COLUMNS = ("source", "url", "title", "image", "category", "description", "rank", "pubdate", "video")
OUTPUT_COLUMNS = ("source", "title", "description", "date", "time", "rank")
ERA_B_START = "2006-01-01"   # any date inside the Feb 2005 - May 2006 gap separates the eras

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


def main():
    ap = argparse.ArgumentParser(description="Convert the newsSpace dump to the project CSV format (era B only).")
    ap.add_argument("source", nargs="?", default="newsSpace.bz2",
                    help="the original dump: newsSpace.bz2 or the extracted newsSpace file (default: %(default)s)")
    ap.add_argument("-o", "--output", default="newsspace_era_b.csv", help="era B CSV to write (default: %(default)s)")
    ap.add_argument("--raw", metavar="CSV",
                    help="also write the whole dump (both eras, all columns, NULL as \\N) to this CSV")
    args = ap.parse_args()

    total = kept = 0
    first_kept = None
    with contextlib.ExitStack() as stack:
        out = csv.writer(stack.enter_context(open(args.output, "w", encoding="utf-8", newline="")))
        out.writerow(OUTPUT_COLUMNS)
        raw = None
        if args.raw:
            raw = csv.writer(stack.enter_context(open(args.raw, "w", encoding="utf-8", newline="")))
            raw.writerow(COLUMNS)

        for row in read_dump(args.source):
            if raw:
                raw.writerow([NULL if v is None else v for v in row])
            r = dict(zip(COLUMNS, row))
            pubdate = r["pubdate"] or ""          # zero date '0000-...' sorts before ERA_B_START
            if pubdate >= ERA_B_START:
                if first_kept is None:
                    first_kept = total
                date, _, time = pubdate.partition(" ")
                out.writerow([r["source"], r["title"], r["description"] or "", date, time, r["rank"] or ""])
                kept += 1
            elif first_kept is not None:
                raise ValueError("row %d is undated or older than era B, but era B started at row %d"
                                 % (total, first_kept))
            total += 1

    print("read %d rows from %s" % (total, args.source))
    print("wrote %d era-B rows (rows %d-%d of the dump) to %s" % (kept, first_kept, total - 1, args.output))
    if args.raw:
        print("wrote all %d rows to %s" % (total, args.raw))


if __name__ == "__main__":
    main()
