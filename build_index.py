"""
AJN N-gram Indexer
==================
Reads a corpus of text organized by year, builds 1-5 gram frequency tables,
applies a corpus-wide minimum-count threshold (like Google Books), and stores
everything in a SQLite database.

Google Books used:  max n=5, min corpus-wide count=40
This tool defaults: max n=5, min corpus-wide count=10

The min-count filter is applied AFTER all years are indexed, on the total count
across the entire corpus -- so a gram seen 2x/year over 10 years (total 20)
passes a threshold of 10, but one seen 15x in a single year but nowhere else
would need threshold <= 15 to survive.

Input layouts supported (auto-detected):
  A) Directory of .txt files with a 4-digit year in the filename or path
  B) Single CSV with columns: year, text  (your Gabriel-style article table)
  C) Directory of subdirs named YYYY/ containing .txt files

Usage:
  python build_index.py --input /path/to/corpus --output ngrams.db
  python build_index.py --input articles.csv --text-col text --year-col year
  python build_index.py --input corpus/ --max-n 5 --min-count 10
"""

import argparse
import csv
csv.field_size_limit(10_000_000)  # handle large OCR text fields
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path


# -- Text normalization --------------------------------------------------------

_PUNCT_RE = re.compile(r"[^a-z0-9\s'-]")
_SPACE_RE = re.compile(r"\s+")

def normalize(text: str) -> list:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    tokens = text.split()
    return [t for t in tokens if len(t) > 1 or t in ("i", "a")]


def extract_ngrams(tokens: list, n: int) -> Counter:
    grams = zip(*[tokens[i:] for i in range(n)])
    return Counter(" ".join(g) for g in grams)


# -- Corpus readers ------------------------------------------------------------

def iter_corpus_from_dir(root: Path) -> dict:
    year_texts = {}
    skipped = 0
    for path in sorted(root.rglob("*.txt")):
        m = re.search(r"(?<!\d)(1[89]\d{2}|20[012]\d)(?!\d)", str(path))
        if not m:
            skipped += 1
            continue
        year = int(m.group(1))
        tokens = normalize(path.read_text(errors="replace"))
        year_texts.setdefault(year, []).extend(tokens)
    if skipped:
        print(f"  [note] skipped {skipped} files (no year in path)", file=sys.stderr)
    return year_texts


def iter_corpus_from_csv(csv_path: Path, year_col: str, text_col: str) -> dict:
    year_texts = {}
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % 1000 == 0:
                print(f"  reading row {i:,}...", end="\r", file=sys.stderr)
            try:
                year = int(float(row[year_col]))
            except (KeyError, ValueError):
                continue
            tokens = normalize(row.get(text_col, "") or "")
            year_texts.setdefault(year, []).extend(tokens)
    print(file=sys.stderr)
    return year_texts


# -- Database ------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS ngrams_raw (
    gram  TEXT    NOT NULL,
    n     INTEGER NOT NULL,
    year  INTEGER NOT NULL,
    count INTEGER NOT NULL,
    freq  REAL    NOT NULL,
    PRIMARY KEY (gram, year)
);
CREATE TABLE IF NOT EXISTS ngrams (
    gram  TEXT    NOT NULL,
    n     INTEGER NOT NULL,
    year  INTEGER NOT NULL,
    count INTEGER NOT NULL,
    freq  REAL    NOT NULL,
    PRIMARY KEY (gram, year)
);
CREATE TABLE IF NOT EXISTS year_totals (
    year        INTEGER PRIMARY KEY,
    total_words INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT
);
CREATE INDEX IF NOT EXISTS idx_raw_gram ON ngrams_raw (gram);
CREATE INDEX IF NOT EXISTS idx_gram     ON ngrams (gram);
CREATE INDEX IF NOT EXISTS idx_gram_n   ON ngrams (gram, n);
CREATE INDEX IF NOT EXISTS idx_year     ON ngrams (year);
"""


def open_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-64000")
    return con


def insert_year(con: sqlite3.Connection, year: int, tokens: list, max_n: int):
    total = len(tokens)
    if total == 0:
        return
    rows = []
    for n in range(1, max_n + 1):
        for gram, cnt in extract_ngrams(tokens, n).items():
            rows.append((gram, n, year, cnt, cnt / total))
    con.executemany(
        "INSERT OR REPLACE INTO ngrams_raw (gram, n, year, count, freq) VALUES (?,?,?,?,?)",
        rows
    )
    con.execute(
        "INSERT OR REPLACE INTO year_totals (year, total_words) VALUES (?,?)",
        (year, total)
    )
    con.commit()
    print(f"  {year}: {total:,} words -> {len(rows):,} entries", file=sys.stderr)


def apply_corpus_min_count(con: sqlite3.Connection, min_count: int):
    """
    Corpus-wide filter: keep only grams whose TOTAL count across ALL years
    >= min_count. Mirrors Google Books methodology ('at least 40 times in
    the corpus', not per year). We default to 10 for AJN's smaller scale.
    """
    print(f"\nApplying corpus-wide min_count = {min_count}...", file=sys.stderr)
    con.execute("DELETE FROM ngrams")
    con.execute("""
        INSERT INTO ngrams (gram, n, year, count, freq)
        SELECT r.gram, r.n, r.year, r.count, r.freq
        FROM   ngrams_raw r
        INNER JOIN (
            SELECT gram FROM ngrams_raw
            GROUP BY gram HAVING SUM(count) >= ?
        ) kept ON r.gram = kept.gram
    """, (min_count,))
    con.commit()

    kept  = con.execute("SELECT COUNT(*) FROM ngrams").fetchone()[0]
    raw   = con.execute("SELECT COUNT(*) FROM ngrams_raw").fetchone()[0]
    u_kept = con.execute("SELECT COUNT(DISTINCT gram) FROM ngrams").fetchone()[0]
    u_raw  = con.execute("SELECT COUNT(DISTINCT gram) FROM ngrams_raw").fetchone()[0]
    pct = 100 * kept // max(raw, 1)
    print(f"  Rows:         {raw:>10,} raw -> {kept:>10,} kept ({pct}%)", file=sys.stderr)
    print(f"  Unique grams: {u_raw:>10,} raw -> {u_kept:>10,} kept", file=sys.stderr)


# -- CLI -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build AJN n-gram index (Google Books-style, corpus-wide threshold)"
    )
    p.add_argument("--input",     required=True)
    p.add_argument("--output",    default="ngrams.db")
    p.add_argument("--max-n",     type=int, default=5,
                   help="Max n-gram size (default: 5, like Google Books)")
    p.add_argument("--min-count", type=int, default=10,
                   help="Corpus-wide min total count (default: 10; Google used 40)")
    p.add_argument("--year-col",  default="year")
    p.add_argument("--text-col",  default="text")
    p.add_argument("--keep-raw",  action="store_true",
                   help="Retain ngrams_raw table after filtering")
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)

    print("AJN N-gram Indexer", file=sys.stderr)
    print(f"  input    : {inp}", file=sys.stderr)
    print(f"  output   : {args.output}", file=sys.stderr)
    print(f"  max n    : {args.max_n}  (Google Books: 5)", file=sys.stderr)
    print(f"  min count: {args.min_count}  (corpus-wide; Google Books used 40)\n", file=sys.stderr)

    if inp.is_dir():
        year_texts = iter_corpus_from_dir(inp)
    elif inp.suffix.lower() == ".csv":
        year_texts = iter_corpus_from_csv(inp, args.year_col, args.text_col)
    else:
        sys.exit(f"ERROR: --input must be a directory or .csv file, got: {inp}")

    if not year_texts:
        sys.exit("ERROR: No text extracted. Check file paths and column names.")

    print(f"Found {len(year_texts)} years: {min(year_texts)}-{max(year_texts)}", file=sys.stderr)
    print("Indexing...\n", file=sys.stderr)

    con = open_db(args.output)

    for year in sorted(year_texts):
        insert_year(con, year, year_texts[year], args.max_n)

    apply_corpus_min_count(con, args.min_count)

    for k, v in [("max_n", str(args.max_n)), ("min_count", str(args.min_count))]:
        con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, v))
    con.commit()

    if not args.keep_raw:
        con.execute("DELETE FROM ngrams_raw")
        con.commit()
        con.isolation_level = None
        con.execute("VACUUM")
        con.isolation_level = ""

    con.close()
    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"\nDone. {args.output}  ({size_mb:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
