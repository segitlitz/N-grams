"""
AJN N-gram Viewer
=================
A Google Books Ngram Viewer–style Streamlit app for querying ngrams.db.

Run:
  streamlit run viewer.py -- --db ngrams.db
  streamlit run viewer.py          (looks for ngrams.db in current directory)

Features:
  • Multi-term comparison with color-coded lines
  • Raw count OR relative frequency (% of all words that year)
  • Smoothing slider (rolling mean)
  • Year range filter
  • Download trend data as CSV
  • Top-N terms for a given year range (explore mode)
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_DB = "ngrams.db"

import os
def download_db_if_needed(db_path: str):
    if not os.path.exists(db_path):
        import urllib.request
        file_id = "1rakIDm6tD_o-CLrIo2mcSybvR6w4av62"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        print(f"Downloading {db_path} from Google Drive...")
        urllib.request.urlretrieve(url, db_path)
        print("Download complete.")



@st.cache_resource
def get_connection(db_path: str) -> sqlite3.Connection:
    if not Path(db_path).exists():
        st.error(f"Database not found: {db_path}\n\nRun `python build_index.py` first.")
        st.stop()
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


@st.cache_data
def year_range(_con) -> tuple[int, int]:
    row = _con.execute("SELECT MIN(year), MAX(year) FROM year_totals").fetchone()
    return int(row[0]), int(row[1])


@st.cache_data
def fetch_trend(_con, gram: str, y1: int, y2: int) -> pd.DataFrame:
    """Return DataFrame(year, count, freq) for a single gram across a year range."""
    rows = _con.execute(
        """
        SELECT n.year, n.count, n.freq
        FROM   ngrams n
        WHERE  n.gram = ? AND n.year BETWEEN ? AND ?
        ORDER  BY n.year
        """,
        (gram.lower().strip(), y1, y2),
    ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["year", "count", "freq"])
    df = pd.DataFrame(rows, columns=["year", "count", "freq"])
    # fill missing years with 0
    all_years = pd.DataFrame({"year": range(y1, y2 + 1)})
    df = all_years.merge(df, on="year", how="left").fillna(0)
    return df


@st.cache_data
def fetch_top_n(_con, n_gram_size: int, y1: int, y2: int, top_n: int) -> pd.DataFrame:
    """Top-N grams by total frequency over a year window."""
    rows = _con.execute(
        """
        SELECT gram, SUM(count) AS total_count, AVG(freq) AS avg_freq
        FROM   ngrams
        WHERE  n = ? AND year BETWEEN ? AND ?
        GROUP  BY gram
        ORDER  BY total_count DESC
        LIMIT  ?
        """,
        (n_gram_size, y1, y2, top_n),
    ).fetchall()
    return pd.DataFrame(rows, columns=["gram", "total_count", "avg_freq"])


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def main(db_path: str):
    st.set_page_config(
        page_title="AJN N-gram Viewer",
        page_icon="📰",
        layout="wide",
    )

    st.title("📰 AJN N-gram Viewer")
    st.caption(
        "Track word and phrase frequency across the *American Journal of Nursing* corpus, "
        "1900–1909. Inspired by Google Books Ngram Viewer."
    )

    download_db_if_needed(db_path)
    con = get_connection(db_path)
    min_year, max_year = year_range(con)

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Search")

        raw_query = st.text_input(
            "Terms (comma-separated)",
            value="nurse, doctor, patient",
            help="Enter 1–5 terms. Phrases like 'bedside manner' work too.",
        )
        terms = [t.strip() for t in raw_query.split(",") if t.strip()]

        col1, col2 = st.columns(2)
        y1 = col1.number_input("From", min_value=min_year, max_value=max_year, value=min_year)
        y2 = col2.number_input("To",   min_value=min_year, max_value=max_year, value=max_year)

        metric = st.radio(
            "Y-axis",
            ["Relative frequency (%)", "Raw count"],
            index=0,
            help="Relative frequency normalizes by total words per year — better for trend comparison.",
        )

        smoothing = st.slider(
            "Smoothing (rolling mean, years)",
            min_value=1, max_value=15, value=3, step=2,
        )

        st.divider()
        st.header("Explore mode")
        explore = st.checkbox("Show top N terms for selected window")
        if explore:
            n_size   = st.selectbox("N-gram size", [1, 2, 3], index=0)
            top_n    = st.slider("Top N", 10, 100, 25)

    if int(y1) >= int(y2):
        st.warning("'From' year must be less than 'To' year.")
        return

    # ── Main trend chart ──────────────────────────────────────────────────────
    if terms:
        trend_frames = {}
        missing = []
        for term in terms[:8]:   # cap at 8 lines
            df = fetch_trend(con, term, int(y1), int(y2))
            if df["count"].sum() == 0:
                missing.append(term)
            else:
                trend_frames[term] = df

        if missing:
            st.warning(f"Not found in corpus: {', '.join(missing)}")

        if trend_frames:
            use_freq = metric.startswith("Relative")
            col = "freq" if use_freq else "count"
            label = "Frequency (% of all words)" if use_freq else "Raw word count"

            # Build combined wide DataFrame for st.line_chart
            combined = pd.DataFrame()
            for term, df in trend_frames.items():
                s = df.set_index("year")[col]
                if smoothing > 1:
                    s = s.rolling(smoothing, center=True, min_periods=1).mean()
                if use_freq:
                    s = s * 100   # → percentage
                combined[term] = s

            combined.index.name = "year"
            combined.index = combined.index.astype(str)

            st.subheader("Frequency over time")
            st.line_chart(combined, height=420)
            st.caption(f"Y-axis: {label}" + (f" | {smoothing}-year rolling mean" if smoothing > 1 else ""))

            # Download
            csv_bytes = combined.reset_index().to_csv(index=False).encode()
            st.download_button(
                "⬇ Download CSV",
                csv_bytes,
                "ajn_ngram_trends.csv",
                "text/csv",
            )

            # Per-term peak year table
            st.subheader("Peak year by term")
            peak_rows = []
            for term, series in combined.items():
                peak_year = int(series.idxmax())
                peak_val  = float(series.max())
                peak_rows.append({"term": term, "peak year": peak_year,
                                   label: round(peak_val, 6)})
            st.dataframe(pd.DataFrame(peak_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Explore mode: top N ───────────────────────────────────────────────────
    if explore:
        st.subheader(f"Top {top_n} {n_size}-grams, {int(y1)}–{int(y2)}")
        df_top = fetch_top_n(con, n_size, int(y1), int(y2), top_n)
        if df_top.empty:
            st.info("No data for this window.")
        else:
            df_top["avg_freq (%)"] = (df_top["avg_freq"] * 100).round(5)
            st.dataframe(
                df_top[["gram", "total_count", "avg_freq (%)"]],
                use_container_width=True,
                hide_index=True,
            )
            st.bar_chart(df_top.set_index("gram")["total_count"].head(30))

    # ── Footer ────────────────────────────────────────────────────────────────
    with st.expander("About"):
        st.markdown("""
**AJN N-gram Viewer** — Little Devices Lab, Emory University

Methodology mirrors [Google Books Ngram Viewer](https://books.google.com/ngrams):
- Text is lowercased and punctuation-stripped before tokenization
- Frequency = n-gram count ÷ total word tokens that year
- Smoothing applies a centered rolling mean

**Database:** `ngrams.db`  
**Built with:** `build_index.py` — set `--max-n 3` for 1-, 2-, and 3-grams
        """)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Streamlit passes args after `--`
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", default=DEFAULT_DB)
    args, _ = parser.parse_known_args()
    main(args.db)
