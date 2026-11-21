"""Loading and cleaning the Udacity e-commerce landing-page A/B dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def load_raw(raw_dir: Path | str = RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    ab = pd.read_csv(raw_dir / "ab_data.csv", parse_dates=["timestamp"])
    countries = pd.read_csv(raw_dir / "countries.csv")
    return ab, countries


def clean_ab_data(ab: pd.DataFrame, countries: pd.DataFrame | None = None,
                  verbose: bool = True) -> pd.DataFrame:
    """Apply the data-quality rules used throughout the project.

    1. Drop rows whose group and landing_page disagree
       (control must see old_page, treatment must see new_page).
       These users were exposed to the "wrong" experience, so we cannot tell
       which effect they measure.
    2. Drop users that appear more than once, keeping the first exposure, so that
       each row is one independent unit (the unit of randomisation is the user).
    3. Attach country for segment analysis.
    """
    log = {"raw_rows": len(ab)}
    aligned = ((ab["group"] == "control") & (ab["landing_page"] == "old_page")) | \
              ((ab["group"] == "treatment") & (ab["landing_page"] == "new_page"))
    df = ab.loc[aligned].copy()
    log["mismatched_rows_removed"] = int((~aligned).sum())

    df = df.sort_values("timestamp")
    dup = df["user_id"].duplicated(keep="first")
    log["duplicate_user_rows_removed"] = int(dup.sum())
    df = df.loc[~dup]

    if countries is not None:
        df = df.merge(countries, on="user_id", how="left", validate="one_to_one")
        log["missing_country"] = int(df["country"].isna().sum())

    df["date"] = df["timestamp"].dt.date
    df["treatment"] = (df["group"] == "treatment").astype(int)
    log["clean_rows"] = len(df)
    if verbose:
        for k, v in log.items():
            print(f"{k:>30}: {v:,}")
    return df.reset_index(drop=True)
