from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "Predictor5.xls"
COUNTRIES_PATH = BASE_DIR / "countries.csv"
OUTPUT_DIR = BASE_DIR / "Predictor_05"

YEAR_PATTERN = re.compile(r"^(\d{4})(?:\.0)?$")


def _ensure_xlrd_installed() -> None:
    try:
        import xlrd  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing optional dependency 'xlrd' required to read .xls files. "
            "Install it with: python -m pip install xlrd"
        ) from exc


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_col = {str(col).strip().lower(): str(col) for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lower_to_col:
            return lower_to_col[key]
    return None


def _detect_header_row(path: Path, max_rows: int = 25) -> int:
    """World Bank .xls files often have metadata rows above the real header."""

    preview = pd.read_excel(path, header=None, nrows=max_rows)
    for i in range(len(preview)):
        row_values = (
            preview.iloc[i]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": ""})
            .tolist()
        )
        if "country name" in row_values and "country code" in row_values:
            return i
    return 0


def _coerce_year(col: object) -> int | None:
    if isinstance(col, (int,)) and 1800 <= col <= 2200:
        return int(col)
    if isinstance(col, float) and col.is_integer() and 1800 <= int(col) <= 2200:
        return int(col)

    text = str(col).strip()
    match = YEAR_PATTERN.match(text)
    if match:
        year = int(match.group(1))
        if 1800 <= year <= 2200:
            return year
    return None


def _load_master_countries(countries_path: Path) -> pd.DataFrame:
    if not countries_path.exists():
        raise FileNotFoundError(
            f"Master country list not found: {countries_path}. This run requires countries.csv."
        )

    df = pd.read_csv(countries_path)
    if "Country Code" not in df.columns or "Country Name" not in df.columns:
        raise KeyError("countries.csv must contain 'Country Name' and 'Country Code' columns.")

    master = (
        df[["Country Name", "Country Code"]]
        .rename(columns={"Country Name": "Country", "Country Code": "Country Code"})
        .copy()
    )
    master["Country"] = master["Country"].astype(str).str.strip()
    master["Country Code"] = master["Country Code"].astype(str).str.strip().str.upper()
    master = master.dropna(subset=["Country", "Country Code"])
    master = master.drop_duplicates(subset=["Country Code"], keep="first")
    master = master[master["Country Code"].str.fullmatch(r"[A-Z]{3}", na=False)]
    master = master.sort_values(["Country"], kind="stable").reset_index(drop=True)
    return master


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}. Place Predictor5.xls next to this script."
        )

    _ensure_xlrd_installed()

    header_row = _detect_header_row(INPUT_PATH)
    df = pd.read_excel(INPUT_PATH, header=header_row)

    country_col = _find_column(df, ["country", "country name", "country_name"])
    code_col = _find_column(df, ["country code", "country_code", "iso3", "iso"])
    if country_col is None or code_col is None:
        raise KeyError(
            "Predictor5.xls: could not detect 'Country Name'/'Country Code' columns. "
            "If your file format changed, open it and confirm the header row."
        )

    year_cols = [c for c in df.columns if _coerce_year(c) is not None]
    if not year_cols:
        raise KeyError(
            "Predictor5.xls: no year columns were detected (expected 1960, 1961, ...)."
        )

    df_long = df.melt(
        id_vars=[country_col, code_col],
        value_vars=year_cols,
        var_name="Year",
        value_name="Value",
    )

    df_long.rename(columns={country_col: "Country", code_col: "Country Code"}, inplace=True)
    df_long["Country"] = df_long["Country"].astype(str).str.strip()
    df_long["Country Code"] = df_long["Country Code"].astype(str).str.strip().str.upper()

    df_long["Year"] = df_long["Year"].map(_coerce_year)
    df_long = df_long.dropna(subset=["Year"])
    df_long["Year"] = df_long["Year"].astype(int)

    df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")

    # Ensure one row per (Country Code, Year)
    df_long = df_long.drop_duplicates(subset=["Country Code", "Year"], keep="first")
    df_long = df_long[["Country Code", "Year", "Value"]].copy()

    master = _load_master_countries(COUNTRIES_PATH)

    OUTPUT_DIR.mkdir(exist_ok=True)
    for old_file in list(OUTPUT_DIR.glob("*_Population_Growth.xlsx")) + list(
        OUTPUT_DIR.glob("*_Population_Growth.csv")
    ):
        old_file.unlink(missing_ok=True)

    # Export only years that have at least one non-null value.
    years = sorted(
        {
            int(y)
            for y in df_long.loc[df_long["Value"].notna(), "Year"].unique().tolist()
        }
    )
    for year in years:
        out_path = OUTPUT_DIR / f"{year}_Population_Growth.csv"

        df_out = master.copy()
        df_out["Year"] = str(year)

        year_data = df_long[df_long["Year"] == year][["Country Code", "Value"]]
        df_out = df_out.merge(year_data, on="Country Code", how="left")
        df_out.to_csv(out_path, index=False)

    print(
        f"✅ Created {len(years)} files in {OUTPUT_DIR.name} ({len(master)} countries per year)."
    )


if __name__ == "__main__":
    main()