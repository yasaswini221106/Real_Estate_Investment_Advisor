
from pathlib import Path
import pandas as pd

# Find the main project folder automatically
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Input and output file locations
INPUT_FILE = PROJECT_DIR / "DATA" / "india_housing_prices.csv"
OUTPUT_FILE = PROJECT_DIR / "DATA" / "india_housing_prices_cleaned.csv"

def main():
    print("=" * 60)
    print("REAL ESTATE INVESTMENT ADVISOR - DATA CLEANING")
    print("=" * 60)

    # Check that the original dataset exists
    if not INPUT_FILE.exists():
        print(f"ERROR: Dataset not found at:\n{INPUT_FILE}")
        return

    print("\nLoading original dataset...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {len(df.columns)}")

    # Standardize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    # Remove completely empty rows and columns
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    # Remove duplicate rows
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # Clean text fields without changing their meaning
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        df[column] = df[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        df[column] = df[column].replace("", pd.NA)

    # Convert common numeric fields when present
    numeric_candidates = [
        "ID", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
        "Price_per_SqFt", "Year_Built", "Floor_No",
        "Total_Floors", "Age_of_Property", "Nearby_Schools",
        "Nearby_Hospitals"
    ]

    converted_columns = []
    for column in numeric_candidates:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            converted_columns.append(column)

    # Report missing values; do not guess or invent replacements
    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0].sort_values(
        ascending=False
    )

    # Save cleaned dataset
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nCLEANING SUMMARY")
    print("-" * 40)
    print(f"Rows after cleaning: {len(df):,}")
    print(f"Columns after cleaning: {len(df.columns)}")
    print(f"Duplicate rows removed: {duplicate_count}")
    print(f"Numeric columns checked: {len(converted_columns)}")
    print(f"Rows removed as completely empty: see original and final counts")

    print("\nMissing values by column:")
    if missing_values.empty:
        print("No missing values found.")
    else:
        print(missing_values.to_string())

    print(f"\nCleaned dataset saved to:\n{OUTPUT_FILE}")
    print("\nDATA CLEANING FINISHED!")

if __name__ == "__main__":
    main()
