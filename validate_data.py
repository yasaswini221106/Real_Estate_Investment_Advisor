
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "DATA" / "india_housing_prices_cleaned.csv"
OUTPUT_DIR = PROJECT_DIR / "OUTPUTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA_FILE, low_memory=False)

    print("=" * 60)
    print("REAL ESTATE DATA QUALITY VALIDATION")
    print("=" * 60)

    checks = {}

    # 1. Floor number must not exceed total floors
    if {"Floor_No", "Total_Floors"}.issubset(df.columns):
        bad = (
            (df["Floor_No"] < 0)
            | (df["Total_Floors"] <= 0)
            | (df["Floor_No"] > df["Total_Floors"])
        )
        checks["Invalid floor relationships"] = int(bad.sum())
        print("\nInvalid floor relationships:", int(bad.sum()))

    # 2. Check whether price per square foot is consistent
    if {"Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"}.issubset(df.columns):
        size = df["Size_in_SqFt"]
        expected = df["Price_in_Lakhs"] / size
        difference = (expected - df["Price_per_SqFt"]).abs()

        # Allow small rounding differences
        mismatch = (size <= 0) | (difference > 0.02)
        checks["Price per square foot mismatches"] = int(mismatch.sum())
        print("Price-per-sq-ft mismatches:", int(mismatch.sum()))
        print("Expected unit: lakhs per sq ft")

    # 3. Check age against the year built
    if {"Year_Built", "Age_of_Property"}.issubset(df.columns):
        age_difference = (
            (2025 - df["Year_Built"]) - df["Age_of_Property"]
        ).abs()
        mismatch = age_difference > 1
        checks["Age/year mismatches"] = int(mismatch.sum())
        print("Age/year mismatches (allowing 1 year):", int(mismatch.sum()))

    # 4. Check values outside basic expected ranges
    if "Price_in_Lakhs" in df.columns:
        bad = df["Price_in_Lakhs"] <= 0
        checks["Non-positive prices"] = int(bad.sum())
        print("Non-positive prices:", int(bad.sum()))

    if "Size_in_SqFt" in df.columns:
        bad = df["Size_in_SqFt"] <= 0
        checks["Non-positive property sizes"] = int(bad.sum())
        print("Non-positive sizes:", int(bad.sum()))

    if "Year_Built" in df.columns:
        bad = (df["Year_Built"] < 1800) | (df["Year_Built"] > 2026)
        checks["Unusual construction years"] = int(bad.sum())
        print("Unusual construction years:", int(bad.sum()))

    # 5. Compare prices by BHK
    if {"BHK", "Price_in_Lakhs"}.issubset(df.columns):
        print("\nMedian price by BHK:")
        print(df.groupby("BHK")["Price_in_Lakhs"].median().to_string())

    # 6. Save validation results
    report = OUTPUT_DIR / "data_quality_report.txt"
    with open(report, "w", encoding="utf-8") as file:
        file.write("DATA QUALITY VALIDATION REPORT\n")
        file.write("=" * 40 + "\n")
        file.write(f"Rows checked: {len(df):,}\n\n")
        for name, count in checks.items():
            file.write(f"{name}: {count:,}\n")

    print(f"\nValidation report saved to:\n{report}")
    print("\nValidation finished. No data has been modified.")

if __name__ == "__main__":
    main()
