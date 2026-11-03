
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "DATA" / "india_housing_prices_cleaned.csv"
OUTPUT_DIR = PROJECT_DIR / "OUTPUTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 60)
    print("REAL ESTATE INVESTMENT ADVISOR - DATA EXPLORATION")
    print("=" * 60)

    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found: {DATA_FILE}")
        return

    print("\nLoading cleaned dataset...")
    df = pd.read_csv(DATA_FILE, low_memory=False)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # 1. Inspect the actual dataset
    print("\n1. COLUMN NAMES")
    print(df.columns.tolist())

    print("\n2. DATA TYPES")
    print(df.dtypes.to_string())

    print("\n3. FIRST FIVE ROWS")
    print(df.head().to_string())

    print("\n4. NUMERICAL SUMMARY")
    print(df.describe(include="number").T.to_string())

    print("\n5. MISSING VALUES")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print(missing.to_string() if not missing.empty else "No missing values found.")

    # Save a detailed report
    report_file = OUTPUT_DIR / "eda_summary.txt"
    with open(report_file, "w", encoding="utf-8") as file:
        file.write("REAL ESTATE INVESTMENT ADVISOR - EDA SUMMARY\n")
        file.write("=" * 55 + "\n\n")
        file.write(f"Rows: {len(df):,}\n")
        file.write(f"Columns: {len(df.columns)}\n\n")
        file.write("COLUMN NAMES\n")
        file.write(str(df.columns.tolist()) + "\n\n")
        file.write("DATA TYPES\n")
        file.write(df.dtypes.to_string() + "\n\n")
        file.write("NUMERICAL SUMMARY\n")
        file.write(df.describe(include="number").T.to_string() + "\n\n")
        file.write("MISSING VALUES\n")
        file.write(
            missing.to_string() if not missing.empty
            else "No missing values found."
        )

    # 2. Create charts using columns that actually exist
    price_column = "Price_in_Lakhs"
    city_column = "City"
    state_column = "State"
    type_column = "Property_Type"
    bhk_column = "BHK"

    if price_column in df.columns:
        prices = pd.to_numeric(df[price_column], errors="coerce").dropna()

        if not prices.empty:
            plt.figure(figsize=(9, 5))
            plt.hist(prices, bins=50)
            plt.title("Distribution of Property Prices")
            plt.xlabel("Price (Lakhs)")
            plt.ylabel("Number of Properties")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "price_distribution.png", dpi=150)
            plt.close()

            print("\nProperty price summary:")
            print(prices.describe().to_string())

    if city_column in df.columns and price_column in df.columns:
        city_prices = df[[city_column, price_column]].copy()
        city_prices[price_column] = pd.to_numeric(
            city_prices[price_column], errors="coerce"
        )
        city_prices = city_prices.dropna()

        if not city_prices.empty:
            top_cities = (
                city_prices.groupby(city_column)[price_column]
                .median()
                .sort_values(ascending=False)
                .head(10)
                .sort_values()
            )

            plt.figure(figsize=(10, 6))
            top_cities.plot(kind="barh")
            plt.title("10 Cities with Highest Median Property Prices")
            plt.xlabel("Median Price (Lakhs)")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "city_median_prices.png", dpi=150)
            plt.close()

            print("\nCities with highest median prices:")
            print(top_cities.sort_values(ascending=False).to_string())

    if type_column in df.columns and price_column in df.columns:
        type_prices = df[[type_column, price_column]].copy()
        type_prices[price_column] = pd.to_numeric(
            type_prices[price_column], errors="coerce"
        )
        type_summary = type_prices.groupby(type_column)[price_column].agg(
            ["count", "median", "mean"]
        ).sort_values("median", ascending=False)

        print("\nPRICE BY PROPERTY TYPE")
        print(type_summary.to_string())
        type_summary.to_csv(OUTPUT_DIR / "price_by_property_type.csv")

    if bhk_column in df.columns and price_column in df.columns:
        bhk_prices = df[[bhk_column, price_column]].copy()
        bhk_prices[bhk_column] = pd.to_numeric(
            bhk_prices[bhk_column], errors="coerce"
        )
        bhk_prices[price_column] = pd.to_numeric(
            bhk_prices[price_column], errors="coerce"
        )
        bhk_summary = bhk_prices.groupby(bhk_column)[price_column].median()

        print("\nMEDIAN PRICE BY BHK")
        print(bhk_summary.to_string())

    print(f"\nEDA report saved: {report_file}")
    print(f"Charts and outputs saved in: {OUTPUT_DIR}")
    print("\nEXPLORATORY DATA ANALYSIS FINISHED!")

if __name__ == "__main__":
    main()
