
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "DATA" / "india_housing_prices_cleaned.csv"

df = pd.read_csv(file_path, low_memory=False)

floor = df["Floor_No"]
total = df["Total_Floors"]

print("=" * 55)
print("FLOOR DATA DIAGNOSTIC")
print("=" * 55)

print("\nNegative floor numbers:", int((floor < 0).sum()))
print("Zero or negative total floors:", int((total <= 0).sum()))
print("Floor above total floors:", int((floor > total).sum()))

print("\nMost common floor/total-floor combinations:")
print(
    df.groupby(["Floor_No", "Total_Floors"])
      .size()
      .sort_values(ascending=False)
      .head(15)
      .to_string()
)

print("\nSample invalid records:")
invalid = (floor < 0) | (total <= 0) | (floor > total)
print(df.loc[invalid, ["City", "Property_Type", "Floor_No",
                       "Total_Floors"]].head(10).to_string(index=False))

print("\nDiagnostic complete. No data modified.")
