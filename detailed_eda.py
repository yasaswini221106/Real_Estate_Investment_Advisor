import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'DATA', 'india_housing_prices_cleaned.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'OUTPUTS')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=' * 65)
print('REAL ESTATE INVESTMENT ADVISOR - DETAILED EDA')
print('=' * 65)
print('Loading cleaned data...')
df = pd.read_csv(DATA_PATH)
print(f'Loaded {len(df):,} rows and {len(df.columns)} columns.')

# Keep analysis focused on valid positive prices and sizes. Do not alter source data.
analysis = df.copy()
analysis['Price_per_SqFt_Calculated'] = analysis['Price_in_Lakhs'] * 100000 / analysis['Size_in_SqFt']
analysis['Floor_Relationship_Valid'] = analysis['Floor_No'] <= analysis['Total_Floors']
analysis['City_Median_Price'] = analysis.groupby('City')['Price_in_Lakhs'].transform('median')
analysis['Below_City_Median'] = analysis['Price_in_Lakhs'] < analysis['City_Median_Price']

# Price per sq ft in original data appears to be lakhs/sq ft; calculate conventional rupees/sq ft for comparisons.
# Since Price_in_Lakhs is lakhs of rupees, multiplying by 100,000 converts it to rupees.
analysis['Price_per_SqFt_Rupees'] = analysis['Price_per_SqFt'] * 100000

report = []
def add_section(title, content):
    report.append('\n' + '=' * 70 + '\n' + title + '\n' + '=' * 70 + '\n')
    report.append(str(content) + '\n')

add_section('DATASET OVERVIEW', f'Rows: {len(df):,}\nColumns: {len(df.columns)}\nUnique states: {df["State"].nunique()}\nUnique cities: {df["City"].nunique()}\nUnique localities: {df["Locality"].nunique()}\nProperty types: {df["Property_Type"].nunique()}')
add_section('KEY PRICE STATISTICS (PRICE IN LAKHS)', df['Price_in_Lakhs'].describe().round(2).to_string())
add_section('PRICE BY PROPERTY TYPE', df.groupby('Property_Type')['Price_in_Lakhs'].agg(['count','mean','median']).round(2).to_string())
add_section('PRICE BY BHK', df.groupby('BHK')['Price_in_Lakhs'].agg(['count','mean','median']).round(2).to_string())
add_section('TOP 10 CITIES BY MEDIAN PRICE', df.groupby('City')['Price_in_Lakhs'].median().sort_values(ascending=False).head(10).round(2).to_string())
add_section('TOP 5 EXPENSIVE LOCALITIES BY AVERAGE PRICE', df.groupby(['State','City','Locality'])['Price_in_Lakhs'].agg(['count','mean','median']).query('count >= 10').sort_values('mean', ascending=False).head(5).round(2).to_string())
add_section('PROPERTY COUNT BY STATE', df['State'].value_counts().to_string())
add_section('FURNISHING STATUS ANALYSIS', df.groupby('Furnished_Status')['Price_in_Lakhs'].agg(['count','mean','median']).round(2).to_string())
add_section('FACING DIRECTION ANALYSIS', df.groupby('Facing')['Price_in_Lakhs'].agg(['count','mean','median']).round(2).to_string())
add_section('OWNER TYPE ANALYSIS', df.groupby('Owner_Type')['Price_in_Lakhs'].agg(['count','mean','median']).round(2).to_string())
add_section('AVAILABILITY STATUS ANALYSIS', df.groupby('Availability_Status')['Price_in_Lakhs'].agg(['count','mean','median']).round(2).to_string())
add_section('PROPERTY AGE ANALYSIS', df.groupby('Age_of_Property')['Price_in_Lakhs'].median().describe().round(2).to_string())
add_section('BELOW CITY MEDIAN PRICE', f'Properties below their city median: {int(analysis["Below_City_Median"].sum()):,} ({analysis["Below_City_Median"].mean()*100:.1f}%). This is a screening indicator, not proof of a good investment.')
add_section('FLOOR DATA QUALITY', f'Records where Floor_No > Total_Floors: {(~analysis["Floor_Relationship_Valid"]).sum():,} ({(~analysis["Floor_Relationship_Valid"]).mean()*100:.1f}%). These records are retained; floor-based conclusions should be treated cautiously.')

# Amenity presence derived from the text field; keep original Amenities intact.
amenity_values = ['Gym','Garden','Pool','Clubhouse','Playground']
amenity_counts = {amenity: df['Amenities'].fillna('').str.contains(amenity, case=False, regex=False).sum() for amenity in amenity_values}
add_section('AMENITY PRESENCE COUNTS', pd.Series(amenity_counts).sort_values(ascending=False).to_string())

# Correlation for numeric fields, excluding ID as an identifier.
numeric_cols = ['BHK','Size_in_SqFt','Price_in_Lakhs','Price_per_SqFt','Year_Built','Age_of_Property','Nearby_Schools','Nearby_Hospitals','Floor_No','Total_Floors']
corr = df[numeric_cols].corr(numeric_only=True)
add_section('NUMERIC CORRELATION MATRIX', corr.round(3).to_string())

report_path = os.path.join(OUTPUT_DIR, 'detailed_eda_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.writelines(report)

sns.set_theme(style='whitegrid')

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

# 1 Price vs size scatter sample for performance/readability.
plot_df = df.sample(n=min(5000, len(df)), random_state=42)
plt.figure(figsize=(9, 6))
sns.scatterplot(data=plot_df, x='Size_in_SqFt', y='Price_in_Lakhs', hue='Property_Type', alpha=0.55, s=18)
plt.title('Property Price vs Size (sample of up to 5,000 records)')
plt.xlabel('Size (sq. ft.)'); plt.ylabel('Price (₹ lakhs)')
save_plot('price_vs_size.png')

# 2 Top city median prices
city_prices = df.groupby('City')['Price_in_Lakhs'].median().sort_values().tail(15)
plt.figure(figsize=(10, 7)); city_prices.plot(kind='barh')
plt.title('15 Cities with Highest Median Property Prices'); plt.xlabel('Median price (₹ lakhs)'); plt.ylabel('City')
save_plot('top_city_median_prices.png')

# 3 State property count
state_counts = df['State'].value_counts().head(20).sort_values()
plt.figure(figsize=(10, 7)); state_counts.plot(kind='barh')
plt.title('Property Count by State (top 20)'); plt.xlabel('Number of properties'); plt.ylabel('State')
save_plot('property_count_by_state.png')

# 4 Property type distribution
plt.figure(figsize=(8, 5)); df['Property_Type'].value_counts().plot(kind='bar')
plt.title('Property Type Distribution'); plt.xlabel('Property type'); plt.ylabel('Count'); plt.xticks(rotation=20)
save_plot('property_type_distribution.png')

# 5 BHK distribution by city: use the most represented cities.
bhk_city = pd.crosstab(df['City'], df['BHK'])
top_cities = df['City'].value_counts().head(12).index
plt.figure(figsize=(11, 6)); bhk_city.loc[bhk_city.index.intersection(top_cities)].plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('BHK Distribution Across 12 High-Count Cities'); plt.xlabel('City'); plt.ylabel('Property count'); plt.xticks(rotation=45)
save_plot('bhk_distribution_by_city.png')

# 6 Correlation heatmap
plt.figure(figsize=(11, 8)); sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numeric Features')
save_plot('correlation_heatmap.png')

# 7 School and hospital accessibility by price. Counts are not causation.
for col, filename, title in [('Nearby_Schools','schools_vs_price.png','Nearby Schools vs Median Property Price'), ('Nearby_Hospitals','hospitals_vs_price.png','Nearby Hospitals vs Median Property Price')]:
    grouped = df.groupby(col)['Price_in_Lakhs'].median()
    plt.figure(figsize=(8, 5)); grouped.plot(marker='o')
    plt.title(title); plt.xlabel(f'{col} count'); plt.ylabel('Median price (₹ lakhs)')
    save_plot(filename)

# 8 Price/sq ft by property type and state. Source field convention is validated separately.
state_type = df.pivot_table(index='State', columns='Property_Type', values='Price_per_SqFt', aggfunc='median')
plt.figure(figsize=(11, 8)); sns.heatmap(state_type, cmap='viridis')
plt.title('Median Price per SqFt by State and Property Type (source units)')
plt.ylabel('State'); plt.xlabel('Property type')
save_plot('price_per_sqft_state_type.png')

# 9 Age analysis
age_prices = df.groupby('Age_of_Property')['Price_in_Lakhs'].median()
plt.figure(figsize=(9, 5)); age_prices.plot()
plt.title('Property Age vs Median Price'); plt.xlabel('Age of property (years)'); plt.ylabel('Median price (₹ lakhs)')
save_plot('property_age_analysis.png')

# 10 Furnishing, facing, owner, availability comparison charts
for col, filename, title in [
    ('Furnished_Status','furnishing_price.png','Price by Furnishing Status'),
    ('Facing','facing_price.png','Price by Facing Direction'),
    ('Owner_Type','owner_type_price.png','Price by Owner Type'),
    ('Availability_Status','availability_price.png','Price by Availability Status')]:
    stats = df.groupby(col)['Price_in_Lakhs'].median().sort_values(ascending=False)
    plt.figure(figsize=(9, 5)); stats.plot(kind='bar')
    plt.title(title); plt.xlabel(col.replace('_',' ')); plt.ylabel('Median price (₹ lakhs)'); plt.xticks(rotation=25)
    save_plot(filename)

# 11 Amenities median price comparison (property may contain multiple amenities)
amenity_price_rows = []
for amenity in amenity_values:
    mask = df['Amenities'].fillna('').str.contains(amenity, case=False, regex=False)
    amenity_price_rows.append({'Amenity': amenity, 'With amenity median price': df.loc[mask, 'Price_in_Lakhs'].median(), 'Without amenity median price': df.loc[~mask, 'Price_in_Lakhs'].median(), 'Count with amenity': int(mask.sum())})
amenity_df = pd.DataFrame(amenity_price_rows).set_index('Amenity')
amenity_df.to_csv(os.path.join(OUTPUT_DIR, 'amenity_price_comparison.csv'))

# Export below-city-median candidate list and locality ranking.
cols_to_export = ['ID','State','City','Locality','Property_Type','BHK','Size_in_SqFt','Price_in_Lakhs','Price_per_SqFt','Furnished_Status','Amenities','Security','Parking_Space','Public_Transport_Accessibility','Availability_Status','City_Median_Price','Below_City_Median']
analysis.loc[analysis['Below_City_Median'], cols_to_export].to_csv(os.path.join(OUTPUT_DIR, 'properties_below_city_median.csv'), index=False)
locality_rank = df.groupby(['State','City','Locality']).agg(Property_Count=('ID','count'), Average_Price_Lakhs=('Price_in_Lakhs','mean'), Median_Price_Lakhs=('Price_in_Lakhs','median'), Median_Price_per_SqFt=('Price_per_SqFt','median')).reset_index()
locality_rank.sort_values(['Average_Price_Lakhs','Property_Count'], ascending=[False, False]).to_csv(os.path.join(OUTPUT_DIR, 'locality_rankings.csv'), index=False)

print('\nDETAILED EDA COMPLETE')
print(f'Report: {report_path}')
print(f'Charts and CSV outputs saved in: {OUTPUT_DIR}')
print('\nImportant interpretation notes:')
print('- Prices are in lakhs of rupees; source Price_per_SqFt convention is retained in charts.')
print('- Price below city median is only a screening flag, not investment advice.')
print('- Floor inconsistencies are reported and not silently removed or corrected.')
