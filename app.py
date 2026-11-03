
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 Real Estate Investment Advisor")
st.markdown(
    "Explore Indian property markets, compare properties "
    "and identify opportunities using data-driven analysis."
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "india_housing_prices_cleaned.zip"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, compression="zip") 

try:
    df = load_data()
except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Property Filters")

filtered = df.copy()

states = sorted(df["State"].dropna().unique().tolist())
selected_states = st.sidebar.multiselect(
    "Select State", states, default=states
)
if selected_states:
    filtered = filtered[filtered["State"].isin(selected_states)]

cities = sorted(filtered["City"].dropna().unique().tolist())
selected_cities = st.sidebar.multiselect(
    "Select City", cities, default=cities
)
if selected_cities:
    filtered = filtered[filtered["City"].isin(selected_cities)]

property_types = sorted(
    filtered["Property_Type"].dropna().unique().tolist()
)
selected_types = st.sidebar.multiselect(
    "Property Type", property_types, default=property_types
)
if selected_types:
    filtered = filtered[filtered["Property_Type"].isin(selected_types)]

bhks = sorted(filtered["BHK"].dropna().unique().tolist())
selected_bhks = st.sidebar.multiselect(
    "BHK", bhks, default=bhks
)
if selected_bhks:
    filtered = filtered[filtered["BHK"].isin(selected_bhks)]

furnishing = sorted(
    filtered["Furnished_Status"].dropna().unique().tolist()
)
selected_furnishing = st.sidebar.multiselect(
    "Furnishing Status", furnishing, default=furnishing
)
if selected_furnishing:
    filtered = filtered[
        filtered["Furnished_Status"].isin(selected_furnishing)
    ]

availability = sorted(
    filtered["Availability_Status"].dropna().unique().tolist()
)
selected_availability = st.sidebar.multiselect(
    "Availability Status", availability, default=availability
)
if selected_availability:
    filtered = filtered[
        filtered["Availability_Status"].isin(selected_availability)
    ]

if not filtered.empty:
    min_price = float(filtered["Price_in_Lakhs"].min())
    max_price = float(filtered["Price_in_Lakhs"].max())

    price_range = st.sidebar.slider(
        "Price Range (₹ Lakhs)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price)
    )

    filtered = filtered[
        filtered["Price_in_Lakhs"].between(
            price_range[0], price_range[1]
        )
    ]

if not filtered.empty:
    min_area = int(filtered["Size_in_SqFt"].min())
    max_area = int(filtered["Size_in_SqFt"].max())

    area_range = st.sidebar.slider(
        "Area Range (Sq Ft)",
        min_value=min_area,
        max_value=max_area,
        value=(min_area, max_area)
    )

    filtered = filtered[
        filtered["Size_in_SqFt"].between(
            area_range[0], area_range[1]
        )
    ]

# --------------------------------------------------
# MARKET OVERVIEW - KPI CARDS
# --------------------------------------------------
st.header("📊 Market Overview")

if filtered.empty:
    st.warning("No properties match your selected filters.")
    st.stop()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Properties", f"{len(filtered):,}")

col2.metric(
    "Average Price",
    f"₹{filtered['Price_in_Lakhs'].mean():.2f} L"
)

col3.metric(
    "Median Price",
    f"₹{filtered['Price_in_Lakhs'].median():.2f} L"
)

col4.metric(
    "Average Price-to-Area Ratio (L/sq ft)",
    f"{filtered['Price_per_SqFt'].mean():.4f}"
)

col5.metric(
    "Ready-to-Move Properties",
    f"{(filtered['Availability_Status'].astype(str).str.lower() == 'ready_to_move').sum():,}"
)

st.divider()

# --------------------------------------------------
# PROPERTY MARKET ANALYSIS - CHARTS
# --------------------------------------------------
st.header("📈 Property Market Analysis")

chart1, chart2 = st.columns(2)

with chart1:
    st.subheader("Property Price Distribution")
    fig = px.histogram(
        filtered,
        x="Price_in_Lakhs",
        nbins=40,
        title="Distribution of Property Prices"
    )
    st.plotly_chart(fig, use_container_width=True)

with chart2:
    st.subheader("Average Price by City")
    city_prices = (
        filtered.groupby("City", as_index=False)["Price_in_Lakhs"]
        .mean()
        .sort_values("Price_in_Lakhs", ascending=False)
        .head(15)
    )
    fig = px.bar(
        city_prices,
        x="City",
        y="Price_in_Lakhs",
        title="Top 15 Cities by Average Property Price"
    )
    st.plotly_chart(fig, use_container_width=True)

chart3, chart4 = st.columns(2)

with chart3:
    st.subheader("Property Count by State")
    state_counts = filtered["State"].value_counts().reset_index()
    state_counts.columns = ["State", "Property_Count"]

    fig = px.bar(
        state_counts,
        x="State",
        y="Property_Count",
        title="Properties Across States"
    )
    st.plotly_chart(fig, use_container_width=True)

with chart4:
    st.subheader("Property Type Distribution")
    type_counts = filtered["Property_Type"].value_counts().reset_index()
    type_counts.columns = ["Property_Type", "Count"]

    fig = px.pie(
        type_counts,
        names="Property_Type",
        values="Count",
        title="Property Types"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------
# PROPERTY LISTINGS
# --------------------------------------------------
st.header("🏠 Property Listings")

display_columns = [
    "ID", "State", "City", "Locality",
    "Property_Type", "BHK", "Size_in_SqFt",
    "Price_in_Lakhs", "Price_per_SqFt",
    "Furnished_Status", "Availability_Status"
]

sort_column = st.selectbox(
    "Sort properties by",
    ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]
)

ascending = st.checkbox("Show lowest values first", value=True)

sorted_properties = filtered.sort_values(
    sort_column, ascending=ascending
)

st.dataframe(
    sorted_properties[display_columns],
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# PROPERTY DETAILS
# --------------------------------------------------
st.header("🔍 Property Details")

property_ids = filtered["ID"].tolist()

selected_id = st.selectbox(
    "Choose a property ID",
    property_ids
)

property_info = filtered[
    filtered["ID"] == selected_id
].iloc[0]

detail_columns = st.columns(3)

for index, column in enumerate(df.columns):
    with detail_columns[index % 3]:
        st.write(f"**{column.replace('_', ' ')}**")
        st.write(str(property_info[column]))

# --------------------------------------------------
# PROPERTY COMPARISON
# --------------------------------------------------
st.divider()
st.header("⚖️ Compare Properties")

compare_ids = st.multiselect(
    "Select properties to compare",
    property_ids,
    max_selections=3
)

if compare_ids:
    comparison = filtered[filtered["ID"].isin(compare_ids)]

    comparison_columns = [
        "ID", "City", "Locality", "Property_Type",
        "BHK", "Size_in_SqFt", "Price_in_Lakhs",
        "Price_per_SqFt", "Amenities", "Security",
        "Parking_Space", "Public_Transport_Accessibility"
    ]

    st.dataframe(
        comparison[comparison_columns],
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# RULE-BASED INVESTMENT SCREENING
# --------------------------------------------------
st.divider()
st.header("💡 Investment Screening")

st.caption(
    "Screening indicator only. A price below the city median "
    "does not guarantee a profitable investment."
)

city_medians = filtered.groupby("City")[
    "Price_in_Lakhs"
].transform("median")

screened = filtered.copy()
screened["City_Median_Price"] = city_medians

screened["Screening_Result"] = screened.apply(
    lambda row: (
        "Below city median - review further"
        if row["Price_in_Lakhs"] < row["City_Median_Price"]
        else "At or above city median"
    ),
    axis=1
)

below_median = screened[
    screened["Screening_Result"] ==
    "Below city median - review further"
]

st.metric(
    "Properties Below Filtered City Median",
    f"{len(below_median):,}"
)

st.dataframe(
    below_median[
        [
            "ID", "City", "Locality", "Property_Type",
            "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "City_Median_Price", "Screening_Result"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------
st.info(
    "This tool supports property comparison and preliminary screening. "
    "It does not predict future prices or guarantee investment returns. "
    "Verify property documents, condition, location and actual market rates "
    "before making an investment decision."
)

st.caption("Real Estate Investment Advisor | Academic Project")
