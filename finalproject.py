# Name:       Your Name
# CS230:      Section XXX
# Data:       Top2000CompaniesGlobally.csv
# URL:
#
# Description:
# This program visualizes the top 2000 global companies using sales, profits, assets, and market value data.
# It allows users to filter and analyze companies by country, continent, and sales range, and features various charts and maps.

import streamlit as st  # core Streamlit
import pandas as pd
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # [Extra][SEA1][Extra][SEA2]
import folium  # [Extra][FOLIUM1][Extra][FOLIUM2]
from streamlit_folium import folium_static  # Folium integration
import requests  # [Extra]PACKAGE: currency API
from wordcloud import WordCloud  # [Extra]PACKAGE: word cloud
from collections import Counter

# [ST4] Customized page layout
st.set_page_config(page_title="Global 2000 Company Analysis", layout="wide")

# [DA1] Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("Top2000CompaniesGlobally.csv")
    df.dropna(subset=["Company","Country","Continent","Sales ($billion)","Profits ($billion)","Assets ($billion)","Market Value ($billion)","Latitude_final","Longitude_final"], inplace=True)
    df.columns = df.columns.str.strip()
    for col in ["Sales ($billion)","Profits ($billion)","Assets ($billion)","Market Value ($billion)"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# load data
df = load_data()

# Sidebar filters
st.sidebar.title("Filters")
# [ST1] selectbox for Continent
continent = st.sidebar.selectbox("Select Continent", ["All"] + sorted(df["Continent"].unique().tolist()))

# [ST2] multiselect for Country
if continent != "All":
    available_countries = sorted(df[df["Continent"] == continent]["Country"].unique().tolist())
else:
    available_countries = sorted(df["Country"].unique().tolist())
selected_countries = st.sidebar.multiselect("Select Countries", available_countries)

# [ST3] slider for Sales range
sales_min, sales_max = st.sidebar.slider(
    "Select Sales Range ($billion)",
    float(df["Sales ($billion)"].min()),
    float(df["Sales ($billion)"].max()),
    (float(df["Sales ($billion)"].min()), float(df["Sales ($billion)"].max()))
)

# [Extra] widget: currency conversion
currency_choice = st.sidebar.selectbox("Convert to currency:", ["None","CNY","EUR","JPY","INR"])

# [DA4] Filter by Sales range
df_filtered = df[(df["Sales ($billion)"] >= sales_min) & (df["Sales ($billion)"] <= sales_max)]
# [DA5] Filter by Continent & Country
if continent != "All": df_filtered = df_filtered[df_filtered["Continent"] == continent]
if selected_countries: df_filtered = df_filtered[df_filtered["Country"].isin(selected_countries)]

# [Extra] Call third-party API for live exchange rate and [PY3] error checking
if currency_choice != "None":
    try:
        data = requests.get("https://api.exchangerate-api.com/v4/latest/USD").json()
        rate = data.get("rates", {}).get(currency_choice)
        if rate:
            st.subheader("Real-time Currency Conversion")
            st.write(f"1 USD = {rate} {currency_choice}")
            for metric in ["Sales","Profits","Assets","Market Value"]:
                df_filtered[f"{metric} ({currency_choice})"] = df_filtered[f"{metric} ($billion)"] * rate
            st.subheader(f"Company Financials in {currency_choice}")
            st.write(df_filtered[["Company"] + [f"{m} ({currency_choice})" for m in ["Sales","Profits","Assets","Market Value"]]])
        else:
            st.write(f"{currency_choice} not available.")
    except Exception:
        st.write("Failed to fetch exchange rate.")

# Main display
st.title("Top Global Companies Analysis")
st.write(df_filtered[["Global Rank","Company","Country","Sales ($billion)","Profits ($billion)"]])

# [DA2] Sort & [Extra][DA3] Top10 Sales
st.subheader("Top 10 by Sales")
st.write(df_filtered.sort_values(by="Sales ($billion)", ascending=False).head(10))
# [DA2] Sort & [Extra][DA3] Top10 Profits
st.subheader("Top 10 by Profits")
st.write(df_filtered.sort_values(by="Profits ($billion)", ascending=False).head(10))

# [Extra][DA7] New column & [Extra][DA9] calculate ratio
st.subheader("Top 10 Companies by Sales-to-Asset Ratio")
df_filtered["Sales to Assets"] = df_filtered["Sales ($billion)"] / df_filtered["Assets ($billion)"]
st.write(df_filtered[["Company","Country","Sales to Assets"]].sort_values(by="Sales to Assets", ascending=False).head(10))

# [Extra][DA8] Iterate rows for large assets
st.write("Companies with assets over $1000B:", [r["Company"] for _,r in df_filtered.iterrows() if r["Assets ($billion)"] > 1000])

# Conditional Visuals
# [Extra][CHART1] Avg Profits by Country
if df_filtered["Country"].nunique() > 1:
    st.subheader("Average Profits by Country")
    avg_profits = df_filtered.groupby("Country")["Profits ($billion)"].mean().sort_values(ascending=False).head(10)
    fig,ax = plt.subplots(); avg_profits.plot(kind="bar",ax=ax); ax.set_ylabel("Profits ($billion)"); st.pyplot(fig)
# [Extra][CHART2] Company Count by Continent
if df_filtered["Continent"].nunique() > 1:
    st.subheader("Company Count by Continent")
    pie_vals = df_filtered["Continent"].value_counts()
    fig2,ax2 = plt.subplots(); ax2.pie(pie_vals,labels=pie_vals.index,autopct="%1.1f%%",startangle=90); st.pyplot(fig2)

# [MAP] Basic map
st.subheader("Company HQ Locations")
df_map = df_filtered.rename(columns={"Latitude_final":"latitude","Longitude_final":"longitude"})
st.map(df_map[["latitude","longitude"]].dropna())

# [Extra][FOLIUM1] Interactive Folium Map
st.subheader("Interactive Folium Map")
map1 = folium.Map(location=[20,0],zoom_start=2)
for _,r in df_filtered.iterrows(): folium.Marker(location=[r["Latitude_final"],r["Longitude_final"]],popup=f"{r['Company']} - {r['Country']}").add_to(map1)
folium_static(map1)

# [Extra][FOLIUM2] Company Size Circles
st.subheader("Company Size Circles")
map2 = folium.Map(location=[20,0],zoom_start=2)
for _,r in df_filtered.iterrows(): folium.CircleMarker(location=[r["Latitude_final"],r["Longitude_final"]],radius=5,popup=f"{r['Company']}, Market Value: {r['Market Value ($billion)']}B",fill=True).add_to(map2)
folium_static(map2)

# [Extra][SEA1] Seaborn scatterplot
st.subheader("Profits vs Sales by Continent")
fig3,ax3 = plt.subplots(); sns.scatterplot(data=df_filtered,x="Sales ($billion)",y="Profits ($billion)",hue="Continent",ax=ax3); st.pyplot(fig3)

# [Extra][SEA2] Seaborn boxplot
st.subheader("Assets Distribution per Continent")
fig4,ax4 = plt.subplots(); sns.boxplot(data=df_filtered,x="Continent",y="Assets ($billion)",ax=ax4); st.pyplot(fig4)

# [PY1][PY2] Function with default param and multi-return
def calc_summary_stats(data, metric="Sales ($billion)"):
    try: return data[metric].max(), data[metric].min()
    except: return 0,0
# call default
max_def,min_def = calc_summary_stats(df_filtered); st.write(f"Default Sales Stats – Max: {max_def}, Min: {min_def}")
# call explicit
metric_choice = st.selectbox("Metric for Summary",["Sales ($billion)","Profits ($billion)"])
max_v,min_v = calc_summary_stats(df_filtered,metric_choice); st.write(f"Max {metric_choice}: {max_v}, Min: {min_v}")

# [PY4] List comprehension
st.write("Companies with long names:",[c for c in df_filtered["Company"] if len(c)>20])
# [PY5] Dictionary usage
st.write("Continent company counts:", df_filtered["Continent"].value_counts().to_dict())

# [Extra] Word Cloud
st.subheader("Word Cloud of Company Names")
company_counts = Counter(df_filtered["Company"])
wordcloud = WordCloud(width=800,height=400,background_color="white").generate_from_frequencies(company_counts)
fig_wc,ax_wc = plt.subplots(figsize=(10,5)); ax_wc.imshow(wordcloud,interpolation="bilinear"); ax_wc.axis("off"); st.pyplot(fig_wc)
