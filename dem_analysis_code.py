import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# --- Page Config ---
st.set_page_config(page_title="Supply Chain Optimizer: 30-Day Edition", layout="wide")

st.title("🎯 The Golden Window: 30-Day Optimization Engine")
st.markdown("""
This engine tests every combination of **Window Size** (up to 30 days) and **Start Date Offset**. 
The goal is to find the 'Scientific Best' logistics cycle where your demand becomes **Statistically Normal**.
""")

# --- 1. Sidebar: Data & Search Range ---
st.sidebar.header("1. Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Search Parameters")
# Increased the max window to 30 as requested
max_test_window = st.sidebar.slider("Max Window to Test (Days)", 7, 30, 30)
p_threshold = 0.05

target_col = "Order_Demand"
data_series_daily = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            xl = pd.ExcelFile(uploaded_file)
            sheet = st.sidebar.selectbox("Select Sheet", xl.sheet_names)
            df_raw = xl.parse(sheet)
        
        # Data Cleaning
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
        df_raw = df_raw.dropna(subset=['Date', target_col])
        
        # AGGREGATION: Summing orders by day
        df_daily = df_raw.groupby('Date')[target_col].sum().sort_index()
        
        # 0-FILLING: Creating a continuous timeline
        full_range = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
        data_series_daily = df_daily.reindex(full_range, fill_value=0)
        
        st.sidebar.success(f"Analyzed {len(data_series_daily)} days of demand.")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    # Sample "Lumpy" Data for Demonstration
    t = pd.date_range(start="2025-01-01", periods=365)
    lumpy = [np.random.randint(1000, 5000) if np.random.random() > 0.9 else 0 for _ in range(365)]
    data_series_daily = pd.Series(lumpy, index=t)
    st.info("💡 Using sample data. Upload your file to run the 30-day optimization.")

# --- 2. THE 30-DAY OPTIMIZATION LOOP ---
with st.spinner("Running 30-day grid search... This checks ~465 scenarios."):
    results = []
    for w in range(1, max_test_window + 1):
        for o in range(w):
            temp_start = data_series_daily.index.min() + pd.Timedelta(days=o)
            temp_data = data_series_daily[data_series_daily.index >= temp_start]
            clubbed = temp_data.resample(f'{w}D').sum()
            
            # Shapiro test requires at least 3 buckets
            if len(clubbed) >= 3:
                # We check the normality of the aggregated buckets
                _, p = stats.shapiro(clubbed)
                results.append({'Window (Days)': w, 'Offset (Days)': o, 'p_value': p})

df_opt = pd.DataFrame(results)

# --- 3. THE LEADERBOARD ---
st.subheader(f"🏆 Top Performers: Scenarios with p > {p_threshold}")
df_leaderboard = df_opt[df_opt['p_value'] > p_threshold].sort_values(by='p_value', ascending=False)

if not df_leaderboard.empty:
    display_df = df_leaderboard.copy()
    display_df['Normality Score'] = (display_df['p_value'] * 100).round(1).astype(str) + "%"
    st.dataframe(display_df[['Window (Days)', 'Offset (Days)', 'p_value', 'Normality Score']], 
                 use_container_width=True, hide_index=True)
else:
    st.warning("No scenarios found with p > 0.05. Try a larger window or check for extreme outliers.")

# --- 4. VISUALIZATION & HEATMAP ---
st.divider()
c_heat, c_sim = st.columns([2, 1])

with c_heat:
    st.subheader("Normality Heatmap (1-30 Days)")
    fig_heat = px.density_heatmap(df_opt, x="Offset (Days)", y="Window (Days)", z="p_value",
                                 color_continuous_scale="Viridis",
                                 title="Finding the Pockets of Stability")
    st.plotly_chart(fig_heat, use_container_width=True)

with c_sim:
    st.subheader("🕹️ Manual Simulator")
    # Auto-select the winner from the table
    best_w = int(df_leaderboard.iloc[0]['Window (Days)']) if not df_leaderboard.empty else 7
    best_o = int(df_leaderboard.iloc[0]['Offset (Days)']) if not df_leaderboard.empty else 0
    
    sel_w = st.slider("Bucket Size (Days)", 1, max_test_window, best_w)
    sel_o = st.slider("Start Offset (Days)", 0, sel_w-1, best_o)

# --- 5. SELECTED SCENARIO DRILLDOWN ---
final_start = data_series_daily.index.min() + pd.Timedelta(days=sel_o)
final_series = data_series_daily[data_series_daily.index >= final_start].resample(f"{sel_w}D").sum()

st.subheader(f"Analysis: {sel_w}-Day Window | {sel_o}-Day Offset")
shapiro_p = stats.shapiro(final_series)[1]

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Bucket Timeline**")
    st.plotly_chart(px.bar(final_series, color_discrete_sequence=['#3182CE']), use_container_width=True)
with col2:
    st.markdown("**Bucket Distribution (Histogram)**")
    st.plotly_chart(px.histogram(final_series, nbins=15, color_discrete_sequence=['#38A169']), use_container_width=True)

if shapiro_p > 0.05:
    st.success(f"✅ **Normal Distribution Confirmed (p={shapiro_p:.4f})**. This Logistics Cycle is stable.")
else:
    st.warning(f"⚠️ **Still Non-Normal (p={shapiro_p:.4f})**. Demand remains erratic at this setting.")
