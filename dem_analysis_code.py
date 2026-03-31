import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Page Config ---
st.set_page_config(page_title="Bucket Optimization Engine", layout="wide")

st.title("🎯 The Golden Window: Bucket Optimization")
st.markdown("""
This engine runs a loop through all possible **Windows** and **Offsets** to find the setup where 
your 'Order_Demand' becomes **Statistically Normal**. 
The 'Golden Window' is the one with the highest p-value.
""")

# --- 1. Sidebar: Data & Optimization Range ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Optimization Range")
max_window = st.sidebar.slider("Max Window to Test (Days)", 7, 31, 14)
target_col = "Order_Demand"

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            xl = pd.ExcelFile(uploaded_file)
            sheet = st.sidebar.selectbox("Select Sheet", xl.sheet_names)
            df_raw = xl.parse(sheet)
        
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
        df_raw = df_raw.dropna(subset=['Date', target_col])
        
        # Aggregate to daily first
        df_daily = df_raw.groupby('Date')[target_col].sum().sort_index()
        full_range = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
        data_series_daily = df_daily.reindex(full_range, fill_value=0)
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    # Sample "Distributor" Data
    t = pd.date_range(start="2025-01-01", periods=200)
    vals = [np.random.randint(500, 2000) if np.random.random() > 0.8 else 0 for _ in range(200)]
    data_series_daily = pd.Series(vals, index=t)
    st.info("💡 Using sample lumpy data. Upload your file to optimize your 'Order_Demand' buckets.")

# --- 2. THE OPTIMIZATION LOOP (Grid Search) ---
st.subheader("🔍 Finding the Golden Window...")

results = []
# Loop through windows (e.g., 1 to 14 days)
for w in range(1, max_window + 1):
    # Loop through all possible offsets for that window
    for o in range(w):
        # Apply offset and resample
        temp_start = data_series_daily.index.min() + pd.Timedelta(days=o)
        temp_data = data_series_daily[data_series_daily.index >= temp_start]
        clubbed = temp_data.resample(f'{w}D').sum()
        
        # We need at least 3 buckets to run a Shapiro test
        if len(clubbed) >= 3:
            _, p = stats.shapiro(clubbed)
            results.append({'Window': w, 'Offset': o, 'p_value': p})

df_opt = pd.DataFrame(results)

# Identify the winner
golden_window = df_opt.loc[df_opt['p_value'].idxmax()]

# --- 3. Visualization ---
col1, col2 = st.columns([2, 1])

with col1:
    # Heatmap of results
    fig_heat = px.density_heatmap(df_opt, x="Offset", y="Window", z="p_value",
                                 title="Normality Heatmap (Darker/Blue = More Predictable)",
                                 labels={'p_value': 'p-value (Normality)'},
                                 color_continuous_scale="Viridis")
    st.plotly_chart(fig_heat, use_container_width=True)

with col2:
    st.success(f"🏆 **Golden Window Found!**")
    st.metric("Optimal Window", f"{int(golden_window['Window'])} Days")
    st.metric("Optimal Offset", f"{int(golden_window['Offset'])} Days")
    st.metric("Max p-value", f"{golden_window['p_value']:.4f}")
    
    if golden_window['p_value'] > 0.05:
        st.write("✨ At this configuration, your demand is **statistically normal**. Your supply chain is optimized for predictability.")
    else:
        st.write("⚠️ Even at the best setting, demand remains 'Lumpy'. Consider a larger window.")

# --- 4. The Resulting Distribution ---
st.divider()
st.subheader(f"Analysis of the Golden Window ({int(golden_window['Window'])}D Window, {int(golden_window['Offset'])}D Offset)")

# Re-calculate the winning bucket
final_start = data_series_daily.index.min() + pd.Timedelta(days=int(golden_window['Offset']))
final_series = data_series_daily[data_series_daily.index >= final_start].resample(f"{int(golden_window['Window'])}D").sum()

c_a, c_b = st.columns(2)
with c_a:
    fig_final_bar = px.bar(final_series, title="Golden Window: Demand per Period", color_discrete_sequence=['#3182CE'])
    st.plotly_chart(fig_final_bar, use_container_width=True)
with c_b:
    fig_final_hist = px.histogram(final_series, title="Golden Window: Distribution (Should look like a Bell)", color_discrete_sequence=['#38A169'])
    st.plotly_chart(fig_final_hist, use_container_width=True)
