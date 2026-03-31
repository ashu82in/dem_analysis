import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# --- Page Config ---
st.set_page_config(page_title="Supply Chain Optimization Engine", layout="wide")

st.title("🎯 The Golden Window: Automated Scenario Search")
st.markdown("""
This engine tests every combination of **Window Size** and **Start Date Offset**. 
The table below identifies the scenarios that turn your 'Lumpy' demand into a **Normal Distribution**.
""")

# --- 1. Sidebar: Data & Search Range ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Search Parameters")
max_test_window = st.sidebar.slider("Max Window to Test (Days)", 7, 31, 14)
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
        
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
        df_raw = df_raw.dropna(subset=['Date', target_col])
        
        # Aggregate to daily & fill with 0
        df_daily = df_raw.groupby('Date')[target_col].sum().sort_index()
        full_range = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
        data_series_daily = df_daily.reindex(full_range, fill_value=0)
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    # Sample "Distributor" Data
    t = pd.date_range(start="2025-01-01", periods=200)
    vals = [np.random.randint(500, 2000) if np.random.random() > 0.8 else 0 for _ in range(200)]
    data_series_daily = pd.Series(vals, index=t)
    st.info("💡 Using sample data. Upload your file to see the optimized 'Order_Demand' table.")

# --- 2. THE OPTIMIZATION LOOP ---
results = []
for w in range(1, max_test_window + 1):
    for o in range(w):
        temp_start = data_series_daily.index.min() + pd.Timedelta(days=o)
        temp_data = data_series_daily[data_series_daily.index >= temp_start]
        clubbed = temp_data.resample(f'{w}D').sum()
        
        if len(clubbed) >= 3:
            _, p = stats.shapiro(clubbed)
            results.append({'Window (Days)': w, 'Offset (Days)': o, 'p_value': p})

df_opt = pd.DataFrame(results)

# --- 3. THE LEADERBOARD TABLE ---
st.subheader(f"🏆 Top Predictable Scenarios (p > {p_threshold})")
st.write("These scenarios pass the 'Normality Test'. Higher p-values mean the demand is more like a stable bell curve.")

# Filter for p > threshold and sort descending
df_leaderboard = df_opt[df_opt['p_value'] > p_threshold].sort_values(by='p_value', ascending=False)

if not df_leaderboard.empty:
    # Formatting for display
    display_df = df_leaderboard.copy()
    display_df['Predictability Score'] = (display_df['p_value'] * 100).round(2).astype(str) + "%"
    st.dataframe(display_df[['Window (Days)', 'Offset (Days)', 'p_value', 'Predictability Score']], 
                 use_container_width=True, hide_index=True)
else:
    st.warning(f"No scenarios found with p > {p_threshold}. Try increasing the Max Window to club more data.")

# --- 4. VISUALIZATION & SLIDERS ---
st.divider()
col_viz, col_ctrl = st.columns([2, 1])

with col_viz:
    st.subheader("Normality Heatmap")
    fig_heat = px.density_heatmap(df_opt, x="Offset (Days)", y="Window (Days)", z="p_value",
                                 color_continuous_scale="Viridis",
                                 title="Darker = More Statistically Stable")
    st.plotly_chart(fig_heat, use_container_width=True)

with col_ctrl:
    st.subheader("🕹️ Manual Simulator")
    # Set default values to the top scenario from the table if it exists
    def_w = int(df_leaderboard.iloc[0]['Window (Days)']) if not df_leaderboard.empty else 7
    def_o = int(df_leaderboard.iloc[0]['Offset (Days)']) if not df_leaderboard.empty else 0
    
    sel_w = st.slider("Select Window Size", 1, max_test_window, def_w)
    sel_o = st.slider("Select Offset", 0, sel_w-1, def_o)

# --- 5. DETAILED VIEW OF SELECTED SCENARIO ---
final_start = data_series_daily.index.min() + pd.Timedelta(days=sel_o)
final_series = data_series_daily[data_series_daily.index >= final_start].resample(f"{sel_w}D").sum()

st.subheader(f"Surgical View: {sel_w}-Day Window | {sel_o}-Day Offset")
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Load Timeline**")
    st.plotly_chart(px.bar(final_series, color_discrete_sequence=['#3182CE']), use_container_width=True)
with c2:
    st.markdown("**Distribution (Histogram)**")
    st.plotly_chart(px.histogram(final_series, nbins=15, color_discrete_sequence=['#38A169']), use_container_width=True)
