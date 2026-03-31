import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# --- Page Config ---
st.set_page_config(page_title="Demand Scenario Simulator", layout="wide")

st.title("🎯 Demand Scenario Simulator: Windows & Offsets")
st.markdown("""
Use the **Heatmap** to find the 'Scientific Best' setup, or use the **Sliders** to manually 
explore how different clubbing scenarios change your 'Order_Demand' patterns.
""")

# --- 1. Sidebar: Data & Optimization Range ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Search Range")
max_test_window = st.sidebar.slider("Max Window to Test in Heatmap", 7, 31, 14)

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
        
        # Aggregate to daily first & fill with 0
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
    st.info("💡 Using sample data. Upload your file to see the 'Order_Demand' heatmap.")

# --- 2. THE OPTIMIZATION HEATMAP (The 'Check for Each Scenario' logic) ---
st.subheader("📊 Scenario Optimization Heatmap")
st.write("Each block below is a unique scenario. The color shows the **Normality Score (p-value)**.")

results = []
for w in range(1, max_test_window + 1):
    for o in range(w):
        temp_start = data_series_daily.index.min() + pd.Timedelta(days=o)
        temp_data = data_series_daily[data_series_daily.index >= temp_start]
        clubbed = temp_data.resample(f'{w}D').sum()
        
        if len(clubbed) >= 3:
            _, p = stats.shapiro(clubbed)
            results.append({'Window': w, 'Offset': o, 'p_value': p})

df_opt = pd.DataFrame(results)

# Plot Heatmap
fig_heat = px.density_heatmap(df_opt, x="Offset", y="Window", z="p_value",
                             color_continuous_scale="Viridis",
                             labels={'p_value': 'Predictability (p-value)'})
st.plotly_chart(fig_heat, use_container_width=True)

# --- 3. MANUAL SCENARIO EXPLORATION (The Sliders) ---
st.divider()
st.subheader("🕹️ Manual Scenario Explorer")
st.write("Adjust the sliders to see exactly how demand 'clubs' together in that specific scenario.")

col_s1, col_s2 = st.columns(2)
with col_s1:
    sel_window = st.slider("Select Window Size (Days)", 1, max_test_window, 7)
with col_s2:
    sel_offset = st.slider("Select Start Offset (Days)", 0, sel_window-1, 0)

# Process the Selected Scenario
final_start = data_series_daily.index.min() + pd.Timedelta(days=sel_offset)
final_series = data_series_daily[data_series_daily.index >= final_start].resample(f"{sel_window}D").sum()

# --- 4. THE GRAPHS FOR THE SELECTED SCENARIO ---
st.subheader(f"Results for Scenario: {sel_window}-Day Window, {sel_offset}-Day Offset")

# Metrics
shapiro_p = stats.shapiro(final_series)[1]
m1, m2, m3 = st.columns(3)
m1.metric("Average Bucket Load", f"{final_series.mean():.0f} units")
m2.metric("Predictability (p-value)", f"{shapiro_p:.4f}")
m3.metric("Status", "Normal/Stable" if shapiro_p > 0.05 else "Lumpy/Erratic")

# The Graphs
fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.markdown("**Bucket Timeline (The 'When')**")
    fig_bar = px.bar(final_series, labels={'index': 'Period Start', 'value': 'Total Demand'},
                    color_discrete_sequence=['#3182CE'])
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with fig_col2:
    st.markdown("**Bucket Distribution (The 'How Much')**")
    fig_hist = px.histogram(final_series, nbins=15, 
                           labels={'value': 'Units per Bucket', 'count': 'Frequency'},
                           color_discrete_sequence=['#38A169'])
    fig_hist.update_layout(bargap=0.1)
    st.plotly_chart(fig_hist, use_container_width=True)

if shapiro_p > 0.05:
    st.success("✨ This scenario is **Statistically Normal**. You can safely use standard inventory math to unlock cash here!")
else:
    st.warning("⚠️ This scenario is still **Non-Normal**. Try increasing the window or changing the offset to find a 'Smoother' pattern.")
