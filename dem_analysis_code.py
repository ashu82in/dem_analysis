import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from scipy.stats import kstest, norm, lognorm, gamma, poisson

# --- Page Config ---
st.set_page_config(page_title="Supply Chain DNA: Synchronized", layout="wide")

st.title("🎯 Synchronized Demand DNA & Optimizer")
st.markdown("""
This version uses the **Kolmogorov-Smirnov (K-S) Test** for consistency and includes 
a **Data Export** section to view the raw bucketed volumes.
""")

# --- 1. Sidebar ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Search Parameters")
max_test_window = st.sidebar.slider("Max Window to Test (Days)", 7, 30, 30)
p_threshold = 0.05

st.sidebar.header("3. Distributor Filters")
ignore_zeros = st.sidebar.toggle("Ignore Zeros (Active Demand Only)", value=True)

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
        df_daily = df_raw.groupby('Date')[target_col].sum().sort_index()
        full_range = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
        data_series_daily = df_daily.reindex(full_range, fill_value=0)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    t = pd.date_range(start="2026-01-01", periods=365)
    lumpy = [np.random.lognormal(mean=7, sigma=1) if np.random.random() > 0.8 else 0 for _ in range(365)]
    data_series_daily = pd.Series(lumpy, index=t)
    st.info("💡 Using sample data for demonstration.")

# --- 2. SYNCHRONIZED OPTIMIZATION LOOP ---
results = []
with st.spinner("Analyzing scenarios..."):
    for w in range(1, max_test_window + 1):
        for o in range(w):
            temp_start = data_series_daily.index.min() + pd.Timedelta(days=o)
            temp_data = data_series_daily[data_series_daily.index >= temp_start]
            clubbed = temp_data.resample(f'{w}D').sum()
            test_data = clubbed[clubbed > 0] if ignore_zeros else clubbed
            
            if len(test_data) >= 3 and test_data.std() > 0:
                params = norm.fit(test_data)
                _, p = kstest(test_data, 'norm', args=params)
                results.append({'Window': w, 'Offset': o, 'p_value': p})

df_opt = pd.DataFrame(results)

# --- 3. DASHBOARD TABS ---
tab1, tab2, tab3 = st.tabs(["🚀 Scenario Leaderboard", "🔬 Surgical DNA Matcher", "📊 Raw Bucket Data"])

with tab1:
    st.subheader(f"🏆 Top Predictable Windows (K-S p > {p_threshold})")
    df_leaderboard = df_opt[df_opt['p_value'] > p_threshold].sort_values(by='p_value', ascending=False)
    
    if not df_leaderboard.empty:
        st.dataframe(df_leaderboard.head(15), width='stretch', hide_index=True)
    else:
        st.warning("No scenarios found with p > 0.05. Demand remains erratic.")

    fig_heat = px.density_heatmap(df_opt, x="Offset", y="Window", z="p_value", 
                                 title="Predictability Heatmap (K-S Method)", color_continuous_scale="Viridis")
    st.plotly_chart(fig_heat, width='stretch')

# Define the selected data globally so Tab 3 can access it
best_w = int(df_leaderboard.iloc[0]['Window']) if not df_leaderboard.empty else 7
best_o = int(df_leaderboard.iloc[0]['Offset']) if not df_leaderboard.empty else 0

with tab2:
    st.subheader("Distribution DNA Matching")
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        sel_w = st.slider("Select Window Size", 1, max_test_window, best_w)
    with c_s2:
        if sel_w > 1:
            sel_o = st.slider("Select Offset", 0, sel_w-1, best_o if best_o < sel_w else 0)
        else:
            sel_o = 0

    final_start = data_series_daily.index.min() + pd.Timedelta(days=sel_o)
    final_series = data_series_daily[data_series_daily.index >= final_start].resample(f"{sel_w}D").sum()
    fit_data = final_series[final_series > 0] if ignore_zeros else final_series

    # DNA TEST
    dist_names = ["norm", "lognorm", "gamma", "poisson"]
    dist_results = []
    for name in dist_names:
        try:
            if name == "poisson":
                mu = fit_data.mean()
                _, p = kstest(fit_data.astype(int), 'poisson', args=(mu,))
                dist_results.append({"Distribution": "Poisson", "p-value": p})
            else:
                dist = getattr(stats, name)
                params = dist.fit(fit_data)
                _, p = kstest(fit_data, name, args=params)
                dist_results.append({"Distribution": name.capitalize(), "p-value": p})
        except: continue

    df_dist = pd.DataFrame(dist_results).sort_values(by="p-value", ascending=False)
    st.info(f"🧬 DNA Match for {sel_w}D Window: **{df_dist.iloc[0]['Distribution'].upper()}**")
    st.table(df_dist)

    v1, v2 = st.columns(2)
    with v1:
        st.plotly_chart(px.bar(final_series, title="Bucket Timeline"), width='stretch')
    with v2:
        st.plotly_chart(px.histogram(fit_data, nbins=15, title="Distribution Shape (Spikes Only)"), width='stretch')

with tab3:
    st.subheader(f"Detailed View: {sel_w}-Day Window")
    st.write("This table shows the exact demand volume aggregated for each time bucket.")
    
    # Format the bucket data for the user
    df_buckets = final_series.reset_index()
    df_buckets.columns = ['Period Start Date', 'Aggregated Demand']
    
    # Add some helpful stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Buckets", len(df_buckets))
    col2.metric("Avg Demand/Bucket", f"{df_buckets['Aggregated Demand'].mean():.2f}")
    col3.metric("Max Spike", f"{df_buckets['Aggregated Demand'].max():.0f}")

    st.dataframe(df_buckets, width='stretch', hide_index=True)
    
    # Download option
    csv = df_buckets.to_csv(index=False).encode('utf-8')
    st.download_button("Download Bucket Data", csv, f"demand_buckets_{sel_w}d.csv", "text/csv")
