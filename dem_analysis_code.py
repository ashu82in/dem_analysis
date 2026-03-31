import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import poisson, kstest

# --- Page Config ---
st.set_page_config(page_title="Universal Supply Chain DNA", layout="wide")

st.title("🎯 Universal Demand DNA & Optimizer")
st.markdown("""
Identifying the **best-fit distribution** for your data. 
Testing for **Normal, Lognormal, Gamma, and Poisson** patterns.
""")

# --- 1. Sidebar: Data & Logic ---
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
    lumpy = [np.random.lognormal(mean=7, sigma=1) if np.random.random() > 0.85 else 0 for _ in range(365)]
    data_series_daily = pd.Series(lumpy, index=t)
    st.info("💡 Using sample 'Distributor' data.")

# --- 2. OPTIMIZATION LOOP ---
results = []
with st.spinner("Analyzing all scenarios..."):
    for w in range(1, max_test_window + 1):
        for o in range(w):
            temp_start = data_series_daily.index.min() + pd.Timedelta(days=o)
            temp_data = data_series_daily[data_series_daily.index >= temp_start]
            clubbed = temp_data.resample(f'{w}D').sum()
            test_data = clubbed[clubbed > 0] if ignore_zeros else clubbed
            
            if len(test_data) >= 3 and test_data.std() > 0:
                _, p = stats.shapiro(test_data)
                results.append({'Window': w, 'Offset': o, 'p_value': p})

df_opt = pd.DataFrame(results)

# --- 3. DASHBOARD TABS ---
tab1, tab2 = st.tabs(["🚀 Scenario Leaderboard", "🔬 Surgical DNA Matcher"])

with tab1:
    st.subheader(f"🏆 Top Predictable Windows (p > {p_threshold})")
    df_leaderboard = df_opt[df_opt['p_value'] > p_threshold].sort_values(by='p_value', ascending=False)
    
    if not df_leaderboard.empty:
        # UPDATED: width='stretch' replaces use_container_width
        st.dataframe(df_leaderboard.head(10), width='stretch', hide_index=True)
    else:
        st.warning("⚠️ No 'Normal' windows found.")

    fig_heat = px.density_heatmap(df_opt, x="Offset", y="Window", z="p_value", 
                                 title="Predictability Heatmap", color_continuous_scale="Viridis")
    # UPDATED: width='stretch' replaces use_container_width
    st.plotly_chart(fig_heat, width='stretch')

with tab2:
    st.subheader("Distribution DNA Matching")
    
    c_s1, c_s2 = st.columns(2)
    best_w = int(df_leaderboard.iloc[0]['Window']) if not df_leaderboard.empty else 7
    best_o = int(df_leaderboard.iloc[0]['Offset']) if not df_leaderboard.empty else 0
    
    with c_s1:
        sel_w = st.slider("Select Window Size", 1, max_test_window, best_w)
    with c_s2:
        # --- THE FIX FOR THE SLIDER ERROR ---
        if sel_w > 1:
            sel_o = st.slider("Select Offset", 0, sel_w-1, best_o if best_o < sel_w else 0)
        else:
            st.info("Offset: 0 (Window is 1 day)")
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
    st.info(f"🧬 DNA Match: **{df_dist.iloc[0]['Distribution'].upper()}**")
    st.table(df_dist)

    v1, v2 = st.columns(2)
    with v1:
        # UPDATED: width='stretch' replaces use_container_width
        st.plotly_chart(px.bar(final_series, title="Bucket Timeline"), width='stretch')
    with v2:
        # UPDATED: width='stretch' replaces use_container_width
        st.plotly_chart(px.histogram(fit_data, nbins=15, title="Distribution Shape"), width='stretch')
