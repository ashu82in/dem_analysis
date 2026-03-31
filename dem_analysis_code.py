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
This engine identifies the **best-fit distribution** for your data. 
It tests for **Normal, Lognormal, Gamma, and Poisson** patterns to find your true inventory risk.
""")

# --- 1. Sidebar: Data & Logic ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Search Parameters")
max_test_window = st.sidebar.slider("Max Window to Test (Days)", 7, 30, 30)
p_threshold = 0.05

st.sidebar.header("3. Distributor Filters")
ignore_zeros = st.sidebar.toggle("Ignore Zeros (Active Demand Only)", value=True, 
                                 help="Filters out 0-demand buckets to check the 'Shape of the Spikes'.")

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
    # Generate Sample Lognormal Demand (Typical Distributor Profile)
    t = pd.date_range(start="2025-01-01", periods=365)
    lumpy = [np.random.lognormal(mean=7, sigma=1) if np.random.random() > 0.85 else 0 for _ in range(365)]
    data_series_daily = pd.Series(lumpy, index=t)
    st.info("💡 Using sample 'Distributor' data. Upload your file to begin.")

# --- 2. THE OPTIMIZATION LOOP (Scenario Search) ---
results = []
with st.spinner("Running grid search across all windows and offsets..."):
    for w in range(1, max_test_window + 1):
        for o in range(w):
            temp_start = data_series_daily.index.min() + pd.Timedelta(days=o)
            temp_data = data_series_daily[data_series_daily.index >= temp_start]
            clubbed = temp_data.resample(f'{w}D').sum()
            
            # Apply Filter for the Test
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
        st.dataframe(df_leaderboard.head(10), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No 'Normal' windows found. Check the DNA Matcher for alternative distributions.")

    # Heatmap
    fig_heat = px.density_heatmap(df_opt, x="Offset", y="Window", z="p_value", 
                                 title="Predictability Heatmap (High p-value = Most Stable)", 
                                 color_continuous_scale="Viridis")
    st.plotly_chart(fig_heat, use_container_width=True)

with tab2:
    st.subheader("Distribution DNA Matching")
    
    # 4. MANUAL SIMULATOR (Within Tab 2)
    c_s1, c_s2 = st.columns(2)
    best_w = int(df_leaderboard.iloc[0]['Window']) if not df_leaderboard.empty else 7
    best_o = int(df_leaderboard.iloc[0]['Offset']) if not df_leaderboard.empty else 0
    
    with c_s1:
        sel_w = st.slider("Select Window Size", 1, max_test_window, best_w)
    with c_s2:
        sel_o = st.slider("Select Offset", 0, sel_w-1 if sel_w > 1 else 0, best_o if best_o < sel_w else 0)

    # Process Selection
    final_start = data_series_daily.index.min() + pd.Timedelta(days=sel_o)
    final_series = data_series_daily[data_series_daily.index >= final_start].resample(f"{sel_w}D").sum()
    fit_data = final_series[final_series > 0] if ignore_zeros else final_series

    # TEST ALL DISTRIBUTIONS (Including Poisson)
    dist_names = ["norm", "lognorm", "gamma", "poisson"]
    dist_results = []
    
    for name in dist_names:
        try:
            if name == "poisson":
                # Convert to integers for Poisson; Lambda = Mean
                mu = fit_data.mean()
                _, p = kstest(fit_data.astype(int), 'poisson', args=(mu,))
                dist_results.append({"Distribution": "Poisson (Slow Mover)", "p-value": p})
            else:
                dist = getattr(stats, name)
                params = dist.fit(fit_data)
                _, p = kstest(fit_data, name, args=params)
                dist_results.append({"Distribution": name.capitalize(), "p-value": p})
        except:
            continue

    df_dist = pd.DataFrame(dist_results).sort_values(by="p-value", ascending=False)
    winner = df_dist.iloc[0]['Distribution']

    # Display Results
    st.info(f"🧬 **DNA Match:** Your demand in this scenario best fits a **{winner.upper()}** distribution.")
    st.table(df_dist)

    # Visuals
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Bucket Timeline (The 'When')**")
        st.plotly_chart(px.bar(final_series, color_discrete_sequence=['#3182CE']), use_container_width=True)
    with v2:
        st.markdown("**Bucket Distribution (The 'How Much')**")
        st.plotly_chart(px.histogram(fit_data, nbins=15, color_discrete_sequence=['#38A169']), use_container_width=True)

    with st.expander("📄 View Bucket Data Table"):
        st.dataframe(final_series.reset_index(name="Total Demand"), use_container_width=True)
