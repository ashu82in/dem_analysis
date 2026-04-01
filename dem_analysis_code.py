import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import kstest, norm, lognorm, gamma, poisson
from statsmodels.tsa.seasonal import seasonal_decompose
from concurrent.futures import ProcessPoolExecutor

# --- 1. Page Configuration ---
st.set_page_config(page_title="Universal Supply Chain DNA: Performance Edition", layout="wide")

# --- 2. Session State Management (Anti-Jump & Cache Guard) ---
if 'sel_w' not in st.session_state: st.session_state.sel_w = 7
if 'sel_o' not in st.session_state: st.session_state.sel_o = 0
if 'sim_results' not in st.session_state: st.session_state.sim_results = None
if 'last_file' not in st.session_state: st.session_state.last_file = None
if 'last_settings_key' not in st.session_state: st.session_state.last_settings_key = ""

st.title("🎯 Universal Demand DNA & Optimizer (High Performance)")

# --- 3. THE SURGICAL ENGINE (Optimized for Parallel Cores) ---
def analyze_single_scenario(args):
    """Worker function for parallel processing."""
    window, offset, values, index, mode, ignore_zeros = args
    
    # Rebuild temporary series
    temp_series = pd.Series(values, index=index)
    start_date = index.min() + pd.Timedelta(days=offset)
    temp_series = temp_series[temp_series.index >= start_date]
    
    # Fast Resample
    clubbed = temp_series.resample(f'{window}D').sum()
    
    # Process Data (Decomposition or Raw)
    if "Residuals" in mode and len(clubbed) > 10:
        try:
            decomp = seasonal_decompose(clubbed, model='additive', period=4)
            data = decomp.resid.dropna()
        except: data = clubbed
    else:
        data = clubbed
    
    test_data = data[data > 0] if ignore_zeros else data
    
    if len(test_data) < 3 or test_data.std() == 0:
        return None

    # Multi-DNA Competition
    results = []
    # Normal
    try: results.append(('Normal', kstest(test_data, 'norm', args=norm.fit(test_data))[1]))
    except: pass
    # Lognormal
    try: results.append(('Lognormal', kstest(test_data, 'lognorm', args=lognorm.fit(test_data))[1]))
    except: pass
    # Poisson
    try: results.append(('Poisson', kstest(test_data.astype(int), 'poisson', args=(test_data.mean(),))[1]))
    except: pass
    
    if not results: return None
    best = max(results, key=lambda x: x[1])
    
    return {'Window': window, 'Offset': offset, 'Best_p': best[1], 'Winner': best[0]}

# --- 4. Sidebar Controls ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Global Strategy")
analysis_mode = st.sidebar.selectbox("Analysis Target", ["Raw Demand", "Residuals (Noise Only)"])
max_test_window = st.sidebar.slider("Max Window to Test (Days)", 7, 30, 30)
ignore_zeros = st.sidebar.toggle("Ignore Zeros (Active Demand Only)", value=True)

# Generate a settings key to detect changes
current_settings_key = f"{analysis_mode}_{max_test_window}_{ignore_zeros}"

if st.sidebar.button("🚀 Force Refresh Simulation"):
    st.session_state.sim_results = None

target_col = "Order_Demand"
data_series_daily = None

# --- 5. Data Loading ---
if uploaded_file:
    if st.session_state.last_file != uploaded_file.name:
        st.session_state.last_file = uploaded_file.name
        st.session_state.sim_results = None # Reset on new file

    try:
        if uploaded_file.name.endswith('.csv'): df_raw = pd.read_csv(uploaded_file)
        else: df_raw = pd.read_excel(uploaded_file)
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
        df_daily = df_raw.groupby('Date')[target_col].sum().sort_index()
        full_range = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
        data_series_daily = df_daily.reindex(full_range, fill_value=0)
    except: st.error("Error loading data."); st.stop()
else:
    t = pd.date_range(start="2026-01-01", periods=365)
    data_series_daily = pd.Series(np.random.lognormal(5, 0.8, 365), index=t)
    st.info("💡 Using sample data. Upload your file to start.")

# Auto-reset simulation if global settings changed
if st.session_state.last_settings_key != current_settings_key:
    st.session_state.last_settings_key = current_settings_key
    st.session_state.sim_results = None

# --- 6. THE PARALLEL SIMULATION ENGINE ---
if st.session_state.sim_results is None:
    tasks = []
    for w in range(1, max_test_window + 1):
        for o in range(w):
            tasks.append((w, o, data_series_daily.values, data_series_daily.index, analysis_mode, ignore_zeros))
    
    with st.spinner(f"Vectorizing and analyzing {len(tasks)} scenarios..."):
        with ProcessPoolExecutor() as executor:
            raw_results = list(executor.map(analyze_single_scenario, tasks))
    
    st.session_state.sim_results = pd.DataFrame([r for r in raw_results if r is not None])

df_opt = st.session_state.sim_results

# --- 7. DASHBOARD TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Leaderboard", "🔬 DNA Matcher", "📉 Decomposition", "📊 Buckets"])

with tab1:
    st.subheader(f"🏆 Top Scenarios ({analysis_mode})")
    df_valid = df_opt[df_opt['Best_p'] > 0.05].sort_values('Best_p', ascending=False)
    
    if not df_valid.empty:
        st.dataframe(df_valid.head(15), width='stretch', hide_index=True)
        if st.button("Apply Absolute Best Scenario"):
            st.session_state.sel_w = int(df_valid.iloc[0]['Window'])
            st.session_state.sel_o = int(df_valid.iloc[0]['Offset'])
            st.rerun()
    else: st.warning("No significant patterns found.")
    
    st.plotly_chart(px.density_heatmap(df_opt, x="Offset", y="Window", z="Best_p", color_continuous_scale="Viridis"), width='stretch')

with tab2:
    st.subheader("🔬 DNA Surgical Matcher")
    c1, c2 = st.columns(2)
    sel_w = c1.slider("Window Size", 1, max_test_window, value=st.session_state.sel_w, key="w_v7")
    st.session_state.sel_w = sel_w
    if sel_w > 1:
        sel_o = c2.slider("Start Offset", 0, sel_w-1, value=st.session_state.sel_o if st.session_state.sel_o < sel_w else 0, key="o_v7")
        st.session_state.sel_o = sel_o
    else: sel_o = 0

    # RE-RUN SYNCED LOGIC FOR SELECTED VIEW
    final_start = data_series_daily.index.min() + pd.Timedelta(days=sel_o)
    final_series = data_series_daily[data_series_daily.index >= final_start].resample(f"{sel_w}D").sum()
    
    if "Residuals" in analysis_mode and len(final_series) > 10:
        decomp_view = seasonal_decompose(final_series, model='additive', period=4)
        dna_data = decomp_view.resid.dropna()
    else: dna_data = final_series
    
    fit_data = dna_data[dna_data > 0] if ignore_zeros else dna_data

    # Display DNA Results Table
    results_comp = []
    for name, func in [('Normal', norm), ('Lognormal', lognorm), ('Gamma', gamma)]:
        try: results_comp.append({'DNA': name, 'p_value': kstest(fit_data, name.lower() if name != 'Lognormal' else 'lognorm', args=func.fit(fit_data))[1]})
        except: pass
    try: results_comp.append({'DNA': 'Poisson', 'p_value': kstest(fit_data.astype(int), 'poisson', args=(fit_data.mean(),))[1]})
    except: pass
    
    st.table(pd.DataFrame(results_comp).sort_values('p_value', ascending=False))
    
    v1, v2 = st.columns(2)
    v1.plotly_chart(px.bar(dna_data, title="DNA Source Data"), width='stretch')
    v2.plotly_chart(px.histogram(dna_data, nbins=15, title="DNA Frequency Shape"), width='stretch')

with tab3:
    st.subheader("🔬 Interactive Decomposition Breakdown")
    if len(final_series) > 10:
        try:
            res = seasonal_decompose(final_series, model='additive', period=4)
            fig_hd = go.Figure()
            fig_hd.add_trace(go.Scatter(x=res.observed.index, y=res.observed, name="Raw Demand", line=dict(color="#CBD5E0")))
            fig_hd.add_trace(go.Scatter(x=res.trend.index, y=res.trend, name="Trend", line=dict(color="#3182CE", width=3)))
            fig_hd.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal, name="Seasonality", line=dict(color="#805AD5", dash='dot')))
            fig_hd.add_trace(go.Scatter(x=res.resid.index, y=res.resid, name="Residuals", mode='markers', marker=dict(color="#E53E3E")))
            fig_hd.update_layout(height=500, legend_orientation="h")
            st.plotly_chart(fig_hd, width='stretch')
        except: st.warning("Pattern too short.")

with tab4:
    st.subheader("📊 Raw Bucket Data")
    df_raw_bucket = final_series.reset_index()
    df_raw_bucket.columns = ['Period Start', 'Total Volume']
    st.dataframe(df_raw_bucket, width='stretch', hide_index=True)
