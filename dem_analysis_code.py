import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import kstest, norm, lognorm, gamma, poisson
from statsmodels.tsa.seasonal import seasonal_decompose
from concurrent.futures import ThreadPoolExecutor

# --- 1. Page Configuration ---
st.set_page_config(page_title="Universal Supply Chain DNA", layout="wide")

# --- 2. Session State Management (The App's Memory) ---
if 'sel_w' not in st.session_state: st.session_state.sel_w = 7
if 'sel_o' not in st.session_state: st.session_state.sel_o = 0
if 'sim_results' not in st.session_state: st.session_state.sim_results = None
if 'last_file' not in st.session_state: st.session_state.last_file = None
if 'last_mode' not in st.session_state: st.session_state.last_mode = "Raw Demand"
if 'last_ignore_zeros' not in st.session_state: st.session_state.last_ignore_zeros = True

st.title("🎯 Universal Demand DNA & Optimizer (Stable v8.0)")
st.markdown("""
Identifying the **Best Distribution Match** across all possible windows. 
This engine tests for **Normal, Lognormal, Gamma, and Poisson** patterns using synchronized math.
""")

# --- 3. Master Sync & DNA Functions ---
@st.cache_data(show_spinner=False)
def get_bucketed_data(full_series_values, full_series_index, window, offset, mode, ignore_zeros):
    """The Single Source of Truth for data slicing. Cached for speed."""
    full_series = pd.Series(full_series_values, index=full_series_index)
    
    # Step 1: Apply Calendar Offset
    start_date = full_series.index.min() + pd.Timedelta(days=offset)
    temp_series = full_series[full_series.index >= start_date]
    
    # Step 2: Aggregate into Windows
    clubbed = temp_series.resample(f'{window}D').sum()
    
    # Step 3: Perform Seasonal Decomposition if 'Residuals' is selected
    if "Residuals" in mode and len(clubbed) > 10:
        try:
            decomp = seasonal_decompose(clubbed, model='additive', period=4)
            data = decomp.resid.dropna()
        except:
            data = clubbed
    else:
        data = clubbed
        
    return data[data > 0] if ignore_zeros else data

def run_dna_competition(data):
    """Tests all 4 DNA types and returns a ranked result set."""
    results = []
    # Test Normal
    try: results.append({'DNA': 'Normal', 'p_value': kstest(data, 'norm', args=norm.fit(data))[1]})
    except: pass
    # Test Lognormal
    try: results.append({'DNA': 'Lognormal', 'p_value': kstest(data, 'lognorm', args=lognorm.fit(data))[1]})
    except: pass
    # Test Gamma
    try: results.append({'DNA': 'Gamma', 'p_value': kstest(data, 'gamma', args=gamma.fit(data))[1]})
    except: pass
    # Test Poisson (Requires Integers)
    try:
        mu = data.mean()
        results.append({'DNA': 'Poisson', 'p_value': kstest(data.astype(int), 'poisson', args=(mu,))[1]})
    except: pass
    
    return pd.DataFrame(results).sort_values('p_value', ascending=False)

# --- 4. Sidebar Controls ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Global Strategy")
analysis_mode = st.sidebar.selectbox("Analysis Target", ["Raw Demand", "Residuals (Noise Only)"])
max_test_window = st.sidebar.slider("Max Window to Test (Days)", 7, 30, 30)
ignore_zeros = st.sidebar.toggle("Ignore Zeros (Active Demand Only)", value=True)

# SENSITIVITY CHECK: Auto-Refresh Simulation if settings change
if (st.session_state.last_mode != analysis_mode or 
    st.session_state.last_ignore_zeros != ignore_zeros):
    st.session_state.last_mode = analysis_mode
    st.session_state.last_ignore_zeros = ignore_zeros
    st.session_state.sim_results = None

if st.sidebar.button("🚀 Force Refresh Simulation"):
    st.session_state.sim_results = None

target_col = "Order_Demand"
data_series_daily = None

# --- 5. Data Loading & Cleaning ---
if uploaded_file:
    if st.session_state.last_file != uploaded_file.name:
        st.session_state.last_file = uploaded_file.name
        st.session_state.sim_results = None # Reset on new file
        st.session_state.sel_w, st.session_state.sel_o = 7, 0

    try:
        if uploaded_file.name.endswith('.csv'): df_raw = pd.read_csv(uploaded_file)
        else: df_raw = pd.read_excel(uploaded_file)
        
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
        df_raw = df_raw.dropna(subset=['Date', target_col])
        df_daily = df_raw.groupby('Date')[target_col].sum().sort_index()
        
        full_range = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
        data_series_daily = df_daily.reindex(full_range, fill_value=0)
    except Exception as e:
        st.error(f"Data Error: {e}"); st.stop()
else:
    t = pd.date_range(start="2026-01-01", periods=365)
    data_series_daily = pd.Series(np.random.lognormal(5, 0.7, 365), index=t)
    st.info("💡 Using sample data. Upload your file to start.")

# --- 6. THE STABLE THREADED SIMULATION ---
def worker(w, o):
    """Thread worker for simulation tasks."""
    test_data = get_bucketed_data(data_series_daily.values, data_series_daily.index, w, o, analysis_mode, ignore_zeros)
    if len(test_data) >= 3 and test_data.std() > 0:
        df_c = run_dna_competition(test_data)
        best = df_c.iloc[0]
        return {'Window': w, 'Offset': o, 'Best_p': best['p_value'], 'Winner_DNA': best['DNA']}
    return None

if st.session_state.sim_results is None:
    results_list = []
    with st.spinner(f"Surgically analyzing {analysis_mode} scenarios..."):
        with ThreadPoolExecutor() as executor:
            scenarios = [(w, o) for w in range(1, max_test_window + 1) for o in range(w)]
            futures = [executor.submit(worker, w, o) for w, o in scenarios]
            for f in futures:
                res = f.result()
                if res: results_list.append(res)
    st.session_state.sim_results = pd.DataFrame(results_list)

df_opt = st.session_state.sim_results

# --- 7. DASHBOARD TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Global Leaderboard", "🔬 DNA Matcher", "📉 Decomposition", "📊 Buckets"])

with tab1:
    st.subheader(f"🏆 Top Scenarios ({analysis_mode})")
    df_visible = df_opt[df_opt['Best_p'] > 0.05].sort_values(by='Best_p', ascending=False)
    
    if not df_visible.empty:
        st.dataframe(df_visible, width='stretch', hide_index=True)
        if st.button("Apply Top Recommended Setting"):
            st.session_state.sel_w = int(df_visible.iloc[0]['Window'])
            st.session_state.sel_o = int(df_visible.iloc[0]['Offset'])
            st.rerun()
    else: st.warning("No significant DNA match found.")

    st.plotly_chart(px.density_heatmap(df_opt, x="Offset", y="Window", z="Best_p", 
                                     color_continuous_scale="Viridis", title="Predictability Heatmap"), width='stretch')

with tab2:
    st.subheader("🔬 DNA Surgical Matcher")
    c1, c2 = st.columns(2)
    sel_w = c1.slider("Window Size", 1, max_test_window, value=st.session_state.sel_w, key="w_v8")
    st.session_state.sel_w = sel_w
    sel_o = c2.slider("Start Offset", 0, sel_w-1, value=st.session_state.sel_o if st.session_state.sel_o < sel_w else 0, key="o_v8") if sel_w > 1 else 0
    st.session_state.sel_o = sel_o

    # SYNCED DATA CALL
    dna_data = get_bucketed_data(data_series_daily.values, data_series_daily.index, sel_w, sel_o, analysis_mode, ignore_zeros)
    
    st.markdown("**Complete DNA Breakdown:**")
    st.table(run_dna_competition(dna_data))
    
    v1, v2 = st.columns(2)
    v1.plotly_chart(px.bar(dna_data, title=f"DNA Source Data ({sel_w}D)"), width='stretch')
    v2.plotly_chart(px.histogram(dna_data, nbins=15, title="DNA Frequency Shape"), width='stretch')

with tab3:
    st.subheader("🔬 Interactive Decomposition Breakdown")
    plot_series = data_series_daily[data_series_daily.index >= (data_series_daily.index.min() + pd.Timedelta(days=sel_o))].resample(f"{sel_w}D").sum()
    if len(plot_series) > 10:
        try:
            res = seasonal_decompose(plot_series, model='additive', period=4)
            fig_hd = go.Figure()
            fig_hd.add_trace(go.Scatter(x=res.observed.index, y=res.observed, name="1. Raw Demand", line=dict(color="#CBD5E0")))
            fig_hd.add_trace(go.Scatter(x=res.trend.index, y=res.trend, name="2. Trend (Signal)", line=dict(color="#3182CE", width=3)))
            fig_hd.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal, name="3. Seasonality (Cycle)", line=dict(color="#805AD5", dash='dot')))
            fig_hd.add_trace(go.Scatter(x=res.resid.index, y=res.resid, name="4. Residuals (The DNA)", mode='markers', marker=dict(color="#E53E3E")))
            fig_hd.update_layout(height=500, legend_orientation="h")
            st.plotly_chart(fig_hd, width='stretch')
        except: st.warning("Data volume too low for decomposition.")

with tab4:
    st.subheader("📊 Raw Bucket Data")
    df_raw_bucket = plot_series.reset_index()
    df_raw_bucket.columns = ['Period Start', 'Total Volume']
    st.dataframe(df_raw_bucket, width='stretch', hide_index=True)
