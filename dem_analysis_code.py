import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import kstest, norm, lognorm, gamma, poisson
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Page Config ---
st.set_page_config(page_title="Universal Supply Chain DNA", layout="wide")

# --- 1. Session State Initialization (The "Anti-Jump" Guard) ---
if 'sel_w' not in st.session_state: st.session_state.sel_w = 7
if 'sel_o' not in st.session_state: st.session_state.sel_o = 0
if 'sim_results' not in st.session_state: st.session_state.sim_results = None
if 'last_file_id' not in st.session_state: st.session_state.last_file_id = None

st.title("🎯 Universal Demand DNA & Optimizer")
st.markdown("""
Identifying the **Best Distribution Match** across all possible windows. 
This engine tests for **Normal, Lognormal, Gamma, and Poisson** patterns.
""")

# --- 2. Helper Functions ---
def get_best_dna(data):
    """Tests all distributions and returns the winner's p-value and name."""
    results = []
    # Test Normal
    params_n = norm.fit(data)
    results.append(('Normal', kstest(data, 'norm', args=params_n)[1]))
    # Test Lognormal
    params_l = lognorm.fit(data)
    results.append(('Lognormal', kstest(data, 'lognorm', args=params_l)[1]))
    # Test Gamma
    params_g = gamma.fit(data)
    results.append(('Gamma', kstest(data, 'gamma', args=params_g)[1]))
    # Poisson Test
    mu = data.mean()
    results.append(('Poisson', kstest(data.astype(int), 'poisson', args=(mu,))[1]))
    
    best = max(results, key=lambda x: x[1])
    return best[1], best[0]

def get_processed_data(series, mode, ignore_zeros, period=4):
    """Separates Trend/Seasonality if requested."""
    if "Residuals" in mode and len(series) > (period * 2):
        try:
            decomp = seasonal_decompose(series, model='additive', period=period)
            data = decomp.resid.dropna()
        except:
            data = series
    else:
        data = series
    return data[data > 0] if ignore_zeros else data

# --- 3. Sidebar ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# Reset Simulation if global settings change
st.sidebar.header("2. Global Strategy")
analysis_mode = st.sidebar.selectbox("Analysis Target", ["Raw Demand", "Residuals (Noise Only)"])
max_test_window = st.sidebar.slider("Max Window to Test (Days)", 7, 30, 30)
ignore_zeros = st.sidebar.toggle("Ignore Zeros (Active Demand Only)", value=True)

if st.sidebar.button("🚀 Run/Refresh Simulation"):
    st.session_state.sim_results = None

target_col = "Order_Demand"
data_series_daily = None

# --- 4. Data Loading Logic ---
if uploaded_file:
    if st.session_state.last_file_id != uploaded_file.name:
        st.session_state.last_file_id = uploaded_file.name
        st.session_state.sim_results = None # Force re-run on new file

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
    data_series_daily = pd.Series(np.random.lognormal(5, 0.8, 365), index=t)
    st.info("💡 Using sample data. Upload your file to start.")

# --- 5. THE SIMULATION ENGINE ---
if st.session_state.sim_results is None:
    results = []
    with st.spinner("Surgically scanning all windows for all DNA types..."):
        for w in range(1, max_test_window + 1):
            for o in range(w):
                temp_start = data_series_daily.index.min() + pd.Timedelta(days=o)
                clubbed = data_series_daily[data_series_daily.index >= temp_start].resample(f'{w}D').sum()
                test_data = get_processed_data(clubbed, analysis_mode, ignore_zeros)
                
                if len(test_data) >= 3 and test_data.std() > 0:
                    p_val, dna = get_best_dna(test_data)
                    results.append({'Window': w, 'Offset': o, 'Best_p': p_val, 'DNA_Match': dna})
    st.session_state.sim_results = pd.DataFrame(results)

df_opt = st.session_state.sim_results

# --- 6. TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Global Leaderboard", "🔬 DNA Matcher", "📉 Decomposition", "📊 Buckets"])

with tab1:
    st.subheader(f"🏆 Top Scenarios for {analysis_mode}")
    df_visible = df_opt[df_opt['Best_p'] > 0.05].sort_values(by='Best_p', ascending=False)
    
    if not df_visible.empty:
        st.dataframe(df_visible, width='stretch', hide_index=True)
        if st.button("Apply Top Scenario"):
            st.session_state.sel_w = int(df_visible.iloc[0]['Window'])
            st.session_state.sel_o = int(df_visible.iloc[0]['Offset'])
            st.rerun()
    else:
        st.warning("No significant DNA match found. Data is too irregular.")

    st.plotly_chart(px.density_heatmap(df_opt, x="Offset", y="Window", z="Best_p", 
                                     color_continuous_scale="Viridis", title="Predictability Map"), width='stretch')

with tab2:
    st.subheader("🔬 DNA Surgical Matcher")
    c1, c2 = st.columns(2)
    
    # Anti-Jump Slider Logic
    sel_w = c1.slider("Window Size", 1, max_test_window, value=st.session_state.sel_w, key="main_w")
    st.session_state.sel_w = sel_w

    if sel_w > 1:
        safe_o = st.session_state.sel_o if st.session_state.sel_o < sel_w else 0
        sel_o = c2.slider("Start Offset", 0, sel_w-1, value=safe_o, key="main_o")
        st.session_state.sel_o = sel_o
    else:
        sel_o = 0

    # Execute selection
    current_series = data_series_daily[data_series_daily.index >= (data_series_daily.index.min() + pd.Timedelta(days=sel_o))].resample(f"{sel_w}D").sum()
    dna_data = get_processed_data(current_series, analysis_mode, ignore_zeros)

    # Detailed DNA Comparison Table
    dist_results = []
    for d_name, d_func in [('Normal', norm), ('Lognormal', lognorm), ('Gamma', gamma)]:
        try:
            p = kstest(dna_data, d_name.lower() if d_name != 'Lognormal' else 'lognorm', args=d_func.fit(dna_data))[1]
            dist_results.append({'Distribution': d_name, 'p-value': p})
        except: continue
    
    st.table(pd.DataFrame(dist_results).sort_values('p-value', ascending=False))
    
    v1, v2 = st.columns(2)
    v1.plotly_chart(px.bar(dna_data, title="DNA Source Data"), width='stretch')
    v2.plotly_chart(px.histogram(dna_data, nbins=15, title="DNA Shape"), width='stretch')

with tab3:
    st.subheader("🔬 Interactive Decomposition")
    if len(current_series) > 10:
        try:
            res = seasonal_decompose(current_series, model='additive', period=4)
            fig_hd = go.Figure()
            fig_hd.add_trace(go.Scatter(x=res.observed.index, y=res.observed, name="Raw Demand", line=dict(color="#CBD5E0")))
            fig_hd.add_trace(go.Scatter(x=res.trend.index, y=res.trend, name="Trend", line=dict(color="#3182CE", width=3)))
            fig_hd.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal, name="Seasonality", line=dict(color="#805AD5", dash='dot')))
            fig_hd.add_trace(go.Scatter(x=res.resid.index, y=res.resid, name="Residuals", mode='markers', marker=dict(color="#E53E3E")))
            fig_hd.update_layout(height=500, legend_orientation="h")
            st.plotly_chart(fig_hd, width='stretch')
        except: st.warning("Not enough patterns for decomposition.")

with tab4:
    st.subheader("📊 Raw Bucket Data")
    df_buckets = current_series.reset_index()
    df_buckets.columns = ['Period Start', 'Volume']
    st.dataframe(df_buckets, width='stretch', hide_index=True)
