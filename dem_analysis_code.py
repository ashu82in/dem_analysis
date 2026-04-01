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

# --- 1. Session State Initialization (The Anti-Jump Guard) ---
if 'sel_w' not in st.session_state: st.session_state.sel_w = 7
if 'sel_o' not in st.session_state: st.session_state.sel_o = 0
if 'sim_results' not in st.session_state: st.session_state.sim_results = None
if 'last_file' not in st.session_state: st.session_state.last_file = None

st.title("🎯 Universal Demand DNA & Optimizer (v5.0)")

# --- 2. THE SURGICAL SYNC ENGINE (Master Function) ---
def get_bucketed_data(full_series, window, offset, mode, ignore_zeros):
    """Ensures Leaderboard and DNA Matcher see the exact same slice of data."""
    # Step 1: Apply Offset
    start_date = full_series.index.min() + pd.Timedelta(days=offset)
    temp_series = full_series[full_series.index >= start_date]
    
    # Step 2: Resample
    clubbed = temp_series.resample(f'{window}D').sum()
    
    # Step 3: Decompose if requested
    if "Residuals" in mode and len(clubbed) > 10:
        try:
            decomp = seasonal_decompose(clubbed, model='additive', period=4)
            data = decomp.resid.dropna()
        except:
            data = clubbed
    else:
        data = clubbed
        
    # Step 4: Filter Zeros
    return data[data > 0] if ignore_zeros else data

def run_dna_competition(data):
    """Tests all 4 DNA types and returns a ranked dataframe."""
    results = []
    # Normal
    try: results.append({'DNA': 'Normal', 'p_value': kstest(data, 'norm', args=norm.fit(data))[1]})
    except: pass
    # Lognormal
    try: results.append({'DNA': 'Lognormal', 'p_value': kstest(data, 'lognorm', args=lognorm.fit(data))[1]})
    except: pass
    # Gamma
    try: results.append({'DNA': 'Gamma', 'p_value': kstest(data, 'gamma', args=gamma.fit(data))[1]})
    except: pass
    # Poisson
    try:
        mu = data.mean()
        results.append({'DNA': 'Poisson', 'p_value': kstest(data.astype(int), 'poisson', args=(mu,))[1]})
    except: pass
    
    return pd.DataFrame(results).sort_values('p_value', ascending=False)

# --- 3. Sidebar & Data Loading ---
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

if uploaded_file:
    # Reset state for a new file
    if st.session_state.last_file != uploaded_file.name:
        st.session_state.last_file = uploaded_file.name
        st.session_state.sim_results = None
        st.session_state.sel_w = 7
        st.session_state.sel_o = 0

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
    # Sample Data Generator
    t = pd.date_range(start="2026-01-01", periods=365)
    data_series_daily = pd.Series(np.random.lognormal(5, 0.8, 365), index=t)
    st.info("💡 Using sample data. Upload your file to start.")

# --- 4. THE LOCKED SIMULATION ---
if st.session_state.sim_results is None:
    results_list = []
    with st.spinner("Surgically scanning all windows for all DNA types..."):
        for w in range(1, max_test_window + 1):
            for o in range(w):
                test_data = get_bucketed_data(data_series_daily, w, o, analysis_mode, ignore_zeros)
                if len(test_data) >= 3 and test_data.std() > 0:
                    df_comp = run_dna_competition(test_data)
                    best = df_comp.iloc[0]
                    results_list.append({'Window': w, 'Offset': o, 'Best_p': best['p_value'], 'Winner': best['DNA']})
    st.session_state.sim_results = pd.DataFrame(results_list)

df_opt = st.session_state.sim_results

# --- 5. TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Global Leaderboard", "🔬 DNA Matcher", "📉 Decomposition", "📊 Buckets"])

with tab1:
    st.subheader(f"🏆 Top Scenarios ({analysis_mode})")
    df_visible = df_opt[df_opt['Best_p'] > 0.05].sort_values(by='Best_p', ascending=False)
    
    if not df_visible.empty:
        st.dataframe(df_visible.head(15), width='stretch', hide_index=True)
        if st.button("Apply Absolute Best Scenario"):
            st.session_state.sel_w = int(df_visible.iloc[0]['Window'])
            st.session_state.sel_o = int(df_visible.iloc[0]['Offset'])
            st.rerun()
    else:
        st.warning("No predictable DNA found in any window. Data is too erratic.")

    st.plotly_chart(px.density_heatmap(df_opt, x="Offset", y="Window", z="Best_p", 
                                     color_continuous_scale="Viridis", title="Predictability Map"), width='stretch')

with tab2:
    st.subheader("🔬 DNA Surgical Matcher")
    c1, c2 = st.columns(2)
    
    # Anti-Jump Slider Logic
    sel_w = c1.slider("Window Size", 1, max_test_window, value=st.session_state.sel_w, key="w_slider_v5")
    st.session_state.sel_w = sel_w

    if sel_w > 1:
        safe_o = st.session_state.sel_o if st.session_state.sel_o < sel_w else 0
        sel_o = c2.slider("Start Offset", 0, sel_w-1, value=safe_o, key="o_slider_v5")
        st.session_state.sel_o = sel_o
    else:
        sel_o = 0

    # THE CRITICAL SYNC CALL
    dna_data = get_bucketed_data(data_series_daily, sel_w, sel_o, analysis_mode, ignore_zeros)
    
    # Full DNA Competition Result
    st.markdown("**All Distribution p-values for this Window:**")
    df_full_comp = run_dna_competition(dna_data)
    st.table(df_full_comp)
    
    v1, v2 = st.columns(2)
    v1.plotly_chart(px.bar(dna_data, title=f"DNA Source Data ({sel_w}D)"), width='stretch')
    v2.plotly_chart(px.histogram(dna_data, nbins=15, title="DNA Frequency Shape"), width='stretch')

with tab3:
    st.subheader("🔬 Interactive Decomposition Breakdown")
    # Regenerate raw aggregation for plot (without dropping outliers/zeros for continuity)
    plot_start = data_series_daily.index.min() + pd.Timedelta(days=sel_o)
    plot_series = data_series_daily[data_series_daily.index >= plot_start].resample(f"{sel_w}D").sum()

    if len(plot_series) > 10:
        try:
            res = seasonal_decompose(plot_series, model='additive', period=4)
            fig_hd = go.Figure()
            fig_hd.add_trace(go.Scatter(x=res.observed.index, y=res.observed, name="Raw Demand", line=dict(color="#CBD5E0")))
            fig_hd.add_trace(go.Scatter(x=res.trend.index, y=res.trend, name="Trend", line=dict(color="#3182CE", width=3)))
            fig_hd.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal, name="Seasonality", line=dict(color="#805AD5", dash='dot')))
            fig_hd.add_trace(go.Scatter(x=res.resid.index, y=res.resid, name="Residuals (The DNA)", mode='markers', marker=dict(color="#E53E3E")))
            fig_hd.update_layout(height=500, legend_orientation="h", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_hd, width='stretch')
        except: st.warning("Not enough patterns for interactive decomposition.")

with tab4:
    st.subheader("📊 Raw Bucket Data")
    df_buckets = plot_series.reset_index()
    df_buckets.columns = ['Period Start', 'Total Volume']
    st.dataframe(df_buckets, width='stretch', hide_index=True)
