import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import kstest, norm, lognorm, gamma, poisson
from statsmodels.tsa.seasonal import seasonal_decompose

# --- 1. Page Configuration ---
st.set_page_config(page_title="Universal Supply Chain DNA", layout="wide")

# --- 2. Session State Management (The "Brain" of the App) ---
# These ensure sliders don't jump and the simulation only runs when needed.
if 'sel_w' not in st.session_state: st.session_state.sel_w = 7
if 'sel_o' not in st.session_state: st.session_state.sel_o = 0
if 'sim_results' not in st.session_state: st.session_state.sim_results = None
if 'last_file' not in st.session_state: st.session_state.last_file = None
if 'last_mode' not in st.session_state: st.session_state.last_mode = "Raw Demand"
if 'last_ignore_zeros' not in st.session_state: st.session_state.last_ignore_zeros = True

st.title("🎯 Universal Demand DNA & Optimizer")
st.markdown("""
This engine identifies the **Golden Window** by testing Trend, Seasonality, and Residual DNA. 
It synchronizes the Leaderboard with the DNA Matcher to ensure "Pixel-Perfect" accuracy.
""")

# --- 3. Master Sync Functions ---
def get_bucketed_data(full_series, window, offset, mode, ignore_zeros):
    """The Single Source of Truth for data slicing."""
    # Apply Calendar Offset
    start_date = full_series.index.min() + pd.Timedelta(days=offset)
    temp_series = full_series[full_series.index >= start_date]
    
    # Aggregate into Windows
    clubbed = temp_series.resample(f'{window}D').sum()
    
    # Perform Seasonal Decomposition if 'Residuals' is selected
    if "Residuals" in mode and len(clubbed) > 10:
        try:
            # We use period=4 to detect monthly/cycle patterns in the buckets
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
    try:
        p_norm = kstest(data, 'norm', args=norm.fit(data))[1]
        results.append({'DNA': 'Normal', 'p_value': p_norm})
    except: pass
    
    # Test Lognormal
    try:
        p_log = kstest(data, 'lognorm', args=lognorm.fit(data))[1]
        results.append({'DNA': 'Lognormal', 'p_value': p_log})
    except: pass
    
    # Test Gamma
    try:
        p_gam = kstest(data, 'gamma', args=gamma.fit(data))[1]
        results.append({'DNA': 'Gamma', 'p_value': p_gam})
    except: pass
    
    # Test Poisson (Requires Integers)
    try:
        mu = data.mean()
        p_poi = kstest(data.astype(int), 'poisson', args=(mu,))[1]
        results.append({'DNA': 'Poisson', 'p_value': p_poi})
    except: pass
    
    return pd.DataFrame(results).sort_values('p_value', ascending=False)

# --- 4. Sidebar Controls ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Global Strategy")
analysis_mode = st.sidebar.selectbox("Analysis Target", ["Raw Demand", "Residuals (Noise Only)"])
max_test_window = st.sidebar.slider("Max Window to Test (Days)", 7, 30, 30)
ignore_zeros = st.sidebar.toggle("Ignore Zeros (Active Demand Only)", value=True)

# SENSITIVITY CHECK: If these change, we MUST re-run the simulation
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
    # Reset on new file upload
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
        
        # Fill missing dates with 0 to keep the time-series frequency intact
        full_range = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
        data_series_daily = df_daily.reindex(full_range, fill_value=0)
    except Exception as e:
        st.error(f"Data Error: {e}")
        st.stop()
else:
    # High-quality sample data
    t = pd.date_range(start="2026-01-01", periods=365)
    data_series_daily = pd.Series(np.random.lognormal(5, 0.7, 365), index=t)
    st.info("💡 Using sample data. Upload your file to see the surgical breakdown.")

# --- 6. THE SIMULATION ENGINE ---
if st.session_state.sim_results is None:
    results_list = []
    with st.spinner(f"Surgically analyzing all {analysis_mode} scenarios..."):
        for w in range(1, max_test_window + 1):
            for o in range(w):
                # Use the Master Sync function
                test_data = get_bucketed_data(data_series_daily, w, o, analysis_mode, ignore_zeros)
                
                if len(test_data) >= 3 and test_data.std() > 0:
                    df_comp = run_dna_competition(test_data)
                    best = df_comp.iloc[0]
                    results_list.append({
                        'Window': w, 
                        'Offset': o, 
                        'Best_p': best['p_value'], 
                        'Winner_DNA': best['DNA']
                    })
    st.session_state.sim_results = pd.DataFrame(results_list)

df_opt = st.session_state.sim_results

# --- 7. DASHBOARD TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Global Leaderboard", "🔬 DNA Matcher", "📉 Decomposition", "📊 Raw Buckets"])

with tab1:
    st.subheader(f"🏆 Top Predictable Scenarios ({analysis_mode})")
    # Show all scenarios that pass the statistical significance bar
    df_visible = df_opt[df_opt['Best_p'] > 0.05].sort_values(by='Best_p', ascending=False)
    
    if not df_visible.empty:
        st.dataframe(df_visible, width='stretch', hide_index=True)
        if st.button("Apply Top Recommended Setting"):
            st.session_state.sel_w = int(df_visible.iloc[0]['Window'])
            st.session_state.sel_o = int(df_visible.iloc[0]['Offset'])
            st.rerun()
    else:
        st.warning("No significant DNA match found in any window. Data is currently chaotic.")

    # Heatmap for visual trend spotting
    fig_heat = px.density_heatmap(df_opt, x="Offset", y="Window", z="Best_p", 
                                 color_continuous_scale="Viridis", title="Predictability Heatmap")
    st.plotly_chart(fig_heat, width='stretch')

with tab2:
    st.subheader("🔬 DNA Surgical Matcher")
    c1, c2 = st.columns(2)
    
    # Sliders linked to Session State
    sel_w = c1.slider("Select Window Size", 1, max_test_window, value=st.session_state.sel_w, key="w_v6_key")
    st.session_state.sel_w = sel_w

    if sel_w > 1:
        # Prevent Offset from exceeding current Window Size
        safe_o = st.session_state.sel_o if st.session_state.sel_o < sel_w else 0
        sel_o = c2.slider("Select Start Offset", 0, sel_w-1, value=safe_o, key="o_v6_key")
        st.session_state.sel_o = sel_o
    else:
        sel_o = 0

    # THE CRITICAL SYNC CALL (Matches the Loop exactly)
    dna_data = get_bucketed_data(data_series_daily, sel_w, sel_o, analysis_mode, ignore_zeros)
    
    # Detailed Competition Table
    st.markdown("**Complete Statistical Breakdown:**")
    df_full_comp = run_dna_competition(dna_data)
    st.table(df_full_comp)
    
    # Visual Confirmation
    v1, v2 = st.columns(2)
    v1.plotly_chart(px.bar(dna_data, title=f"Bucket Timeline ({sel_w}D Windows)"), width='stretch')
    v2.plotly_chart(px.histogram(dna_data, nbins=15, title="DNA Frequency Distribution"), width='stretch')

with tab3:
    st.subheader("🔬 Interactive Decomposition Breakdown")
    # Regenerate raw buckets for visual trend mapping
    start_p = data_series_daily.index.min() + pd.Timedelta(days=sel_o)
    plot_series = data_series_daily[data_series_daily.index >= start_p].resample(f"{sel_w}D").sum()

    if len(plot_series) > 10:
        try:
            res = seasonal_decompose(plot_series, model='additive', period=4)
            fig_hd = go.Figure()
            # Layers
            fig_hd.add_trace(go.Scatter(x=res.observed.index, y=res.observed, name="1. Raw Demand", line=dict(color="#CBD5E0")))
            fig_hd.add_trace(go.Scatter(x=res.trend.index, y=res.trend, name="2. Trend (Signal)", line=dict(color="#3182CE", width=3)))
            fig_hd.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal, name="3. Seasonality (Cycle)", line=dict(color="#805AD5", dash='dot')))
            fig_hd.add_trace(go.Scatter(x=res.resid.index, y=res.resid, name="4. Residuals (The DNA)", mode='markers', marker=dict(color="#E53E3E", size=8)))
            
            fig_hd.update_layout(height=500, legend_orientation="h", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_hd, width='stretch')
        except:
            st.warning("Data volume too low for decomposition. Try a smaller window or more history.")
    else:
        st.info("Decomposition requires at least 10 data points.")

with tab4:
    st.subheader("📊 Raw Bucket Data")
    df_raw_table = plot_series.reset_index()
    df_raw_table.columns = ['Bucket Start Date', 'Aggregated Demand']
    st.dataframe(df_raw_table, width='stretch', hide_index=True)
