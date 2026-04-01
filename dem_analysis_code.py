import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import kstest, norm, lognorm, gamma, poisson
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Page Config ---
st.set_page_config(page_title="Supply Chain DNA: Surgical Master", layout="wide")

# --- 1. Session State Initialization (Fixes the Jumping Slider) ---
if 'sel_w' not in st.session_state:
    st.session_state.sel_w = 7
if 'sel_o' not in st.session_state:
    st.session_state.sel_o = 0
if 'last_file' not in st.session_state:
    st.session_state.last_file = None

st.title("🎯 Demand DNA: Surgical Master Engine")
st.markdown("""
This engine identifies your **Golden Window** by testing Trend, Seasonality, and Residual DNA.
""")

# --- 2. Helper Function: Synchronized Processing ---
def get_processed_data(series, mode, ignore_zeros, period=4):
    """Ensures p-values match by using identical trimming/filtering logic."""
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

# Reset sliders only if a NEW file is uploaded
if uploaded_file:
    if st.session_state.last_file != uploaded_file.name:
        st.session_state.last_file = uploaded_file.name
        st.session_state.sel_w = 7
        st.session_state.sel_o = 0

st.sidebar.header("2. Analysis Mode")
analysis_mode = st.sidebar.selectbox("Analysis Target", 
                                     ["Raw Demand", "Residuals (Noise Only)"],
                                     help="Residuals = Demand minus Trend and Seasonality")

st.sidebar.header("3. Optimization Settings")
max_test_window = st.sidebar.slider("Max Window to Test (Days)", 7, 30, 30)
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
    # Default Sample Data
    t = pd.date_range(start="2026-01-01", periods=365)
    trend = np.linspace(50, 250, 365)
    seasonal = 100 * np.sin(2 * np.pi * t.dayofyear / 30)
    noise = np.random.normal(0, 40, 365)
    data_series_daily = pd.Series(trend + seasonal + noise, index=t).clip(lower=0)
    st.info("💡 Using sample data. Upload your file to begin.")

# --- 4. Optimization Loop ---
results = []
with st.spinner("Surgically calculating all scenarios..."):
    for w in range(1, max_test_window + 1):
        for o in range(w):
            temp_start = data_series_daily.index.min() + pd.Timedelta(days=o)
            temp_data = data_series_daily[data_series_daily.index >= temp_start]
            clubbed = temp_data.resample(f'{w}D').sum()
            
            test_data = get_processed_data(clubbed, analysis_mode, ignore_zeros)
            
            if len(test_data) >= 3 and test_data.std() > 0:
                params = norm.fit(test_data)
                _, p = kstest(test_data, 'norm', args=params)
                results.append({'Window': w, 'Offset': o, 'p_value': p})

df_opt = pd.DataFrame(results)

# --- 5. Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Leaderboard", "🔬 DNA Matcher", "📉 Decomposition", "📊 Buckets"])

with tab1:
    df_leaderboard = df_opt[df_opt['p_value'] > 0.05].sort_values(by='p_value', ascending=False)
    st.subheader(f"Top Windows for {analysis_mode}")
    
    if not df_leaderboard.empty:
        st.dataframe(df_leaderboard.head(10), width='stretch', hide_index=True)
        if st.button("Apply Top Window"):
            st.session_state.sel_w = int(df_leaderboard.iloc[0]['Window'])
            st.session_state.sel_o = int(df_leaderboard.iloc[0]['Offset'])
            st.rerun()
    else:
        st.warning("No Normal scenarios found.")
    
    st.plotly_chart(px.density_heatmap(df_opt, x="Offset", y="Window", z="p_value", color_continuous_scale="Viridis"), width='stretch')

with tab2:
    st.subheader("🔬 DNA Matcher")
    c1, c2 = st.columns(2)
    
    # SLIDER LOGIC WITH STATE GUARD
    sel_w = c1.slider("Window Size", 1, max_test_window, value=st.session_state.sel_w, key="w_slider_key")
    st.session_state.sel_w = sel_w

    if sel_w > 1:
        # Check if the saved offset is still valid for this window size
        safe_o = st.session_state.sel_o if st.session_state.sel_o < sel_w else 0
        sel_o = c2.slider("Start Offset", 0, sel_w-1, value=safe_o, key="o_slider_key")
        st.session_state.sel_o = sel_o
    else:
        sel_o = 0

    # Data Aggregation
    final_start = data_series_daily.index.min() + pd.Timedelta(days=sel_o)
    final_series = data_series_daily[data_series_daily.index >= final_start].resample(f"{sel_w}D").sum()
    
    # Shared Processing Function
    dna_data = get_processed_data(final_series, analysis_mode, ignore_zeros)
    
    # DNA Test Competition
    dist_names = ["norm", "lognorm", "gamma", "poisson"]
    dist_results = []
    for name in dist_names:
        try:
            if name == "poisson":
                mu = dna_data.mean()
                _, p = kstest(dna_data.astype(int), 'poisson', args=(mu,))
                dist_results.append({"Distribution": "Poisson", "p-value": p})
            else:
                dist = getattr(stats, name)
                params = dist.fit(dna_data)
                _, p = kstest(dna_data, name, args=params)
                dist_results.append({"Distribution": name.capitalize(), "p-value": p})
        except: continue

    df_dist = pd.DataFrame(dist_results).sort_values(by="p-value", ascending=False)
    st.info(f"🧬 DNA Match: **{df_dist.iloc[0]['Distribution'].upper()}** (p={df_dist.iloc[0]['p-value']:.4f})")
    st.table(df_dist)
    
    v1, v2 = st.columns(2)
    v1.plotly_chart(px.bar(dna_data, title=f"DNA Source Timeline ({analysis_mode})"), width='stretch')
    v2.plotly_chart(px.histogram(dna_data, nbins=15, title="DNA Distribution Shape"), width='stretch')

with tab3:
    st.subheader("🔬 High-Definition Decomposition")
    if len(final_series) > 10:
        try:
            res = seasonal_decompose(final_series, model='additive', period=4)
            fig_clean = go.Figure()
            fig_clean.add_trace(go.Scatter(x=res.observed.index, y=res.observed, name="Raw Demand", line=dict(color="#CBD5E0")))
            fig_clean.add_trace(go.Scatter(x=res.trend.index, y=res.trend, name="Trend", line=dict(color="#3182CE", width=3)))
            fig_clean.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal, name="Seasonality", line=dict(color="#805AD5", dash='dot')))
            fig_clean.add_trace(go.Scatter(x=res.resid.index, y=res.resid, name="Residuals (The DNA)", mode='markers', marker=dict(color="#E53E3E")))
            fig_clean.update_layout(height=500, legend_orientation="h")
            st.plotly_chart(fig_clean, width='stretch')
        except: st.warning("Pattern too short for decomposition visualization.")

with tab4:
    st.subheader("📄 Raw Bucket Data Table")
    df_raw_view = final_series.reset_index()
    df_raw_view.columns = ['Period Start Date', 'Total Volume']
    st.dataframe(df_raw_view, width='stretch', hide_index=True)
