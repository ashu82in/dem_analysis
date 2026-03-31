import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import kstest, norm, lognorm, gamma, poisson
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Page Config ---
st.set_page_config(page_title="Supply Chain DNA: Surgical decomposition", layout="wide")

st.title("🎯 Demand DNA: Surgical Decomposition Engine")
st.markdown("""
This tool separates **Trend** and **Seasonality** from your data to test the **Residuals** (the true random noise). 
This often results in much higher p-values and a better 'Golden Window' match.
""")

# --- 1. Sidebar ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Analysis Mode")
analysis_mode = st.sidebar.selectbox("Analysis Target", 
                                     ["Raw Demand", "Residuals (Noise Only)"],
                                     help="Residuals = Raw Demand - (Trend + Seasonality)")

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
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    t = pd.date_range(start="2026-01-01", periods=365)
    trend = np.linspace(50, 300, 365)
    seasonal = 150 * np.sin(2 * np.pi * t.dayofyear / 30)
    noise = np.random.lognormal(mean=4, sigma=0.8, size=365)
    data_series_daily = pd.Series(trend + seasonal + noise, index=t).clip(lower=0)
    st.info("💡 Using sample data. Upload your file to begin.")

# --- 2. OPTIMIZATION LOOP ---
results = []
with st.spinner("Surgically analyzing scenarios..."):
    for w in range(1, max_test_window + 1):
        for o in range(w):
            temp_start = data_series_daily.index.min() + pd.Timedelta(days=o)
            temp_data = data_series_daily[data_series_daily.index >= temp_start]
            clubbed = temp_data.resample(f'{w}D').sum()
            
            # --- Conditional Decomposition ---
            if "Residuals" in analysis_mode and len(clubbed) > 10:
                try:
                    decomp = seasonal_decompose(clubbed, model='additive', period=4)
                    processed_data = decomp.resid.dropna()
                except:
                    processed_data = clubbed
            else:
                processed_data = clubbed

            test_data = processed_data[processed_data > 0] if ignore_zeros else processed_data
            
            if len(test_data) >= 3 and test_data.std() > 0:
                params = norm.fit(test_data)
                _, p = kstest(test_data, 'norm', args=params)
                results.append({'Window': w, 'Offset': o, 'p_value': p})

df_opt = pd.DataFrame(results)

# --- 3. TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Leaderboard", "🔬 DNA Matcher", "📉 Surgical Decomposition", "📊 Raw Buckets"])

with tab1:
    df_leaderboard = df_opt[df_opt['p_value'] > 0.05].sort_values(by='p_value', ascending=False)
    st.subheader(f"Top Predictable Windows for {analysis_mode}")
    if not df_leaderboard.empty:
        st.dataframe(df_leaderboard.head(10), width='stretch', hide_index=True)
    else:
        st.warning("No Normal scenarios found. Try checking 'Residuals' mode in the sidebar.")
    
    fig_heat = px.density_heatmap(df_opt, x="Offset", y="Window", z="p_value", color_continuous_scale="Viridis")
    st.plotly_chart(fig_heat, width='stretch')

with tab2:
    best_w = int(df_leaderboard.iloc[0]['Window']) if not df_leaderboard.empty else 7
    best_o = int(df_leaderboard.iloc[0]['Offset']) if not df_leaderboard.empty else 0
    
    c1, c2 = st.columns(2)
    sel_w = c1.slider("Window Size (Days)", 1, max_test_window, best_w)
    
    # --- RECTIFIED OFFSET LOGIC (No more Range 0,0 Error) ---
    if sel_w > 1:
        sel_o = c2.slider("Start Offset (Days)", 0, sel_w-1, best_o if best_o < sel_w else 0)
    else:
        c2.info("Offset fixed at 0 for 1-day window.")
        sel_o = 0

    final_series = data_series_daily.resample(f"{sel_w}D").sum()
    
    if "Residuals" in analysis_mode and len(final_series) > 10:
        try:
            decomp_final = seasonal_decompose(final_series, model='additive', period=4)
            dna_data = decomp_final.resid.dropna()
        except: dna_data = final_series
    else:
        dna_data = final_series
    
    fit_data = dna_data[dna_data > 0] if ignore_zeros else dna_data

    # DNA Testing Loop
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
    st.info(f"🧬 Winner: **{df_dist.iloc[0]['Distribution'].upper()}**")
    st.table(df_dist)
    
    v1, v2 = st.columns(2)
    v1.plotly_chart(px.bar(dna_data, title=f"{analysis_mode} View"), width='stretch')
    v2.plotly_chart(px.histogram(fit_data, nbins=15, title="Distribution Shape"), width='stretch')

with tab3:
    st.subheader("🔬 Interactive Decomposition Breakdown")
    if len(final_series) > 10:
        try:
            res = seasonal_decompose(final_series, model='additive', period=4)
            fig_clean = go.Figure()
            fig_clean.add_trace(go.Scatter(x=res.observed.index, y=res.observed, name="1. Raw Demand", line=dict(color="#CBD5E0")))
            fig_clean.add_trace(go.Scatter(x=res.trend.index, y=res.trend, name="2. Trend (Signal)", line=dict(color="#3182CE", width=3)))
            fig_clean.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal, name="3. Seasonality (Rhythm)", line=dict(color="#805AD5", dash='dot')))
            fig_clean.add_trace(go.Scatter(x=res.resid.index, y=res.resid, name="4. Residuals (DNA Source)", mode='markers', marker=dict(color="#E53E3E")))
            fig_clean.update_layout(height=500, legend_orientation="h", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_clean, width='stretch')
        except: st.warning("Pattern too short to decompose.")
    else: st.info("Increase window size to see decomposition components.")

with tab4:
    st.subheader("📊 Raw Bucket Window Data")
    df_raw_view = final_series.reset_index()
    df_raw_view.columns = ['Start Date', 'Volume']
    st.write(f"Showing demand aggregated into {sel_w}-day windows.")
    st.dataframe(df_raw_view, width='stretch', hide_index=True)
