import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from scipy.stats import kstest, norm, lognorm, gamma, poisson
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Page Config ---
st.set_page_config(page_title="Supply Chain DNA: Surgical Decomposition", layout="wide")

st.title("🎯 Demand DNA: Surgical Decomposition")
st.markdown("""
This version allows you to **remove Trend and Seasonality** before testing the distribution. 
By testing only the **Residuals**, we find the true 'Uncertainty' of your supply chain.
""")

# --- 1. Sidebar ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Analysis Mode")
analysis_mode = st.sidebar.selectbox("What should we test?", 
                                     ["Raw Demand", "Residuals (Demand minus Trend/Seasonality)"],
                                     help="Residuals represent the 'Pure Randomness' left after removing patterns.")

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
    # Generate Sample Data with a strong Trend and Seasonality
    t = pd.date_range(start="2026-01-01", periods=365)
    trend = np.linspace(100, 500, 365)
    seasonal = 200 * np.sin(2 * np.pi * t.dayofyear / 30)
    noise = np.random.normal(0, 50, 365)
    data_series_daily = pd.Series(trend + seasonal + noise, index=t).clip(lower=0)
    st.info("💡 Using sample data with Trend and Seasonality.")

# --- 2. THE SURGICAL LOOP ---
results = []
with st.spinner("Decomposing and Testing..."):
    for w in range(1, max_test_window + 1):
        for o in range(w):
            temp_start = data_series_daily.index.min() + pd.Timedelta(days=o)
            temp_data = data_series_daily[data_series_daily.index >= temp_start]
            clubbed = temp_data.resample(f'{w}D').sum()
            
            # --- DECOMPOSITION LOGIC ---
            if "Residuals" in analysis_mode and len(clubbed) > 14:
                try:
                    # Period 4 for monthly/weekly patterns depending on window
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
tab1, tab2, tab3 = st.tabs(["🚀 Scenario Leaderboard", "🔬 DNA Matcher", "📉 Decomposition Plot"])

with tab1:
    df_leaderboard = df_opt[df_opt['p_value'] > 0.05].sort_values(by='p_value', ascending=False)
    st.subheader(f"🏆 Top Windows for {analysis_mode}")
    if not df_leaderboard.empty:
        st.dataframe(df_leaderboard.head(10), width='stretch', hide_index=True)
    else:
        st.warning("Even after decomposition, no Normal scenarios were found. Try a different DNA match.")
    
    fig_heat = px.density_heatmap(df_opt, x="Offset", y="Window", z="p_value", color_continuous_scale="Viridis")
    st.plotly_chart(fig_heat, width='stretch')

with tab2:
    best_w = int(df_leaderboard.iloc[0]['Window']) if not df_leaderboard.empty else 7
    best_o = int(df_leaderboard.iloc[0]['Offset']) if not df_leaderboard.empty else 0
    
    col1, col2 = st.columns(2)
    sel_w = col1.slider("Window Size", 1, max_test_window, best_w)
    sel_o = col2.slider("Offset", 0, sel_w-1 if sel_w > 1 else 0, best_o if best_o < sel_w else 0)

    final_series = data_series_daily.resample(f"{sel_w}D").sum()
    
    # Process the specific data for DNA Matching
    if "Residuals" in analysis_mode and len(final_series) > 14:
        decomp_final = seasonal_decompose(final_series, model='additive', period=4)
        dna_data = decomp_final.resid.dropna()
    else:
        dna_data = final_series
    
    fit_data = dna_data[dna_data > 0] if ignore_zeros else dna_data

    # DNA Test Loop
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
    st.info(f"🧬 DNA Match for {analysis_mode}: **{df_dist.iloc[0]['Distribution'].upper()}**")
    st.table(df_dist)
    
    v1, v2 = st.columns(2)
    v1.plotly_chart(px.bar(dna_data, title=f"{analysis_mode} Timeline"), width='stretch')
    v2.plotly_chart(px.histogram(fit_data, nbins=15, title="Distribution Shape"), width='stretch')

with tab3:
    st.subheader("Visual Decomposition")
    if len(final_series) > 14:
        # Show what the Trend and Seasonality actually look like
        fig_decomp = seasonal_decompose(final_series, model='additive', period=4).plot()
        st.pyplot(fig_decomp)
    else:
        st.info("Not enough data points in this window to perform a visual decomposition.")
