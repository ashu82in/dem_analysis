import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Page Config ---
st.set_page_config(page_title="Inventory DNA: Pro Edition", layout="wide")

st.title("🧬 Inventory DNA: The Retailer vs. Distributor Diagnostic")
st.markdown("""
This app automatically detects your business type. It aggregates daily transactions, 
fills missing days with 0, and performs 'Surgery' on your demand to find your true risk.
""")

# --- 1. Sidebar: Data & Logic Settings ---
st.sidebar.header("1. Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Analysis Settings")
norm_window = st.sidebar.slider("Rolling Normality Window (Days)", 7, 30, 15)
target_col = "Order_Demand"

data_series = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            xl = pd.ExcelFile(uploaded_file)
            sheet = st.sidebar.selectbox("Select Sheet", xl.sheet_names)
            df_raw = xl.parse(sheet)
        
        # Data Preparation
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
        df_raw = df_raw.dropna(subset=['Date'])
        
        # AGGREGATION: Summing orders by day
        df_daily = df_raw.groupby('Date')[target_col].sum().sort_index()
        
        # 0-FILLING: Creating a continuous timeline
        full_range = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
        data_series = df_daily.reindex(full_range, fill_value=0)
        
        st.sidebar.success(f"Analyzed {len(data_series)} consecutive days.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()
else:
    # Sample "Distributor-style" Lumpy Demand
    t = pd.date_range(start="2026-01-01", periods=150)
    # 80% zeros, 20% random spikes
    lumpy = [np.random.randint(500, 2000) if np.random.random() > 0.8 else 0 for _ in range(150)]
    data_series = pd.Series(lumpy, index=t)
    st.info("💡 No file uploaded. Showing 'Distributor-style' sample data.")

# --- 2. Automated Business Classification ---
zero_pct = (data_series == 0).sum() / len(data_series)
is_distributor = zero_pct > 0.4  # More than 40% zeros = Distributor/Wholesale

# --- 3. Surgical Decomposition ---
decomp = seasonal_decompose(data_series, model='additive', period=7, extrapolate_trend='freq')
df_dna = pd.DataFrame({
    'Actual': data_series.values,
    'Trend': decomp.trend.values,
    'Seasonal': decomp.seasonal.values,
    'Residual': decomp.resid.values
}, index=data_series.index)

# --- 4. Dashboard Tabs ---
tab1, tab2, tab3 = st.tabs(["📉 Demand Surgery", "🔬 Risk Diagnostic", "🔮 30-Day Forecast"])

with tab1:
    # Business Type Banner
    if is_distributor:
        st.warning(f"🏢 **Business Detected: DISTRIBUTOR / WHOLESALE** ({zero_pct:.0%} zero-demand days)")
    else:
        st.success(f"🛒 **Business Detected: RETAILER / EVERYDAY** ({1-zero_pct:.0%} active days)")
    
    st.subheader("Peeling the Business Layers")
    fig_layers = go.Figure()
    fig_layers.add_trace(go.Scatter(x=df_dna.index, y=df_dna['Actual'], name="1. Total Daily Demand", line=dict(color='#CBD5E0', width=1)))
    fig_layers.add_trace(go.Scatter(x=df_dna.index, y=df_dna['Trend'], name="2. Growth Trend", line=dict(color='#3182CE', width=4)))
    fig_layers.add_trace(go.Scatter(x=df_dna.index, y=df_dna['Seasonal'], name="3. Weekly Wave", line=dict(color='#F6AD55', width=2)))
    st.plotly_chart(fig_layers, use_container_width=True)

with tab2:
    st.subheader("The Shape of Risk")
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.markdown("**Surgical Residual Histogram (True Noise)**")
        fig_hist = px.histogram(df_dna, x="Residual", nbins=30, color_discrete_sequence=['#38A169'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with c_right:
        # Rolling Normality Check
        def get_rolling_p(series, window):
            p_vals = [stats.shapiro(series.iloc[i:i+window])[1] if series.iloc[i:i+window].std() > 0 else 0 
                      for i in range(len(series)-window+1)]
            return pd.Series(p_vals, index=series.index[window-1:])
        
        rolling_p = get_rolling_p(data_series, norm_window)
        fig_p = px.line(rolling_p, title=f"Rolling {norm_window}-Day Normality (Stability Check)")
        fig_p.add_hline(y=0.05, line_dash="dash", line_color="red")
        st.plotly_chart(fig_p, use_container_width=True)

    # --- Distributor-Specific Logic ---
    if is_distributor:
        non_zero = data_series[data_series > 0]
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg. Spike Size", f"{non_zero.mean():.0f} units")
        m2.metric("Order Frequency", f"{1-zero_pct:.1%}")
        m3.metric("Wait Time between Orders", f"{1/(1-zero_pct):.1f} Days")
        st.info("👉 **Distributor Tip:** Don't stock for the daily average. Stock for the **Avg Spike Size** to survive the 'Flush'.")

with tab3:
    st.subheader("30-Day Forward Forecast")
    model = ExponentialSmoothing(data_series, trend='add', seasonal='add', seasonal_periods=7).fit()
    forecast = model.forecast(30)
    
    fig_fore = go.Figure()
    fig_fore.add_trace(go.Scatter(x=data_series.index, y=data_series.values, name="History", line=dict(color='#CBD5E0')))
    fig_fore.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Projected", line=dict(color='#3182CE', width=4)))
    st.plotly_chart(fig_fore, use_container_width=True)
