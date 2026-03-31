import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Page Config ---
st.set_page_config(page_title="Demand DNA & Forecast", layout="wide")

st.title("🧬 Demand DNA: Surgical Diagnostic & Forecast")
st.markdown("""
We strip away **Trend** and **Seasonality** from **Order Demand** to reveal the **True Noise**. 
This version includes a distribution analysis to visualize your "Shape of Risk."
""")

# --- 1. Sidebar: Data Input ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

data_series = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            xl = pd.ExcelFile(uploaded_file)
            sheet = st.sidebar.selectbox("Select Sheet", xl.sheet_names)
            df_raw = xl.parse(sheet)
        
        # Auto-detect "Order Demand"
        default_col = "Order Demand" if "Order Demand" in df_raw.columns else df_raw.columns[0]
        col = st.sidebar.selectbox("Select Column", df_raw.columns, index=list(df_raw.columns).index(default_col))
        data_series = pd.to_numeric(df_raw[col], errors='coerce').dropna()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    # Sample Data
    t = np.arange(120)
    data = 50 + (0.5 * t) + (15 * np.sin(2 * np.pi * t / 7)) + np.random.normal(0, 5, 120)
    data_series = pd.Series(data)
    st.info("💡 Using sample data. Upload your file to analyze 'Order Demand'.")

# --- 2. Surgical Decomposition ---
decomp = seasonal_decompose(data_series, model='additive', period=7, extrapolate_trend='freq')
df = pd.DataFrame({
    'Actual': data_series.values,
    'Trend': decomp.trend.values,
    'Seasonal': decomp.seasonal.values,
    'Residual': decomp.resid.values
})

# --- 3. Tabs ---
tab1, tab2, tab3 = st.tabs(["📉 Step 1: Trend & Waves", "📊 Step 2: Distribution (Histograms)", "🔮 Step 3: 12-Month Forecast"])

with tab1:
    st.subheader("The Business Signal")
    fig_layers = go.Figure()
    fig_layers.add_trace(go.Scatter(y=df['Actual'], name="Actual Demand", line=dict(color='#CBD5E0', width=1)))
    fig_layers.add_trace(go.Scatter(y=df['Trend'], name="Growth Trend", line=dict(color='#3182CE', width=4)))
    fig_layers.add_trace(go.Scatter(y=df['Seasonal'], name="Weekly Wave", line=dict(color='#F6AD55', width=2)))
    st.plotly_chart(fig_layers, use_container_width=True)

with tab2:
    st.subheader("The Shape of Risk (Histograms)")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.write("**A. Raw Demand Distribution**")
        st.write("This shows how unpredictable your orders look *before* the analysis.")
        fig_hist_raw = px.histogram(df, x="Actual", nbins=30, color_discrete_sequence=['#CBD5E0'],
                                   title="Raw Order Demand")
        st.plotly_chart(fig_hist_raw, use_container_width=True)
        
    with col_right:
        st.write("**B. Residual 'Noise' Distribution**")
        st.write("This shows the *true* random risk after removing patterns.")
        fig_hist_resid = px.histogram(df, x="Residual", nbins=30, color_discrete_sequence=['#38A169'],
                                     title="Surgical Residuals (The Noise)")
        st.plotly_chart(fig_hist_resid, use_container_width=True)

    # Normality Verdict
    shapiro_p = stats.shapiro(df['Residual'].dropna())[1]
    if shapiro_p > 0.05:
        st.success(f"✅ **Verdict:** The Noise is Normal (p={shapiro_p:.3f}). Your inventory risk is predictable.")
    else:
        st.warning(f"⚠️ **Verdict:** The Noise is Non-Normal (p={shapiro_p:.3f}). Watch for freak spikes!")

with tab3:
    st.subheader("12-Month Projected Growth")
    # Monthly aggregation for a stable forecast
    df_monthly = data_series.copy()
    # Assuming daily data for resample; if not date-indexed, we use the raw series
    model = ExponentialSmoothing(data_series, trend='add', seasonal='add', seasonal_periods=7).fit()
    forecast = model.forecast(30) # 30-day forecast for daily data
    
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(y=data_series.values, name="History", line=dict(color='#CBD5E0')))
    fig_forecast.add_trace(go.Scatter(x=np.arange(len(data_series), len(data_series)+30), 
                                     y=forecast, name="Forecast", line=dict(color='#3182CE', width=4)))
    st.plotly_chart(fig_forecast, use_container_width=True)
