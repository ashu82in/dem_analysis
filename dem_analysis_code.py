import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Page Config ---
st.set_page_config(page_title="Demand DNA Analyzer", layout="wide")

st.title("🧬 Demand DNA: Surgical Diagnostic & Histogram Analysis")
st.markdown("""
This tool strips away the **Trend** and **Seasonality** from **Order_Demand** to reveal the **True Noise**. 
We use Histograms to visualize the "Shape of Risk" before and after the analysis.
""")

# --- 1. Sidebar: Data Input ---
st.sidebar.header("1. Upload Data")
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
        
        # Explicitly target "Order_Demand"
        target_col = "Order_Demand"
        if target_col in df_raw.columns:
            col = target_col
        else:
            col = st.sidebar.selectbox("Column not found. Select manually:", df_raw.columns)
            
        data_series = pd.to_numeric(df_raw[col], errors='coerce').dropna()
        
        if data_series.empty:
            st.error(f"The column '{col}' contains no numeric data.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    # Sample Data if no file is uploaded
    t = np.arange(100)
    data = 50 + (0.5 * t) + (15 * np.sin(2 * np.pi * t / 7)) + np.random.normal(0, 5, 100)
    data_series = pd.Series(data)
    st.info("💡 Using sample data. Upload your file to analyze 'Order_Demand'.")

# --- 2. Surgical Decomposition ---
# Using period=7 for weekly cycles
decomp = seasonal_decompose(data_series, model='additive', period=7, extrapolate_trend='freq')

df_results = pd.DataFrame({
    'Actual': data_series.values,
    'Trend': decomp.trend.values,
    'Seasonal': decomp.seasonal.values,
    'Residual': decomp.resid.values
})

# --- 3. Dashboard Tabs ---
tab1, tab2, tab3 = st.tabs(["📉 Step 1: Pattern Surgery", "📊 Step 2: Distribution & Risk", "🔮 Step 3: Forecast"])

with tab1:
    st.subheader("Peeling the Business Layers")
    fig_layers = go.Figure()
    fig_layers.add_trace(go.Scatter(y=df_results['Actual'], name="1. Raw Order_Demand", line=dict(color='#CBD5E0', width=1)))
    fig_layers.add_trace(go.Scatter(y=df_results['Trend'], name="2. Growth Trend", line=dict(color='#3182CE', width=4)))
    fig_layers.add_trace(go.Scatter(y=df_results['Seasonal'], name="3. Seasonal Wave", line=dict(color='#F6AD55', width=2)))
    
    fig_layers.update_layout(title="Order_Demand Decomposition", xaxis_title="Time Index", yaxis_title="Units")
    st.plotly_chart(fig_layers, use_container_width=True)

with tab2:
    st.subheader("The Shape of Risk: Histogram Comparison")
    st.write("Compare the 'Raw' distribution to the 'Surgical' distribution to see how much risk was actually just a predictable pattern.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**A. Raw Demand Histogram**")
        fig_hist_raw = px.histogram(df_results, x="Actual", nbins=30, 
                                   title="Distribution of Raw Orders",
                                   color_discrete_sequence=['#CBD5E0'])
        st.plotly_chart(fig_hist_raw, use_container_width=True)
        
    with col_b:
        st.markdown("**B. Residual Noise Histogram**")
        fig_hist_resid = px.histogram(df_results, x="Residual", nbins=30, 
                                     title="Distribution of Surgical Noise",
                                     color_discrete_sequence=['#38A169'])
        st.plotly_chart(fig_hist_resid, use_container_width=True)

    # Normality Test Result
    shapiro_p = stats.shapiro(df_results['Residual'].dropna())[1]
    st.divider()
    if shapiro_p > 0.05:
        st.success(f"✅ **Verdict:** The residual noise follows a **Normal Distribution** (p={shapiro_p:.3f}). Your inventory math is highly reliable.")
    else:
        st.warning(f"⚠️ **Verdict:** The noise is **Non-Normal** (p={shapiro_p:.3f}). This indicates 'Fat Tails'—expect more frequent freak spikes than a standard model predicts.")

with tab3:
    st.subheader("12-Month Momentum Forecast")
    # Simple Holt-Winters for demonstration
    model = ExponentialSmoothing(data_series, trend='add', seasonal='add', seasonal_periods=7).fit()
    forecast = model.forecast(30)
    
    fig_fore = go.Figure()
    fig_fore.add_trace(go.Scatter(y=data_series.values, name="History", line=dict(color='#CBD5E0')))
    fig_fore.add_trace(go.Scatter(x=np.arange(len(data_series), len(data_series)+30), 
                                 y=forecast, name="Forecast", line=dict(color='#3182CE', width=4)))
    st.plotly_chart(fig_fore, use_container_width=True)
