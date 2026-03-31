import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Page Config ---
st.set_page_config(page_title="Demand DNA Analyzer", layout="wide")

st.title("🧬 Demand DNA: The Surgical Diagnostic")
st.markdown("""
Standard math checks the **Raw Data** for risk. **We check the 'Noise'.** By removing Trend and Seasonality first, we reveal your *True Unpredictability*.
""")

# --- Sidebar: Data Input ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    col = st.sidebar.selectbox("Select Demand Column", df_raw.columns)
    data_series = pd.to_numeric(df_raw[col], errors='coerce').dropna()
else:
    # Create a "Surgical" Sample: Trend + Season + Normal Noise
    t = np.arange(100)
    data = 50 + (0.5 * t) + (15 * np.sin(2 * np.pi * t / 7)) + np.random.normal(0, 5, 100)
    data_series = pd.Series(data)
    st.info("Using sample data (Growth + Weekly Wave). Upload your own for custom analysis.")

# --- 2. The Surgical Decomposition ---
# We use a 7-day period for weekly business cycles
decomp = seasonal_decompose(data_series, model='additive', period=7, extrapolate_trend='freq')

df = pd.DataFrame({
    'Actual': data_series.values,
    'Trend': decomp.trend.values,
    'Seasonal': decomp.seasonal.values,
    'Residual': decomp.resid.values
})

# --- 3. Dashboard Tabs ---
tab1, tab2 = st.tabs(["📉 Step 1: Strip the Patterns", "🔬 Step 2: Analyze the 'Noise' (True Risk)"])

with tab1:
    st.subheader("Peeling the Layers")
    st.write("We separate your demand into the three layers of the 'Business Sandwich'.")
    
    # Layered Plot
    fig_layers = go.Figure()
    fig_layers.add_trace(go.Scatter(y=df['Actual'], name="1. Raw Demand", line=dict(color='#CBD5E0')))
    fig_layers.add_trace(go.Scatter(y=df['Trend'], name="2. The Trend (Growth)", line=dict(color='#3182CE', width=4)))
    fig_layers.add_trace(go.Scatter(y=df['Seasonal'], name="3. The Wave (Seasonality)", line=dict(color='#F6AD55')))
    
    fig_layers.update_layout(title="Decomposing your Business Signal", xaxis_title="Days")
    st.plotly_chart(fig_layers, use_container_width=True)
    
    st.info("💡 **Why do this?** If we only looked at 'Raw Demand', the Friday spikes would look like 'Risk'. By peeling them away, we see they are just a 'Pattern'.")

with tab2:
    st.subheader("The Residual Analysis (The 'Noise')")
    st.write("This is what's left after removing the Trend and Seasonality. This is your **True Risk**.")
    
    # Clean the residuals for testing
    noise = df['Residual'].dropna()
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Histogram of NOISE
        fig_hist = px.histogram(noise, nbins=20, title="Distribution of the 'Noise'",
                                color_discrete_sequence=['#38A169'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_b:
        # Q-Q Plot of NOISE
        sorted_noise = np.sort(noise)
        norm = stats.norm.ppf(np.linspace(0.01, 0.99, len(noise)))
        fig_qq = px.scatter(x=norm, y=sorted_noise, title="Q-Q Plot: Is the Noise Normal?")
        fig_qq.add_shape(type="line", x0=min(norm), y0=min(sorted_noise), x1=max(norm), y1=max(sorted_noise),
                        line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig_qq, use_container_width=True)

    # --- THE FINAL VERDICT ---
    shapiro_p = stats.shapiro(noise)[1]
    
    st.divider()
    st.subheader("The Statistical Verdict")
    
    v1, v2 = st.columns(2)
    
    # Seasonality Strength
    resid_var = np.var(noise)
    total_var = np.var(df['Actual'] - df['Trend'])
    strength = max(0, (1 - (resid_var / total_var)) * 100)
    v1.metric("Predictability (Seasonality)", f"{strength:.1f}%")
    
    # Normality of Noise
    is_normal = shapiro_p > 0.05
    v2.metric("Noise Consistency (Normality)", "Normal" if is_normal else "Irregular", f"p={shapiro_p:.3f}")
    
    if is_normal:
        st.success("**Ready to Unlock Cash:** Your noise is consistent. Standard safety stock math on these residuals will safely minimize your inventory.")
    else:
        st.warning("**Proceed with Caution:** Your noise is irregular. Even after removing patterns, you have unpredictable 'freak events'. Keep a slightly higher buffer.")
