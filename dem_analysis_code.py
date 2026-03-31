import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Page Config ---
st.set_page_config(page_title="Demand Diagnostics", layout="wide")

st.title("🔍 Demand DNA: Normality, Seasonality & Trend")
st.markdown("Before optimizing inventory, we must understand the 'shape' of your business.")

# --- Data Input ---
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    col = st.sidebar.selectbox("Select Demand Column", df.columns)
    data = pd.to_numeric(df[col], errors='coerce').dropna().values
else:
    # Default Sample Data: Growth Trend + Weekly Seasonality + Noise
    t = np.arange(100)
    trend = 0.5 * t  # Upward growth
    seasonal = 10 * np.sin(2 * np.pi * t / 7)  # Weekly wave
    noise = np.random.normal(0, 5, 100)
    data = 50 + trend + seasonal + noise
    st.info("Using sample growth data. Upload your CSV for custom analysis.")

df_raw = pd.DataFrame(data, columns=["Demand"])

# --- 1. THE TREND (The Long-Term Direction) ---
st.subheader("📈 1. The Trend: Where are you going?")
# Simple Moving Average to show trend
df_raw['7-Day MA'] = df_raw['Demand'].rolling(window=7).mean()
df_raw['30-Day MA'] = df_raw['Demand'].rolling(window=30).mean()

fig_trend = px.line(df_raw, y=['Demand', '7-Day MA', '30-Day MA'], 
                    title="Raw Demand vs. Smoothed Trends",
                    color_discrete_map={"Demand": "#CBD5E0", "7-Day MA": "#F6AD55", "30-Day MA": "#3182CE"})
st.plotly_chart(fig_trend, use_container_width=True)

# --- 2. THE SEASONALITY (The Repeating Wave) ---
st.subheader("🌊 2. The Seasonality: Is it a pattern?")
if len(data) >= 14:
    decomp = seasonal_decompose(data, model='additive', period=7, extrapolate_trend='freq')
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_sea = px.line(y=decomp.seasonal[:14], title="The 7-Day Isolated Seasonal 'Wave'")
        fig_sea.update_layout(xaxis_title="Days", yaxis_title="Impact on Sales")
        st.plotly_chart(fig_sea, use_container_width=True)
    with col2:
        # Calculate Strength
        resid_var = np.var(decomp.resid)
        total_var = np.var(data - decomp.trend)
        strength = max(0, (1 - (resid_var / total_var)) * 100)
        st.metric("Seasonality Strength", f"{strength:.1f}%")
        st.write("A high percentage means your business is predictable. You can stock 'for the wave'.")
else:
    st.warning("Need at least 14 days of data to detect weekly patterns.")

# --- 3. THE NORMALITY (The Statistical Shape) ---
st.subheader("🔔 3. The Normality: Is it a Bell Curve?")
col_a, col_b = st.columns(2)

with col_a:
    fig_hist = px.histogram(df_raw, x="Demand", nbins=20, title="Data Distribution (Histogram)",
                            color_discrete_sequence=['#3182CE'])
    st.plotly_chart(fig_hist, use_container_width=True)

with col_b:
    # Q-Q Plot
    sorted_data = np.sort(data)
    norm = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
    fig_qq = px.scatter(x=norm, y=sorted_data, title="Q-Q Plot (Stay on the line for Normality)")
    fig_qq.add_shape(type="line", x0=min(norm), y0=min(sorted_data), x1=max(norm), y1=max(sorted_data),
                    line=dict(color="Red", dash="dash"))
    st.plotly_chart(fig_qq, use_container_width=True)

shapiro_p = stats.shapiro(data)[1]
if shapiro_p > 0.05:
    st.success(f"✅ **Normal Distribution Detected (p={shapiro_p:.3f}).** Standard safety stock math works well.")
else:
    st.warning(f"⚠️ **Non-Normal Data (p={shapiro_p:.3f}).** Your demand is irregular or 'lumpy'.")
