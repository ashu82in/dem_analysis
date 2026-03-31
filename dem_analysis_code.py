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
This diagnostic strips away the **Trend** and **Seasonality** to reveal your **True Noise**. 
By testing the 'Noise' for normality, we calculate the most accurate risk level for your inventory.
""")

# --- 1. Sidebar: Excel & CSV Input ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

data_series = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            # Handle Excel with multiple sheets
            xl = pd.ExcelFile(uploaded_file)
            sheet = st.sidebar.selectbox("Select Excel Sheet", xl.sheet_names)
            df_raw = xl.parse(sheet)
        
        # Look for "Order Demand" automatically
        default_col = "Order_Demand" if "Order_Demand" in df_raw.columns else df_raw.columns[0]
        col = st.sidebar.selectbox("Select Demand Column", df_raw.columns, index=list(df_raw.columns).index(default_col))
        
        # Process numeric data
        data_series = pd.to_numeric(df_raw[col], errors='coerce').dropna()
        
        if data_series.empty:
            st.error(f"The column '{col}' contains no numeric data. Please check your file.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    # Default Sample Data (Trend + Seasonality + Noise)
    t = np.arange(100)
    data = 50 + (0.4 * t) + (12 * np.sin(2 * np.pi * t / 7)) + np.random.normal(0, 4, 100)
    data_series = pd.Series(data)
    st.info("💡 No file uploaded. Using sample data. Upload your file to analyze 'Order_Demand'.")

# --- 2. Surgical Decomposition ---
# Period 7 for Weekly cycles
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
    st.subheader("Peeling the Business Layers")
    st.write("We separate 'Order Demand' into three distinct layers to identify what is predictable.")
    
    fig_layers = go.Figure()
    fig_layers.add_trace(go.Scatter(y=df['Actual'], name="1. Raw Demand", line=dict(color='#CBD5E0', width=1)))
    fig_layers.add_trace(go.Scatter(y=df['Trend'], name="2. The Trend (Growth)", line=dict(color='#3182CE', width=4)))
    fig_layers.add_trace(go.Scatter(y=df['Seasonal'], name="3. The Wave (Seasonality)", line=dict(color='#F6AD55', width=2)))
    
    fig_layers.update_layout(title="Signal Decomposition", xaxis_title="Time (Days)", yaxis_title="Units")
    st.plotly_chart(fig_layers, use_container_width=True)
    
    st.info("**The Logic:** The Blue line is your direction; the Orange line is your cycle. The leftover 'Static' is the only thing that requires Safety Stock.")

with tab2:
    st.subheader("The Residual Analysis (The 'Noise')")
    st.write("This is the 'Static' left after the patterns are gone. We test this for **Normality**.")
    
    noise = df['Residual'].dropna()
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_hist = px.histogram(noise, nbins=20, title="Distribution of the Noise", color_discrete_sequence=['#38A169'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_b:
        # Q-Q Plot
        sorted_noise = np.sort(noise)
        norm = stats.norm.ppf(np.linspace(0.01, 0.99, len(noise)))
        fig_qq = px.scatter(x=norm, y=sorted_noise, title="Q-Q Plot: Is the Noise Normal?")
        fig_qq.add_shape(type="line", x0=min(norm), y0=min(sorted_noise), x1=max(norm), y1=max(sorted_noise),
                        line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig_qq, use_container_width=True)

    # --- THE FINAL VERDICT ---
    shapiro_p = stats.shapiro(noise)[1]
    resid_var = np.var(noise)
    total_var = np.var(df['Actual'] - df['Trend'])
    strength = max(0, (1 - (resid_var / total_var)) * 100)
    
    st.divider()
    st.subheader("Statistical Verdict for 'Order Demand'")
    
    v1, v2 = st.columns(2)
    v1.metric("Predictability (Seasonality)", f"{strength:.1f}%")
    
    is_normal = shapiro_p > 0.05
    v2.metric("Noise Consistency (Normality)", "Normal" if is_normal else "Irregular", f"p={shapiro_p:.3f}")
    
    if is_normal:
        st.success("**Ready to Unlock Cash:** Your noise is consistent. Your current Safety Stock is likely over-budgeted for predictable waves.")
    else:
        st.warning("**Watch for Outliers:** Your noise is irregular. Even after removing patterns, you have unpredictable spikes.")
