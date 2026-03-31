import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Page Config ---
st.set_page_config(page_title="Advanced Demand DNA", layout="wide")

st.title("🧬 Demand DNA: Aggregation & Rolling Normality")
st.markdown("""
This version **groups multiple orders by day** and **fills gaps with 0**. 
We also analyze how 'Normal' your demand is across different time windows.
""")

# --- 1. Sidebar: Data & Settings ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# Settings for the window check
st.sidebar.header("2. Analysis Windows")
norm_window = st.sidebar.slider("Normality Check Window (Days)", 7, 60, 15)

data_series = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            xl = pd.ExcelFile(uploaded_file)
            sheet = st.sidebar.selectbox("Select Sheet", xl.sheet_names)
            df_raw = xl.parse(sheet)
        
        # 1. Clean Dates and Column
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
        df_raw = df_raw.dropna(subset=['Date'])
        target_col = "Order_Demand"
        
        # 2. AGGREGATION: Sum up same-day orders
        df_daily = df_raw.groupby('Date')[target_col].sum().sort_index()
        
        # 3. FILLING GAPS: Add 0 for days with no data
        full_range = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
        data_series = df_daily.reindex(full_range, fill_value=0)
        
        st.sidebar.success(f"Aggregated into {len(data_series)} consecutive days.")
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()
else:
    # Sample Data (Trended, Seasonal, with 0s)
    t = pd.date_range(start="2026-01-01", periods=120)
    vals = 50 + (0.5 * np.arange(120)) + (20 * np.sin(2 * np.pi * np.arange(120) / 7)) + np.random.normal(0, 10, 120)
    data_series = pd.Series(vals, index=t).apply(lambda x: max(0, x))
    st.info("💡 Using sample data. Upload your file to analyze 'Order_Demand'.")

# --- 2. Rolling Normality Math ---
def rolling_normality(series, window):
    # Calculate Shapiro-Wilk p-value for a sliding window
    p_values = []
    for i in range(len(series) - window + 1):
        window_data = series.iloc[i : i + window]
        if window_data.std() == 0: # Avoid errors on flat data
            p_values.append(0)
        else:
            _, p = stats.shapiro(window_data)
            p_values.append(p)
    return pd.Series(p_values, index=series.index[window - 1 :])

rolling_p = rolling_normality(data_series, norm_window)

# --- 3. Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Aggregated Trend", "🔬 Rolling Normality", "📊 Distribution"])

with tab1:
    st.subheader("Total Daily Demand (Aggregated)")
    st.write("Each point below is the **sum of all orders** for that day, including 0s for quiet days.")
    fig_trend = px.line(x=data_series.index, y=data_series.values, 
                        title=f"Total Daily Order_Demand (with 0-filling)",
                        labels={'x': 'Date', 'y': 'Total Units'})
    fig_trend.update_traces(line_color='#3182CE')
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.subheader(f"Rolling {norm_window}-Day Normality Check")
    st.write(f"This chart shows how 'Normal' the last {norm_window} days were. Values above 0.05 mean the business was statistically stable.")
    
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=rolling_p.index, y=rolling_p.values, name="p-value", line=dict(color='#805AD5')))
    fig_p.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="Normality Threshold (0.05)")
    
    fig_p.update_layout(title=f"Is my demand 'Normal' over {norm_window}-day windows?", yaxis_title="p-value")
    st.plotly_chart(fig_p, use_container_width=True)
    
    st.info("💡 **How to read this:** When the line stays ABOVE the red dash, your demand is predictable (Normal). When it dips BELOW, you are experiencing 'Lumpy' or erratic demand.")

with tab3:
    st.subheader("Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Histogram (Includes 0-demand days)**")
        fig_hist = px.histogram(data_series, nbins=30, color_discrete_sequence=['#3182CE'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        st.write("**Q-Q Plot (Diagnostic)**")
        sorted_data = np.sort(data_series.values)
        norm = stats.norm.ppf(np.linspace(0.01, 0.99, len(data_series)))
        fig_qq = px.scatter(x=norm, y=sorted_data, labels={'x': 'Theoretical', 'y': 'Actual'})
        fig_qq.add_shape(type="line", x0=min(norm), y0=min(sorted_data), x1=max(norm), y1=max(sorted_data), line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig_qq, use_container_width=True)
