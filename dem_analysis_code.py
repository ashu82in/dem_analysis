import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Page Config ---
st.set_page_config(page_title="Demand DNA: Clubbing & Buckets", layout="wide")

st.title("🧬 Demand DNA: Professional Bucket Analysis")
st.markdown("""
Distributors often need to **Club Demand** into buckets (e.g., weekly totals) to see the true load. 
Use the sidebar to change the **Window Size** and **Start Date** to see how it affects your 'Shape of Risk'.
""")

# --- 1. Sidebar: Data & Bucket Controls ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Custom Bucket Settings")
window_size = st.sidebar.number_input("Clubbing Window (Days)", min_value=1, max_value=30, value=7)
start_offset = st.sidebar.slider("Start Date Offset (Days)", 0, window_size-1, 0)

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
        df_raw = df_raw.dropna(subset=['Date'])
        
        # Initial Daily Aggregation
        df_daily = df_raw.groupby('Date')[target_col].sum().sort_index()
        full_range = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
        data_series_daily = df_daily.reindex(full_range, fill_value=0)
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    # Sample Data
    t = pd.date_range(start="2026-01-01", periods=180)
    vals = [np.random.randint(500, 2000) if np.random.random() > 0.8 else 0 for _ in range(180)]
    data_series_daily = pd.Series(vals, index=t)
    st.info("💡 Using sample data. Upload your file to analyze 'Order_Demand'.")

# --- 2. CLUBBING LOGIC (The "Bucket" Surgery) ---
# Apply the offset by shifting the start date
shifted_start = data_series_daily.index.min() + pd.Timedelta(days=start_offset)
data_to_club = data_series_daily[data_series_daily.index >= shifted_start]

# Clubbing into buckets
clubbed_series = data_to_club.resample(f'{window_size}D').sum()

# --- 3. Diagnostics ---
tab1, tab2, tab3 = st.tabs(["📦 Step 1: Clubbed Demand", "🔬 Step 2: Bucket Normality", "🔮 Step 3: Surgical Forecast"])

with tab1:
    st.subheader(f"Demand Clubbed into {window_size}-Day Buckets")
    st.write(f"Starting from: {shifted_start.date()} (Offset: {start_offset} days)")
    
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Total Buckets", len(clubbed_series))
    col_m2.metric("Avg. Load per Bucket", f"{clubbed_series.mean():.0f} units")

    fig_club = px.bar(x=clubbed_series.index, y=clubbed_series.values, 
                      title=f"Demand Load per {window_size}-Day Period",
                      labels={'x': 'Bucket Start Date', 'y': 'Total Demand'})
    fig_club.update_traces(marker_color='#3182CE')
    st.plotly_chart(fig_club, use_container_width=True)
    
    st.info("💡 **Why this matters:** Notice how changing the 'Offset' in the sidebar shifts the peaks. This helps you identify if your peaks are tied to specific calendar days.")

with tab2:
    st.subheader("Statistical 'Shape' of the Buckets")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Bucket Distribution (Histogram)**")
        fig_hist = px.histogram(clubbed_series, nbins=20, color_discrete_sequence=['#38A169'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_b:
        st.write("**Bucket Q-Q Plot**")
        sorted_data = np.sort(clubbed_series.values)
        norm = stats.norm.ppf(np.linspace(0.01, 0.99, len(clubbed_series)))
        fig_qq = px.scatter(x=norm, y=sorted_data, labels={'x': 'Theoretical', 'y': 'Actual'})
        fig_qq.add_shape(type="line", x0=min(norm), y0=min(sorted_data), x1=max(norm), y1=max(sorted_data), line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig_qq, use_container_width=True)

    # Normality Verdict
    if len(clubbed_series) > 3:
        shapiro_p = stats.shapiro(clubbed_series)[1]
        st.divider()
        if shapiro_p > 0.05:
            st.success(f"✅ **Verdict:** These {window_size}-day buckets are **Normal** (p={shapiro_p:.3f}). You can use standard safety stock math for this window.")
        else:
            st.warning(f"⚠️ **Verdict:** Buckets are **Non-Normal** (p={shapiro_p:.3f}). Aggregation did not remove the 'Lumpiness'.")

with tab3:
    st.subheader("Future Bucket Forecast")
    if len(clubbed_series) > 10:
        # Forecast for next 4 buckets
        model = ExponentialSmoothing(clubbed_series, trend='add', seasonal=None).fit()
        forecast = model.forecast(4)
        
        fig_fore = go.Figure()
        fig_fore.add_trace(go.Bar(x=clubbed_series.index, y=clubbed_series.values, name="History", marker_color='#CBD5E0'))
        fig_fore.add_trace(go.Bar(x=forecast.index, y=forecast.values, name="Forecast", marker_color='#3182CE'))
        st.plotly_chart(fig_fore, use_container_width=True)
        st.write(f"**Action Plan:** Prepare for a total load of {forecast.iloc[0]:.0f} units in the next {window_size}-day window.")
    else:
        st.error("Not enough buckets to generate a forecast. Increase your data range or decrease window size.")
