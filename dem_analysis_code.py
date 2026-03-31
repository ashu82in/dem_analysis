import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Page Config ---
st.set_page_config(page_title="Demand DNA: Bucket Optimizer", layout="wide")

st.title("🧬 Demand DNA: Professional Bucket Analysis")
st.markdown("""
Distributors often club daily demand into buckets (e.g., weekly totals) to stabilize their supply chain.
**Adjust the offset and window below** to see how 'Clubbing' changes the shape of your risk.
""")

# --- 1. Sidebar: Data & Bucket Controls ---
st.sidebar.header("1. Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Custom Bucket Settings")
window_size = st.sidebar.number_input("Clubbing Window (Days)", min_value=1, max_value=30, value=7)
start_offset = st.sidebar.slider("Start Date Offset (Days)", 0, window_size-1, 0)

target_col = "Order_Demand"
data_series_daily = None

if uploaded_file:
    try:
        # Load logic (CSV vs Excel)
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            xl = pd.ExcelFile(uploaded_file)
            sheet = st.sidebar.selectbox("Select Excel Sheet", xl.sheet_names)
            df_raw = xl.parse(sheet)
        
        # Clean Dates and Demand
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
        df_raw = df_raw.dropna(subset=['Date'])
        
        # Hardcoded for your specific column name
        if target_col not in df_raw.columns:
            target_col = st.sidebar.selectbox("Column 'Order_Demand' not found. Select manually:", df_raw.columns)
        
        # AGGREGATION: Summing orders by day
        df_daily = df_raw.groupby('Date')[target_col].sum().sort_index()
        
        # FILLING GAPS: Add 0 for days with no data
        full_range = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
        data_series_daily = df_daily.reindex(full_range, fill_value=0)
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()
else:
    # Sample "Distributor-style" Data
    t = pd.date_range(start="2026-01-01", periods=180)
    vals = [np.random.randint(500, 2500) if np.random.random() > 0.8 else 0 for _ in range(180)]
    data_series_daily = pd.Series(vals, index=t)
    st.info("💡 Using sample data. Upload your file to analyze 'Order_Demand'.")

# --- 2. CLUBBING LOGIC (The Bucket Surgery) ---
# Apply the user-defined offset
shifted_start = data_series_daily.index.min() + pd.Timedelta(days=start_offset)
data_to_club = data_series_daily[data_series_daily.index >= shifted_start]

# Aggregate into the chosen buckets (e.g. 7D, 14D)
clubbed_series = data_to_club.resample(f'{window_size}D').sum()

# --- 3. Dashboard Tabs ---
tab1, tab2, tab3 = st.tabs(["📦 Step 1: Bucket Analysis", "🔬 Step 2: Statistical DNA", "🔮 Step 3: Logistics Forecast"])

with tab1:
    st.subheader(f"Demand Clubbed into {window_size}-Day Windows")
    st.write(f"Analyzing total load starting from: **{shifted_start.date()}**")
    
    # Summary Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Buckets", len(clubbed_series))
    m2.metric("Peak Bucket Load", f"{clubbed_series.max():.0f} units")
    m3.metric("Avg. Bucket Load", f"{clubbed_series.mean():.0f} units")

    # A. Time-Series Bar Chart
    fig_bar = px.bar(x=clubbed_series.index, y=clubbed_series.values, 
                      title=f"Total Order_Demand per {window_size}-Day Period",
                      labels={'x': 'Bucket Period Start', 'y': 'Total Units'},
                      color_discrete_sequence=['#3182CE'])
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.divider()
    
    # B. Histogram of Bucket Sizes (The user's request)
    st.subheader("The Distribution of Bucket Loads")
    st.write("This histogram shows how often your 'Clubbed' demand hits specific volumes. A narrow, centered peak is easier to manage than a spread-out one.")
    
    fig_hist = px.histogram(clubbed_series, x=clubbed_series.values, nbins=15,
                           title=f"Frequency of Bucket Magnitudes ({window_size} Days)",
                           labels={'x': 'Units per Bucket', 'y': 'Count'},
                           color_discrete_sequence=['#38A169']) # Green for 'Stability'
    fig_hist.update_layout(bargap=0.1)
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.subheader("Statistical Normality Check")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Q-Q Plot (The Diagonal Test)**")
        # Standard Q-Q plot logic
        sorted_data = np.sort(clubbed_series.values)
        norm = stats.norm.ppf(np.linspace(0.01, 0.99, len(clubbed_series)))
        fig_qq = px.scatter(x=norm, y=sorted_data, labels={'x': 'Theoretical Normality', 'y': 'Actual Bucket Load'})
        fig_qq.add_shape(type="line", x0=min(norm), y0=min(sorted_data), x1=max(norm), y1=max(sorted_data), line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig_qq, use_container_width=True)
        
    with col_b:
        st.write("**Normality Verdict**")
        if len(clubbed_series) > 3:
            shapiro_p = stats.shapiro(clubbed_series)[1]
            if shapiro_p > 0.05:
                st.success(f"✅ **Normal Distribution (p={shapiro_p:.3f})**")
                st.write(f"At a {window_size}-day window with a {start_offset}-day offset, your demand becomes predictable. You can use standard math to unlock cash.")
            else:
                st.warning(f"⚠️ **Non-Normal Distribution (p={shapiro_p:.3f})**")
                st.write("Even after clubbing, your demand is 'Lumpy'. You may need a higher safety buffer for freak events.")

with tab3:
    st.subheader(f"Next {window_size}-Day Logistics Forecast")
    if len(clubbed_series) > 5:
        # Predict the next 3 buckets
        model = ExponentialSmoothing(clubbed_series, trend='add', seasonal=None).fit()
        forecast = model.forecast(3)
        
        fig_fore = go.Figure()
        fig_fore.add_trace(go.Bar(x=clubbed_series.index[-10:], y=clubbed_series.values[-10:], name="Recent History", marker_color='#CBD5E0'))
        fig_fore.add_trace(go.Bar(x=forecast.index, y=forecast.values, name="Forecasted Load", marker_color='#3182CE'))
        fig_fore.update_layout(title="Booking Forecast for Upcoming Buckets")
        st.plotly_chart(fig_fore, use_container_width=True)
        
        st.success(f"**Action Plan:** Ensure your logistics can handle ~{forecast.iloc[0]:.0f} units in the next {window_size} days.")
    else:
        st.error("Insufficient data points for forecasting. Try reducing the window size.")
