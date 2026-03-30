import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Page Config ---
st.set_page_config(page_title="Inventory Cash Unlocker", layout="wide")

st.title("📊 Inventory & Demand Pattern Analyzer")
st.markdown("""
Upload your sales data to find **'Frozen Cash'**. This version detects **Weekly Seasonality** to ensure your safety stock isn't over-inflated during slow periods.
""")

# --- 1. Data Input (File Upload) ---
st.sidebar.header("1. Upload Demand Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# Fallback to default if no file uploaded
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_excel(uploaded_file)
    
    # Assume demand is in the first column if not named 'Demand'
    column_name = st.sidebar.selectbox("Select Demand Column", df_input.columns)
    data = df_input[column_name].values
else:
    st.info("Using demo data. Upload your own in the sidebar to see custom results.")
    data = np.array([12, 18, 16, 28, 12, 13, 9, 30, 50, 37, 29, 33, 21, 43, 5, 34, 20, 19, 10, 15, 25, 23])

# Sidebar: Financial Constants
st.sidebar.header("2. Financial Constants")
unit_cost = st.sidebar.number_input("Cost per Unit (₹)", value=1000)
lead_time = st.sidebar.slider("Lead Time (Days to Restock)", 1, 30, 7)

# --- 2. Seasonality Detection Logic ---
try:
    df = pd.DataFrame(data, columns=["Demand"])
    
    # We need at least 14 days to detect a 7-day (weekly) pattern
    has_seasonality = False
    if len(data) >= 14:
        # Decompose the data (Additive model assumes spikes are constant in size)
        decomposition = seasonal_decompose(df['Demand'], model='additive', period=7, extrapolate_trend='freq')
        df['Seasonal'] = decomposition.seasonal
        df['Trend'] = decomposition.trend
        df['Resid'] = decomposition.resid
        
        # Calculate Seasonality Strength
        # If the 'Seasonal' variance is high relative to 'Residual' variance, seasonality is strong
        seasonality_strength = max(0, 1 - (np.var(df['Resid']) / np.var(df['Demand'] - df['Trend'])))
        if seasonality_strength > 0.4: # Threshold for "significant" seasonality
            has_seasonality = True

    # --- 3. Metrics Calculation ---
    mean_val = np.mean(data)
    # Use Residual Std Dev if seasonal, otherwise use total Std Dev
    # This prevents over-calculating safety stock due to predictable peaks
    std_to_use = np.std(df['Resid']) if has_seasonality else np.std(data)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Daily Demand", f"{mean_val:.2f} units")
    col2.metric("Unpredictable Volatility", f"{std_to_use:.2f}")
    
    safety_stock = round(1.65 * std_to_use * np.sqrt(lead_time))
    col3.metric("Optimized Safety Stock", f"{safety_stock} units")
    
    blocked_cash = safety_stock * unit_cost
    col4.metric("Capital Tied in Safety Stock", f"₹{blocked_cash:,}")

    # --- 4. Visualizations ---
    st.subheader("📈 Demand Pattern & Seasonality")
    c1, c2 = st.columns(2)
    
    with c1:
        # Show actual vs trend
        fig_trend = px.line(df, y=["Demand", "Trend"], title="Demand Trend vs. Actual",
                           color_discrete_map={"Demand": "#1f77b4", "Trend": "#ff7f0e"})
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with c2:
        if has_seasonality:
            # Show the weekly 'wave'
            weekly_pattern = df['Seasonal'].iloc[:7].values
            fig_sea = px.bar(x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], 
                             y=weekly_pattern, title="Weekly Seasonal Impact (Units +/- from Mean)")
            st.plotly_chart(fig_sea, use_container_width=True)
        else:
            st.write("Not enough data or no strong weekly pattern detected to show seasonality.")

    # --- 5. Seasonal Monte Carlo Simulation ---
    st.divider()
    st.subheader("🔮 Seasonally-Aware 30-Day Simulation")
    
    sim_days = 30
    sim_results = []
    
    for i in range(sim_days):
        # Base random noise
        noise = np.random.normal(0, std_to_use)
        # Seasonal boost/drop for that day of the week
        seasonal_effect = weekly_pattern[i % 7] if has_seasonality else 0
        # Final simulated day
        day_demand = max(0, mean_val + seasonal_effect + noise)
        sim_results.append(round(day_demand))
    
    sim_df = pd.DataFrame({"Day": range(1, sim_days+1), "Simulated_Demand": sim_results})
    fig_sim = px.line(sim_df, x="Day", y="Simulated_Demand", title="Simulated Future (Includes Seasonal Waves)")
    st.plotly_chart(fig_sim, use_container_width=True)

    # --- Final Insight ---
    if has_seasonality:
        st.success(f"**Seasonality Alert:** We found a strong weekly pattern. By isolating the 'predictable' spikes, we reduced your required safety stock by focusing only on the **unpredictable** noise.")
    else:
        st.warning("No clear seasonality found. Standard safety stock calculations applied.")

except Exception as e:
    st.error(f"Error processing data: {e}")
