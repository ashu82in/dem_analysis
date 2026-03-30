import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Page Config ---
st.set_page_config(page_title="Inventory Cash Unlocker", layout="wide")

st.title("📊 Inventory & Demand Pattern Analyzer")
st.markdown("""
Unlock **'Frozen Cash'** by identifying predictable demand patterns. 
*Don't stock for the average; stock for the reality.*
""")

# --- 1. Sidebar & Sample Data Generation ---
st.sidebar.header("1. Data Input")

# Function to create a sample CSV for the user to download
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Create a robust sample dataset (60 days)
sample_dates = pd.date_range(start="2026-01-01", periods=60)
# Pattern: Low Mon-Thu, High Fri-Sun + some growth
sample_pattern = np.array([10, 12, 15, 20, 45, 55, 30]) 
sample_demand = [max(0, int(sample_pattern[i % 7] + np.random.normal(0, 5) + (i * 0.1))) for i in range(60)]
template_df = pd.DataFrame({"Date": sample_dates, "Demand": sample_demand})

st.sidebar.download_button(
    label="Download Template CSV",
    data=convert_df(template_df),
    file_name='demand_template.csv',
    mime='text/csv',
)

uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel", type=["csv", "xlsx"])

# Sidebar: Financial Constants
st.sidebar.header("2. Financial Constants")
unit_cost = st.sidebar.number_input("Cost per Unit (₹)", value=1000)
lead_time = st.sidebar.slider("Lead Time (Days to Restock)", 1, 30, 7)
service_level = st.sidebar.selectbox("Service Level (Confidence)", [0.90, 0.95, 0.99], index=1)
z_score = stats.norm.ppf(service_level)

# --- 2. Data Processing ---
if uploaded_file:
    df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    column_name = st.sidebar.selectbox("Select Demand Column", df_input.columns)
    data = df_input[column_name].values
else:
    st.info("No file uploaded. Showing analysis for the Template Data.")
    data = template_df["Demand"].values

try:
    df = pd.DataFrame(data, columns=["Demand"])
    
    # Seasonality Analysis
    has_seasonality = False
    if len(data) >= 14:
        decomposition = seasonal_decompose(df['Demand'], model='additive', period=7, extrapolate_trend='freq')
        df['Seasonal'] = decomposition.seasonal
        df['Trend'] = decomposition.trend
        df['Resid'] = decomposition.resid
        
        # Calculate strength
        resid_var = np.var(df['Resid'])
        total_var = np.var(df['Demand'] - df['Trend'])
        if (1 - (resid_var / total_var)) > 0.4:
            has_seasonality = True

    # --- 3. The "Cash Unlock" Calculation ---
    mean_val = np.mean(data)
    std_standard = np.std(data)
    std_seasonal = np.std(df['Resid']) if has_seasonality else std_standard
    
    # Safety Stock Comparison
    ss_standard = round(z_score * std_standard * np.sqrt(lead_time))
    ss_optimized = round(z_score * std_seasonal * np.sqrt(lead_time))
    
    cash_saved = (ss_standard - ss_optimized) * unit_cost

    # --- 4. Dashboard Metrics ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Average Daily Demand", f"{mean_val:.1f} units")
    m2.metric("Optimized Safety Stock", f"{ss_optimized} units", f"{- (ss_standard - ss_optimized)} units vs Std", delta_color="normal")
    m3.metric("Potential Cash Unlocked", f"₹{max(0, cash_saved):,}", delta="Reduced Excess", delta_color="inverse")

    # --- 5. Visualizations ---
    st.subheader("📊 Demand Insights")
    tabs = st.tabs(["Trend Analysis", "Weekly Pattern", "Simulation"])
    
    with tabs[0]:
        fig_trend = px.line(df, y=["Demand", "Trend"], 
                           title="Is your business growing? (Trend vs Actual)",
                           color_discrete_map={"Demand": "#CBD5E0", "Trend": "#3182CE"})
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with tabs[1]:
        if has_seasonality:
            weekly_avg = df['Seasonal'].iloc[:7].values
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            fig_bar = px.bar(x=days, y=weekly_avg, title="Predictable Weekly Peaks/Valleys",
                            labels={'x': 'Day of Week', 'y': 'Impact on Sales'})
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.write("Insufficient data to plot weekly patterns. Upload at least 14 days of data.")

    with tabs[2]:
        # Seasonally Aware Simulation
        sim_days = 30
        sim_results = []
        pattern = df['Seasonal'].iloc[:7].values if has_seasonality else np.zeros(7)
        
        for i in range(sim_days):
            day_demand = max(0, mean_val + pattern[i % 7] + np.random.normal(0, std_seasonal))
            sim_results.append(round(day_demand))
            
        sim_df = pd.DataFrame({"Day": range(1, sim_days+1), "Sim_Demand": sim_results})
        fig_sim = px.line(sim_df, x="Day", y="Sim_Demand", title="30-Day 'Stress Test' Simulation",
                         color_discrete_sequence=['#38A169'])
        fig_sim.add_hline(y=mean_val, line_dash="dash", annotation_text="Average")
        st.plotly_chart(fig_sim, use_container_width=True)

    # --- Final Actionable Advice ---
    st.success(f"""
    **Expert Insight:** By accounting for your weekly cycles, we found that your 'True Volatility' is actually lower than a standard calculation suggests. 
    You can safely reduce inventory by **{max(0, ss_standard - ss_optimized)} units**, freeing up **₹{max(0, cash_saved):,}** in working capital.
    """)

except Exception as e:
    st.error(f"Please ensure your data column contains numbers. Error: {e}")
