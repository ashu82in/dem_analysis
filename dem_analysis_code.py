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
Stop stocking for the 'Average'. This tool identifies **Weekly Waves** in your demand 
to reveal exactly how much **Frozen Cash** you can pull out of your warehouse.
""")

# --- 1. Sidebar & Sample Data Logic ---
st.sidebar.header("1. Data Input")

def get_template_df():
    # Create a robust sample dataset (60 days) with weekly seasonality
    dates = pd.date_range(start="2026-01-01", periods=60)
    # Pattern: Mon-Thu (Low), Fri-Sun (High)
    pattern = np.array([12, 14, 18, 22, 48, 55, 35]) 
    demand = [max(0, int(pattern[i % 7] + np.random.normal(0, 4) + (i * 0.1))) for i in range(60)]
    return pd.DataFrame({"Date": dates, "Demand": demand})

template_df = get_template_df()

st.sidebar.download_button(
    label="Download Template CSV",
    data=template_df.to_csv(index=False).encode('utf-8'),
    file_name='demand_template.csv',
    mime='text/csv',
)

uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Financial Constants")
unit_cost = st.sidebar.number_input("Cost per Unit (₹)", value=1000, min_value=1)
lead_time = st.sidebar.slider("Lead Time (Days to Restock)", 1, 30, 7)
service_level = st.sidebar.selectbox("Service Level (Confidence)", [0.90, 0.95, 0.99], index=1)
z_score = stats.norm.ppf(service_level)

# --- 2. Data Processing ---
if uploaded_file:
    try:
        df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        column_name = st.sidebar.selectbox("Select Demand Column", df_input.columns)
        data = pd.to_numeric(df_input[column_name], errors='coerce').dropna().values
    except Exception as e:
        st.error(f"Error loading file: {e}")
        data = template_df["Demand"].values
else:
    st.info("Using Template Data. Upload your own file in the sidebar to analyze your business.")
    data = template_df["Demand"].values

try:
    df = pd.DataFrame(data, columns=["Demand"])
    
    # Seasonality & Trend Analysis
    has_seasonality = False
    if len(data) >= 14:
        # Period 7 for Weekly cycles
        decomp = seasonal_decompose(df['Demand'], model='additive', period=7, extrapolate_trend='freq')
        df['Seasonal'] = decomp.seasonal
        df['Trend'] = decomp.trend
        df['Resid'] = decomp.resid
        
        # Check Seasonality Strength
        resid_var = np.var(df['Resid'].dropna())
        total_var = np.var(df['Demand'] - df['Trend'])
        if (1 - (resid_var / total_var)) > 0.3: # Threshold for detection
            has_seasonality = True

    # --- 3. The "Dynamic Cash" Math ---
    mean_val = np.mean(data)
    # Use standard deviation of residuals (the unpredicted noise)
    std_noise = np.std(df['Resid'].dropna()) if has_seasonality else np.std(data)
    
    # Base buffer for unpredictable noise
    base_ss = z_score * std_noise * np.sqrt(lead_time)
    
    # Weekly Indexing
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if has_seasonality:
        seasonal_pattern = df['Seasonal'].iloc[:7].values
        seasonal_indices = (mean_val + seasonal_pattern) / mean_val
    else:
        seasonal_indices = np.ones(7)

    # Calculation for Dashboard
    ss_standard = round(z_score * np.std(data) * np.sqrt(lead_time))
    ss_dynamic_avg = round(base_ss * np.mean(seasonal_indices))
    cash_unlocked = (ss_standard - ss_dynamic_avg) * unit_cost

    # --- 4. Main Dashboard Metrics ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Average Daily Demand", f"{mean_val:.1f} units")
    m2.metric("Optimal Safety Stock (Avg)", f"{ss_dynamic_avg} units", f"{- (ss_standard - ss_dynamic_avg)} vs Static")
    m3.metric("Potential Cash Unlocked", f"₹{max(0, cash_unlocked):,}", delta="Working Capital", delta_color="normal")

    # --- 5. Tabs & Visuals ---
    tabs = st.tabs(["📉 Trend & Normality", "📅 Weekly Playbook", "🔮 Stress Test Simulation"])
    
    with tabs[0]:
        st.subheader("Long-term Growth vs. Noise")
        fig_trend = px.line(df, y=["Demand", "Trend"], 
                           color_discrete_map={"Demand": "#CBD5E0", "Trend": "#3182CE"},
                           title="Is the business growing or shrinking?")
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Normality Check
        check_data = df['Resid'].dropna() if has_seasonality else data
        shapiro_p = stats.shapiro(check_data)[1]
        if shapiro_p > 0.05:
            st.success(f"✅ **Statistical Reliability High:** Noise is normally distributed (p={shapiro_p:.2f}).")
        else:
            st.warning(f"⚠️ **Irregular Volatility:** Demand has 'fat tails' (p={shapiro_p:.2f}). Consider adding 5-10% extra buffer.")

    with tabs[1]:
        st.subheader("Your 7-Day Inventory Schedule")
        schedule_list = []
        for i, day_name in enumerate(days):
            daily_ss = round(base_ss * seasonal_indices[i])
            schedule_list.append({"Day": day_name, "Safety Stock": daily_ss, "Value (₹)": f"₹{daily_ss * unit_cost:,}"})
        
        sched_df = pd.DataFrame(schedule_list)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.table(sched_df)
        with c2:
            fig_wave = px.line(sched_df, x="Day", y="Safety Stock", title="The Safety Stock Wave", markers=True)
            fig_wave.update_traces(fill='tozeroy', line_color='#3182CE')
            st.plotly_chart(fig_wave, use_container_width=True)

    with tabs[2]:
        st.subheader("Monte Carlo Simulation (30 Days)")
        sim_days = 30
        sim_results = []
        pattern = df['Seasonal'].iloc[:7].values if has_seasonality else np.zeros(7)
        
        for i in range(sim_days):
            # Base + Seasonality + Random Noise
            d = max(0, mean_val + pattern[i % 7] + np.random.normal(0, std_noise))
            sim_results.append(round(d))
            
        sim_df = pd.DataFrame({"Day": range(1, sim_days+1), "Forecasted_Demand": sim_results})
        fig_sim = px.line(sim_df, x="Day", y="Forecasted_Demand", title="Future Forecast includes Weekly Cycles")
        fig_sim.add_hline(y=mean_val, line_dash="dash", annotation_text="Average")
        st.plotly_chart(fig_sim, use_container_width=True)

    st.divider()
    st.info(f"**Actionable Insight:** By switching to a dynamic strategy, you shift inventory from slow days to peak days. This protects your service level while freeing up **₹{max(0, cash_unlocked):,}** in cash.")

except Exception as e:
    st.error(f"Something went wrong with the data processing: {e}")
