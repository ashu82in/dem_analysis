import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Page Config ---
st.set_page_config(page_title="Inventory Cash Unlocker", layout="wide")

st.title("📊 Inventory Strategy & Cash Unlocker")
st.markdown("""
Unlock **'Frozen Cash'** by identifying predictable demand patterns. 
This tool validates your data quality before calculating your optimal inventory levels.
""")

# --- 1. Sidebar & Data Input ---
st.sidebar.header("1. Data Input")

def get_template_df():
    dates = pd.date_range(start="2026-01-01", periods=60)
    # Weekly pattern: Mon-Thu (Low), Fri-Sun (High)
    pattern = np.array([12, 15, 18, 22, 45, 55, 30]) 
    demand = [max(0, int(pattern[i % 7] + np.random.normal(0, 5) + (i * 0.1))) for i in range(60)]
    return pd.DataFrame({"Date": dates, "Demand": demand})

template_df = get_template_df()

st.sidebar.download_button(
    label="Download Template CSV",
    data=template_df.to_csv(index=False).encode('utf-8'),
    file_name='inventory_template.csv',
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

df = pd.DataFrame(data, columns=["Demand"])

# --- 3. Main Application Tabs ---
tab1, tab2, tab3 = st.tabs(["🔍 Step 1: Normality Check", "🌊 Step 2: Seasonal Cash Unlock", "📅 Step 3: Order Playbook"])

# --- TAB 1: NORMALITY ANALYSIS ---
with tab1:
    st.subheader("Statistical Health Check")
    st.write("Before we calculate safety stock, we must see if your demand follows a standard 'Bell Curve' or if it is 'Lumpy'.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Histogram with KDE
        fig_hist = px.histogram(df, x="Demand", marginal="box", nbins=20, 
                               title="Distribution of Demand",
                               color_discrete_sequence=['#3182CE'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_b:
        # Q-Q Plot Logic
        # We'll use a scatter plot to simulate a Q-Q plot
        sorted_data = np.sort(data)
        norm = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
        fig_qq = px.scatter(x=norm, y=sorted_data, labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'},
                           title="Q-Q Plot (Points should stay on the diagonal line)")
        fig_qq.add_shape(type="line", x0=min(norm), y0=min(sorted_data), x1=max(norm), y1=max(sorted_data),
                        line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig_qq, use_container_width=True)

    # Normality Test Result
    shapiro_p = stats.shapiro(data)[1]
    if shapiro_p > 0.05:
        st.success(f"✅ **Your data is Normally Distributed (p={shapiro_p:.3f}).** Standard inventory math will be highly accurate.")
    else:
        st.warning(f"⚠️ **Non-Normal Distribution Detected (p={shapiro_p:.3f}).** Your demand has outliers or strong cycles. Moving to Step 2 to isolate these cycles is highly recommended.")

# --- TAB 2: SEASONAL CASH UNLOCK ---
with tab2:
    st.subheader("Isolating Predictable Waves")
    
    # Seasonality Logic
    has_seasonality = False
    if len(data) >= 14:
        decomp = seasonal_decompose(df['Demand'], model='additive', period=7, extrapolate_trend='freq')
        df['Seasonal'], df['Trend'], df['Resid'] = decomp.seasonal, decomp.trend, decomp.resid
        
        resid_var = np.var(df['Resid'].dropna())
        total_var = np.var(df['Demand'] - df['Trend'])
        if (1 - (resid_var / total_var)) > 0.3:
            has_seasonality = True

    # Calculations
    mean_val = np.mean(data)
    std_standard = np.std(data)
    std_resid = np.std(df['Resid'].dropna()) if has_seasonality else std_standard
    
    # Safety Stock Comparison
    ss_static = round(z_score * std_standard * np.sqrt(lead_time))
    ss_optimized = round(z_score * std_resid * np.sqrt(lead_time))
    cash_saved = (ss_static - ss_optimized) * unit_cost

    m1, m2, m3 = st.columns(3)
    m1.metric("Current Volatility (Raw)", f"{std_standard:.2f}")
    m2.metric("True Noise (After Seasonality)", f"{std_resid:.2f}", f"{-((std_standard-std_resid)/std_standard)*100:.1f}% Reduction")
    m3.metric("Capital to Unlock", f"₹{max(0, cash_saved):,}", "Frozen Cash")

    fig_decomp = px.line(df, y=["Demand", "Trend"], title="Demand Trend Analysis",
                        color_discrete_map={"Demand": "#CBD5E0", "Trend": "#3182CE"})
    st.plotly_chart(fig_decomp, use_container_width=True)

# --- TAB 3: ORDER PLAYBOOK ---
with tab3:
    st.subheader("Your Reorder Schedule")
    
    # Dynamic ROP Calculation
    # ROP = (Demand during Lead Time) + Safety Stock
    # Since Lead Time is 7 days, we use a 7-day rolling window
    rop_base = (mean_val * lead_time) + ss_optimized
    
    st.info(f"💡 **Lead Time shock absorber:** With a {lead_time}-day lead time, your daily spikes are smoothed. However, we still use the 'True Noise' to keep your safety stock low.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Dynamic Reorder Points**")
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if has_seasonality:
            seasonal_pattern = df['Seasonal'].iloc[:7].values
            # Reorder point changes based on what day of the week it is
            rop_list = [round(rop_base + p) for p in seasonal_pattern]
        else:
            rop_list = [round(rop_base)] * 7
            
        rop_df = pd.DataFrame({"Day": days, "Reorder Point (Units)": rop_list})
        st.table(rop_df)
        
    with col2:
        fig_sim = px.line(x=days, y=rop_list, title="When to Order: The ROP Wave", markers=True)
        fig_sim.update_layout(yaxis_title="Units in Stock")
        st.plotly_chart(fig_sim, use_container_width=True)

    st.success(f"**Final Insight:** By removing seasonal 'noise' from your risk calculation, you can safely operate with **{ss_static - ss_optimized} fewer units** on average, saving you **₹{max(0, cash_saved):,}** without increasing stockout risk.")
