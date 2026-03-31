import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Page Config ---
st.set_page_config(page_title="Inventory Cash Unlocker", layout="wide")

st.title("📊 Strategic Inventory & Cash Analyzer")
st.markdown("""
Identify **Frozen Cash** by separating predictable patterns from random noise.
This tool detects weekly waves, yearly peaks, and one-time 'freak events'.
""")

# --- 1. Sidebar & Data Input ---
st.sidebar.header("1. Data Input")

def get_template_df():
    # Create 90 days of data with a weekly pattern and two major "erratic" spikes
    dates = pd.date_range(start="2026-01-01", periods=90)
    pattern = np.array([10, 12, 15, 20, 45, 55, 30]) 
    demand = [max(0, int(pattern[i % 7] + np.random.normal(0, 5))) for i in range(90)]
    # Add two "Erratic/Yearly" Spikes (Outliers)
    demand[25] = 120  # Promo 1
    demand[70] = 135  # Promo 2
    return pd.DataFrame({"Date": dates, "Demand": demand})

template_df = get_template_df()

st.sidebar.download_button(
    label="Download Template CSV",
    data=template_df.to_csv(index=False).encode('utf-8'),
    file_name='inventory_template.csv',
    mime='text/csv',
)

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Seasonality Type")
cycle_option = st.sidebar.selectbox("Typical Business Cycle", 
                                   ["Weekly (7 days)", "Monthly (30 days)", "Yearly/Custom"])
if cycle_option == "Weekly (7 days)":
    selected_period = 7
elif cycle_option == "Monthly (30 days)":
    selected_period = 30
else:
    selected_period = st.sidebar.number_input("Custom Period (Days)", value=14, min_value=2)

st.sidebar.header("3. Financials")
unit_cost = st.sidebar.number_input("Cost per Unit (₹)", value=1000)
lead_time = st.sidebar.slider("Lead Time (Days)", 1, 30, 7)
service_level = st.sidebar.selectbox("Service Level", [0.90, 0.95, 0.99], index=1)
z_score = stats.norm.ppf(service_level)

# --- 2. Data Processing ---
if uploaded_file:
    try:
        df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        col = st.sidebar.selectbox("Select Demand Column", df_input.columns)
        data = pd.to_numeric(df_input[col], errors='coerce').dropna().values
    except Exception as e:
        st.error(f"Error: {e}")
        data = template_df["Demand"].values
else:
    data = template_df["Demand"].values

df = pd.DataFrame(data, columns=["Demand"])

# --- 3. Diagnostics & Tabs ---
tab1, tab2, tab3 = st.tabs(["🔍 Step 1: Health & Outliers", "🌊 Step 2: Seasonal Cash Unlock", "📅 Step 3: Order Playbook"])

# --- TAB 1: DIAGNOSTICS ---
with tab1:
    st.subheader("Data Health & Freak Event Detection")
    
    # Outlier Detection (IQR Method)
    Q1, Q3 = np.percentile(data, 25), np.percentile(data, 75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    outlier_indices = [i for i, x in enumerate(data) if x > upper_bound]
    
    # Seasonality Strength
    has_seasonality = False
    strength_pct = 0
    if len(data) >= selected_period * 2:
        decomp = seasonal_decompose(df['Demand'], model='additive', period=selected_period, extrapolate_trend='freq')
        df['Seasonal'], df['Trend'], df['Resid'] = decomp.seasonal, decomp.trend, decomp.resid
        resid_var = np.var(df['Resid'].dropna())
        total_var = np.var(df['Demand'] - df['Trend'])
        strength_pct = max(0, (1 - (resid_var / total_var)) * 100)
        has_seasonality = strength_pct > 25

    c1, c2, c3 = st.columns(3)
    shapiro_p = stats.shapiro(data)[1]
    c1.metric("Distribution", "Normal" if shapiro_p > 0.05 else "Irregular", f"p={shapiro_p:.3f}")
    c2.metric("Seasonality", f"{strength_pct:.1f}%", "Predictable" if has_seasonality else "Random")
    c3.metric("Freak Events (Outliers)", len(outlier_indices), "High Impact" if len(outlier_indices) > 0 else "None")

    # Time Series with Outliers
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=list(range(len(data))), y=data, mode='lines', name='Demand', line=dict(color='#3182CE')))
    fig_ts.add_trace(go.Scatter(x=outlier_indices, y=[data[i] for i in outlier_indices], 
                               mode='markers', name='Outliers', marker=dict(color='Red', size=10, symbol='x')))
    fig_ts.update_layout(title="Demand History (X marks the anomalies)", xaxis_title="Days", yaxis_title="Units")
    st.plotly_chart(fig_ts, use_container_width=True)

# --- TAB 2: CASH UNLOCK ---
with tab2:
    st.subheader("The 'Frozen Cash' Analysis")
    
    std_raw = np.std(data)
    std_resid = np.std(df['Resid'].dropna()) if has_seasonality else std_raw
    
    ss_static = round(z_score * std_raw * np.sqrt(lead_time))
    ss_optimized = round(z_score * std_resid * np.sqrt(lead_time))
    cash_saved = (ss_static - ss_optimized) * unit_cost

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Safety Stock Reduced By", f"{max(0, ss_static - ss_optimized)} units")
    col_m2.metric("Working Capital Unlocked", f"₹{max(0, cash_saved):,}", delta="Available Now", delta_color="normal")

    st.write("**Why this works:** Standard math treats your seasonal peaks as 'surprises'. We treat them as 'patterns', allowing you to hold less emergency stock.")
    
    fig_decomp = px.line(df, y=["Demand", "Trend"], title="Underlying Business Trend",
                        color_discrete_map={"Demand": "#CBD5E0", "Trend": "#3182CE"})
    st.plotly_chart(fig_decomp, use_container_width=True)

# --- TAB 3: PLAYBOOK ---
with tab3:
    st.subheader("Your Dynamic Order Schedule")
    
    mean_val = np.mean(data)
    rop_base = (mean_val * lead_time) + ss_optimized
    
    # Calculate Weekly Table
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if has_seasonality and selected_period == 7:
        seasonal_pattern = df['Seasonal'].iloc[:7].values
        rop_list = [round(rop_base + p) for p in seasonal_pattern]
    else:
        rop_list = [round(rop_base)] * 7

    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.write("**Reorder Points (Units)**")
        st.table(pd.DataFrame({"Day": days, "Reorder Level": rop_list}))
    
    with col_t2:
        fig_rop = px.line(x=days, y=rop_list, title="The ROP Wave: When to Refill", markers=True)
        fig_rop.update_traces(fill='tozeroy', line_color='#38A169')
        st.plotly_chart(fig_rop, use_container_width=True)

    st.info(f"**Inventory Strategy:** Keep your stock between {ss_optimized} and {max(rop_list)} units. Anything above this is 'Leaking Cash' from your business tank.")
