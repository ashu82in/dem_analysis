import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


# --- Page Config ---
st.set_page_config(page_title="Inventory Cash Unlocker", layout="wide")

st.title("📊 Inventory & Demand Pattern Analyzer")
st.markdown("""
This tool analyzes your demand patterns to find **'Frozen Cash'** in your inventory.
Stop borrowing from banks; start unlocking your own capital.
""")

# --- Sidebar: Data Input ---
st.sidebar.header("1. Input Historical Demand")
# Using your provided data as default
default_data = "12, 18, 16, 28, 12, 13, 9, 30, 50, 37, 29, 33, 21, 43, 5, 34, 20, 19, 0, 0, 1, 23"
user_input = st.sidebar.text_area("Enter daily demand (comma separated):", default_data)

# Sidebar: Financial Assumptions
st.sidebar.header("2. Financial Constants")
unit_cost = st.sidebar.number_input("Cost per Unit (₹)", value=1000)
lead_time = st.sidebar.slider("Lead Time (Days to Restock)", 1, 30, 7)

# --- Data Processing ---
try:
    data = [float(x.strip()) for x in user_input.split(",")]
    df = pd.DataFrame(data, columns=["Demand"])
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    # --- Metrics Row ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Daily Demand", f"{mean_val:.2f} units")
    col2.metric("Volatility (Std Dev)", f"{std_val:.2f}")
    
    # Safety Stock Calculation (95% Service Level)
    safety_stock = round(1.65 * std_val * np.sqrt(lead_time))
    col3.metric("Required Safety Stock", f"{safety_stock} units")
    
    blocked_cash = safety_stock * unit_cost
    col4.metric("Capital Tied in Safety Stock", f"₹{blocked_cash:,}")

    # --- Distribution Analysis ---
    st.subheader("📈 Demand Pattern Analysis")
    c1, c2 = st.columns(2)
    
    with c1:
        # Histogram
        fig_hist = px.histogram(df, x="Demand", nbins=10, title="Demand Distribution", 
                               color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with c2:
        # Check for Normality
        shapiro_p = stats.shapiro(data)[1]
        if shapiro_p > 0.05:
            st.success(f"✅ Data is Normally Distributed (p={shapiro_p:.2f}). Statistical simulations are highly reliable.")
        else:
            st.warning("⚠️ Data pattern is irregular. Using non-parametric simulation.")

    # --- Monte Carlo Simulation ---
    st.divider()
    st.subheader("🔮 100-Day Demand Simulation")
    
    # Simulate based on analyzed pattern
    sim_days = 100
    simulation = np.random.normal(mean_val, std_val, sim_days)
    simulation = np.maximum(0, np.round(simulation))
    
    sim_df = pd.DataFrame({"Day": range(1, sim_days+1), "Simulated_Demand": simulation})
    
    fig_sim = px.line(sim_df, x="Day", y="Simulated_Demand", title="Simulated Future Demand (Monte Carlo)")
    fig_sim.add_hline(y=mean_val, line_dash="dash", line_color="red", annotation_text="Average")
    st.plotly_chart(fig_sim, use_container_width=True)

    # --- Final Insight ---
    st.info(f"**Actionable Insight:** Your peak simulated demand is **{int(max(simulation))} units**. If you are currently holding more than **{int(mean_val * lead_time + safety_stock)} units** in stock, you are likely sitting on excess cash that could be used for growth.")

except Exception as e:
    st.error(f"Please check your data format. Error: {e}")
