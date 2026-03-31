import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# --- Page Config ---
st.set_page_config(page_title="Supply Chain DNA Pro", layout="wide")

st.title("🎯 The Golden Window: Distribution & Optimization")
st.markdown("""
Use this tool to find the 'Scientific Best' logistics cycle. 
**Note:** If demand is lumpy, toggle 'Ignore Zeros' to see the true shape of your spikes.
""")

# --- 1. Sidebar: Data & Search Range ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.header("2. Optimization Settings")
max_test_window = st.sidebar.slider("Max Window to Test (Days)", 7, 30, 30)
p_threshold = 0.05

st.sidebar.header("3. Distributor Settings")
ignore_zeros = st.sidebar.toggle("Ignore Zeros for Normality", value=True, 
                                 help="Filters out 0-demand buckets to check if the 'Spikes' are normal.")

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
        df_raw = df_raw.dropna(subset=['Date', target_col])
        df_daily = df_raw.groupby('Date')[target_col].sum().sort_index()
        full_range = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
        data_series_daily = df_daily.reindex(full_range, fill_value=0)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    t = pd.date_range(start="2025-01-01", periods=300)
    lumpy = [np.random.randint(1000, 5000) if np.random.random() > 0.8 else 0 for _ in range(300)]
    data_series_daily = pd.Series(lumpy, index=t)
    st.info("💡 Using sample data. Upload your file to run optimization.")

# --- 2. OPTIMIZATION LOOP ---
results = []
for w in range(1, max_test_window + 1):
    for o in range(w):
        temp_start = data_series_daily.index.min() + pd.Timedelta(days=o)
        temp_data = data_series_daily[data_series_daily.index >= temp_start]
        clubbed = temp_data.resample(f'{w}D').sum()
        
        # Apply Zero-Filter if toggled for the loop
        test_data = clubbed[clubbed > 0] if ignore_zeros else clubbed
        
        if len(test_data) >= 3 and test_data.std() > 0:
            _, p = stats.shapiro(test_data)
            results.append({'Window': w, 'Offset': o, 'p_value': p})

df_opt = pd.DataFrame(results)

# --- 3. RESULTS & LEADERBOARD ---
st.subheader("🏆 Golden Scenarios (p-value Rank)")
df_leaderboard = df_opt[df_opt['p_value'] > p_threshold].sort_values(by='p_value', ascending=False)

if not df_leaderboard.empty:
    st.dataframe(df_leaderboard.head(10), use_container_width=True, hide_index=True)
else:
    st.warning("No Normal scenarios found. Demand is highly erratic.")

# --- 4. MANUAL SIMULATOR ---
st.divider()
st.subheader("🕹️ Manual Scenario Explorer")

col_s1, col_s2 = st.columns(2)
with col_s1:
    sel_w = st.slider("Select Window Size", 1, max_test_window, 7)
with col_s2:
    # --- ERROR FIX: Handle range(0,0) ---
    if sel_w > 1:
        sel_o = st.slider("Select Start Offset", 0, sel_w-1, 0)
    else:
        st.write("Offset not applicable for Window Size 1")
        sel_o = 0

# Final Bucket Processing
final_start = data_series_daily.index.min() + pd.Timedelta(days=sel_o)
final_series = data_series_daily[data_series_daily.index >= final_start].resample(f"{sel_w}D").sum()

# Display Visuals
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Bucket Timeline**")
    st.plotly_chart(px.bar(final_series), use_container_width=True)
with c2:
    # Analysis logic for Histogram (Filter zeros if toggled)
    display_series = final_series[final_series > 0] if ignore_zeros else final_series
    st.markdown("**Bucket Distribution (Histogram)**")
    st.plotly_chart(px.histogram(display_series, nbins=15, color_discrete_sequence=['#38A169']), use_container_width=True)

# --- 5. DATA TABLE VIEW ---
with st.expander("📄 View Bucket Data (Raw numbers for Histogram)"):
    df_display = final_series.reset_index()
    df_display.columns = ['Period Start Date', 'Total Demand Volume']
    if ignore_zeros:
        st.write("Showing only Non-Zero demand events:")
        st.dataframe(df_display[df_display['Total Demand Volume'] > 0], use_container_width=True)
    else:
        st.dataframe(df_display, use_container_width=True)

# Final Stats Verdict
if len(display_series) >= 3:
    _, final_p = stats.shapiro(display_series)
    if final_p > 0.05:
        st.success(f"✅ Normal Distribution (p={final_p:.4f}). This scenario is predictable.")
    else:
        st.warning(f"⚠️ Non-Normal (p={final_p:.4f}). High risk of erratic spikes.")
