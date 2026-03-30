# --- 3. The "Dynamic Cash Unlock" Calculation ---
    mean_val = np.mean(data)
    std_resid = np.std(df['Resid']) if has_seasonality else np.std(data)
    
    # Base Safety Stock (The 'Noise' buffer)
    base_ss = z_score * std_resid * np.sqrt(lead_time)
    
    # Calculate Weekly Seasonal Indices (Day Avg / Total Avg)
    if has_seasonality:
        # Get the average seasonal effect for each day of the week
        seasonal_pattern = df['Seasonal'].iloc[:7].values
        # Index = (Mean + Seasonal Effect) / Mean
        seasonal_indices = (mean_val + seasonal_pattern) / mean_val
    else:
        seasonal_indices = np.ones(7)

    # Calculate the range of safety stock needed
    ss_min = round(base_ss * min(seasonal_indices))
    ss_max = round(base_ss * max(seasonal_indices))
    
    # --- 4. Dashboard Metrics ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Daily Demand", f"{mean_val:.1f} units")
    m2.metric("Safety Stock Range", f("{ss_min} - {ss_max} units"))
    
    # Calculate potential savings vs a "flat" high-stock strategy
    potential_savings = (ss_max - ss_min) * unit_cost
    m3.metric("Weekly Cash Flexibility", f"₹{potential_savings:,}", "Dynamic Shift")

    # --- 5. The 7-Day Inventory Playbook ---
    st.subheader("📅 Your 7-Day Inventory Playbook")
    st.markdown("Adjust your stock levels daily to keep cash flowing.")
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    schedule_data = []
    
    for i, day in enumerate(days):
        daily_ss = round(base_ss * seasonal_indices[i])
        daily_investment = daily_ss * unit_cost
        schedule_data.append({"Day": day, "Safety Stock (Units)": daily_ss, "Frozen Cash (₹)": f"₹{daily_investment:,}"})
    
    schedule_df = pd.DataFrame(schedule_data)
    
    # Display as a clean table
    st.table(schedule_df)

    # Add a line chart specifically for the Safety Stock "Wave"
    fig_ss_wave = px.line(schedule_df, x="Day", y="Safety Stock (Units)", 
                         title="The 'Cash Wave': Your Required Safety Stock by Day",
                         markers=True)
    fig_ss_wave.update_traces(line_color='#3182CE', fill='tozeroy') # Blue fill to show 'Frozen' area
    st.plotly_chart(fig_ss_wave, use_container_width=True)
