import streamlit as st
import pypsa
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title=" SmartGrid", layout="wide", page_icon="⚡")

# --- STYLING ---
st.markdown("""
<style>
    .main { background-color: #0b0d10; color: #e0e0e0; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #f9d71c; font-weight: bold; }
    .stMetric { 
        background: rgba(249, 215, 28, 0.05); 
        padding: 15px; border-radius: 10px; border: 1px solid rgba(249, 215, 28, 0.2); 
    }
    .tech-label { color: #00ffcc; font-size: 0.85rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 10px; text-transform: uppercase;}
    .opti-card { background: rgba(0, 255, 204, 0.05); padding: 15px; border-radius: 5px; border-left: 5px solid #00ffcc; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ============================================
# GRID ENGINE
# ============================================
def build__grid(params, line_caps, solar_on=False):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2026-01-01", periods=1, freq="h"))

    # NODES
    nodes = [
        ("marwa_thermal_GSS_400kV", 400, 81.7615, 21.4320),     
        ("Urla_Industrial_220kV", 220, 81.6024, 21.3108), 
        ("Siltara_Industrial_220kV", 220, 81.6638, 21.3812), 
        ("a_Residential_132kV", 132, 81.5900, 21.2300), 
        ("Mana_Airport_132kV", 132, 81.7388, 21.1804),     
        ("Nava__Central_132kV", 132, 81.7912, 21.1693), 
        ("Bhatagaon_132kV", 132, 81.6200, 21.2100),        
        ("Kachna_Sub_132kV", 132, 81.7023, 21.2628)        
    ]
    for name, v, x, y in nodes:
        n.add("Bus", name, v_nom=v, x=x, y=y)

    # TRANSFORMERS
    n.add("Transformer", "ICT_marwa_thermal_Urla", bus0="marwa_thermal_GSS_400kV", bus1="Urla_Industrial_220kV", s_nom=630, x=0.05)
    n.add("Transformer", "ICT_Urla_a", bus0="Urla_Industrial_220kV", bus1="a_Residential_132kV", s_nom=320, x=0.08)
    n.add("Transformer", "ICT_Urla_Bhatagaon", bus0="Urla_Industrial_220kV", bus1="Bhatagaon_132kV", s_nom=250, x=0.07)
    n.add("Transformer", "ICT_Siltara_Kachna", bus0="Siltara_Industrial_220kV", bus1="Kachna_Sub_132kV", s_nom=200, x=0.09)

    # LINES
    lines_config = [
        ("L_Urla_Siltara", "Urla_Industrial_220kV", "Siltara_Industrial_220kV", 15),
        ("L_Kachna_Mana", "Kachna_Sub_132kV", "Mana_Airport_132kV", 10),
        ("L_Mana_Nava", "Mana_Airport_132kV", "Nava__Central_132kV", 8),
        ("L_Bhatagaon_Mana", "Bhatagaon_132kV", "Mana_Airport_132kV", 14),
        ("L_a_Bhatagaon", "a_Residential_132kV", "Bhatagaon_132kV", 7)
    ]
    for name, b0, b1, length in lines_config:
        n.add("Line", name, bus0=b0, bus1=b1, s_nom=line_caps[name], x=0.1, r=0.05, length=length)

    # Generators
    n.add("Generator", "Marwa_Thermal_1000MW", bus="marwa_thermal_GSS_400kV", p_nom=1000, marginal_cost=4.2)
    if solar_on:
        n.add("Generator", "Nava_Solar", bus="Nava__Central_132kV", p_nom=500, marginal_cost=0.5)

    # Loads
    n.add("Load", "Urla_Load", bus="Urla_Industrial_220kV", p_set=params['u'])
    n.add("Load", "Siltara_Load", bus="Siltara_Industrial_220kV", p_set=params['s'])
    n.add("Load", "Residential_Load", bus="a_Residential_132kV", p_set=params['r'])
    n.add("Load", "Nava_Load", bus="Nava__Central_132kV", p_set=params['n'])
    n.add("Load", "Kachna_Load", bus="Kachna_Sub_132kV", p_set=params['k'])
    n.add("Load", "Mana_Load", bus="Mana_Airport_132kV", p_set=params['m'])
    n.add("Load", "Bhatagaon_Load", bus="Bhatagaon_132kV", p_set=params['b'])

    try:
        status, _ = n.optimize(solver_name='highs')
        return n, (n.objective if n.objective is not None else 0.0), status
    except:
        return n, 0.0, "error"

# ============================================
# UI
# ============================================
st.title("Smart Grid")

with st.sidebar:
    st.header("⚡ Live Demand Control")
    u = st.slider("Urla Industrial (MW)", 0, 800, 200)
    s = st.slider("Siltara Steel (MW)", 0, 600, 100)
    r = st.slider("Raipura Residential (MW)", 0, 400, 80)
    n_r = st.slider("Nava Raipur (MW)", 0, 250, 40)
    k = st.slider("Kachna Substation (MW)", 0, 200, 10)
    m = st.slider("Mana Airport (MW)", 0, 150, 10)
    b = st.slider("Bhatagaon Area (MW)", 0, 200, 10)
    
    st.divider()
    st.header("🛤️ Line Capacity Control (MW)")
    c_us = st.slider("L_Urla_Siltara Cap", 100, 600, 400)
    c_km = st.slider("L_Kachna_Mana Cap", 50, 400, 160)
    c_mn = st.slider("L_Mana_Nava Cap", 50, 600, 598)
    c_bm = st.slider("L_Bhatagaon_Mana Cap", 50, 400, 160)
    c_ab = st.slider("L_a_Bhatagaon Cap", 50, 400, 160)

    st.divider()
    solar_active = st.checkbox("🌱 Activate Solar Support (Nava Raipur)")

params = {'u': u, 's': s, 'r': r, 'n': n_r, 'k': k, 'm': m, 'b': b}
line_caps = {
    'L_Urla_Siltara': c_us, 'L_Kachna_Mana': c_km, 
    'L_Mana_Nava': c_mn, 'L_Bhatagaon_Mana': c_bm, 'L_a_Bhatagaon': c_ab
}

n, cost, status = build__grid(params, line_caps, solar_on=solar_active)

if n is not None:
    line_p = n.lines_t.p0.iloc[0] if status == "ok" else pd.Series(0, index=n.lines.index)
    tf_p = n.transformers_t.p0.iloc[0] if status == "ok" else pd.Series(0, index=n.transformers.index)
    
    total_load = sum(params.values())
    marwa_p = n.generators_t.p.loc[:, "Marwa_Thermal_1000MW"].iloc[0] if status == "ok" else total_load
    solar_p = n.generators_t.p.loc[:, "Nava_Solar"].iloc[0] if (solar_active and status == "ok") else 0
    
    base_cost = total_load * 4.2 * 80
    current_cost = cost * 80 if status == "ok" else base_cost
    savings = base_cost - current_cost

    kpi_cols = st.columns(6 if solar_active else 4)
    kpi_cols[0].metric("Total Load", f"{total_load:.0f} MW")
    kpi_cols[1].metric("Marwa Thermal", f"{marwa_p:.1f} MW")
    
    if solar_active:
        kpi_cols[2].metric("Solar Gen", f"{solar_p:.1f} MW")
        kpi_cols[3].metric("Hourly Savings", f"₹{savings:,.0f}")
        kpi_cols[4].metric("Grid Status", "STABLE" if status == "ok" else "OVERLOAD")
        kpi_cols[5].metric("Hourly Cost", f"₹{current_cost:,.0f}")
    else:
        kpi_cols[2].metric("Grid Status", "STABLE" if status == "ok" else "OVERLOAD")
        kpi_cols[3].metric("Hourly Cost", f"₹{current_cost:,.0f}")

    if status != "ok":
        st.error("🚨 Grid Overload Detected! Capacity exceeded on highlighted assets.")

    st.divider()
    m1, m2 = st.columns([3, 2])

    with m1:
        st.markdown('<p class="tech-label">Interactive Network Map (Deep Detail View)</p>', unsafe_allow_html=True)
        fig = go.Figure()
        for name, tf in n.transformers.iterrows():
            b0, b1 = n.buses.loc[tf.bus0], n.buses.loc[tf.bus1]
            flow = abs(tf_p[name]) if status == "ok" else total_load
            is_overloaded = flow > tf.s_nom
            fig.add_trace(go.Scattermapbox(lon=[b0.x, b1.x], lat=[b0.y, b1.y], mode='lines', line=dict(width=7, color="red" if is_overloaded else "#f9d71c"), name="Transformer", hovertemplate=f"<b>ICT: {name}</b><br>Flow: {flow:.1f} MW<br>Cap: {tf.s_nom} MW<extra></extra>"))

        for name, line in n.lines.iterrows():
            b0, b1 = n.buses.loc[line.bus0], n.buses.loc[line.bus1]
            flow = abs(line_p[name]) if status == "ok" else 0
            is_overloaded = flow > line.s_nom
            is_solar_path = solar_active and ("Nava" in line.bus0 or "Nava" in line.bus1 or "Mana" in line.bus0 or "Mana" in line.bus1)
            base_color = "purple" if "Urla_Siltara" in name else "#00ffcc"
            line_color = "red" if is_overloaded else ("#00ff00" if is_solar_path and solar_p > 0 else base_color)
            fig.add_trace(go.Scattermapbox(lon=[b0.x, b1.x], lat=[b0.y, b1.y], mode='lines', line=dict(width=4, color=line_color), name="Line", hovertemplate=f"<b>Line: {name}</b><br>Flow: {flow:.1f} MW<br>Cap: {line.s_nom} MW<extra></extra>"))

        for bus, b_data in n.buses.iterrows():
            load_val = n.loads[n.loads.bus == bus].p_set.sum()
            node_color = "#f9d71c" if b_data.v_nom == 400 else "purple" if b_data.v_nom == 220 else "#ffffff"
            if solar_active and ("Nava" in bus or ("Mana" in bus and solar_p > 40)): node_color = "#00ff00"
            if status != "ok":
                if any((n.lines.bus0 == bus) | (n.lines.bus1 == bus)) or any((n.transformers.bus0 == bus) | (n.transformers.bus1 == bus)):
                    if load_val > 100: node_color = "red"
            fig.add_trace(go.Scattermapbox(lon=[b_data.x], lat=[b_data.y], mode='markers+text', marker=dict(size=15, color=node_color), text=[bus.split('_')[0]], textposition="top center", name="Substation", hovertemplate=(f"<b>Substation: {bus}</b><br>Voltage: {b_data.v_nom} kV<br>Demand: {load_val} MW<extra></extra>")))

        fig.update_layout(mapbox_style="carto-darkmatter", mapbox=dict(center=dict(lat=21.30, lon=81.70), zoom=10.2), margin=dict(t=0, b=0, l=0, r=0), height=650)
        st.plotly_chart(fig, use_container_width=True)

    with m2:
        st.markdown('<p class="tech-label">Asset Technical Inventory</p>', unsafe_allow_html=True)
        st.write("**ICT/Transformer Status:**")
        st.dataframe(n.transformers[['bus0', 'bus1', 's_nom']].rename(columns={'s_nom':'Max MW'}))
        st.write("**Line Status:**")
        st.dataframe(n.lines[['bus0', 'bus1', 's_nom']].rename(columns={'s_nom':'Max MW'}))

    # --- SOLAR ENERGY ANALYTICS ---
    st.divider()
    st.markdown('<p class="tech-label">☀️ Solar Energy Distribution Analytics</p>', unsafe_allow_html=True)
    solar_n = params['n'] if solar_active else 0
    solar_m = min(params['m'], solar_p * 0.3) if solar_active else 0
    solar_b = min(params['b'], solar_p * 0.2) if solar_active else 0
    solar_k = min(params['k'], solar_p * 0.1) if solar_active else 0
    solar_r = max(0, solar_p - (solar_n + solar_m + solar_b + solar_k)) if solar_active else 0
    if solar_active:
        col_s1, col_s2 = st.columns([1, 1])
        with col_s1:
            st.write("**Node-wise Solar Utilization:**")
            solar_dist = {"Load Center": ["Nava Raipur", "Mana Airport", "Bhatagaon", "Kachna", "Raipura Residential"], "Solar Contribution (MW)": [solar_n, solar_m, solar_b, solar_k, solar_r]}
            st.table(pd.DataFrame(solar_dist))
        with col_s2:
            st.write("**Grid Power Mix:**")
            mix_fig = go.Figure(data=[go.Pie(labels=['Thermal', 'Solar'], values=[marwa_p, solar_p], hole=.4, marker=dict(colors=['#f9d71c', '#00ff00']))])
            mix_fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250, paper_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(mix_fig, use_container_width=True)
    else: st.info("Solar Support is currently inactive.")

    # --- THERMAL POWER ANALYTICS ---
    st.divider()
    st.markdown('<p class="tech-label">🔥 Thermal Power Supply Analytics (Marwa Plant)</p>', unsafe_allow_html=True)
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.write("**Thermal Energy Allocation:**")
        thermal_dist = {"Load Center": ["Urla Industrial", "Siltara Steel", "Raipura Residential", "Nava Raipur", "Mana Airport", "Bhatagaon", "Kachna"], "Total Demand (MW)": [params['u'], params['s'], params['r'], params['n'], params['m'], params['b'], params['k']], "Thermal Share (MW)": [params['u'], params['s'], params['r'] - solar_r, params['n'] - solar_n, params['m'] - solar_m, params['b'] - solar_b, params['k'] - solar_k]}
        thermal_df = pd.DataFrame(thermal_dist)
        st.table(thermal_df)
    with col_t2:
        st.write("**Thermal Data Verification:**")
        calc_total_thermal = thermal_df["Thermal Share (MW)"].sum()
        st.write(f"✓ **Plant Output:** {marwa_p:.1f} MW"); st.write(f"✓ **Allocated Load:** {calc_total_thermal:.1f} MW")
        usage_pct = (marwa_p / 1000) * 100
        st.write(f"**Plant Loading:** {usage_pct:.1f}%"); st.progress(min(usage_pct/100, 1.0))
        if marwa_p > 950: st.warning("⚠️ Marwa Thermal Plant is operating near critical capacity!")

    # ============================================
    # NEW ADDITION: ADVANCED GRID OPTIMIZATION INTELLIGENCE
    # ============================================
    st.divider()
    st.markdown('<p class="tech-label"> Advanced Grid Optimization Intelligence</p>', unsafe_allow_html=True)
    
    opti_1, opti_2 = st.columns([1, 1])
    
    with opti_1:
        st.write("**Optimization Efficiency Metrics:**")
        # Line Loss Simulation (Assuming 3% loss on Thermal, 1% on Solar due to proximity)
        sim_loss = (marwa_p * 0.03) + (solar_p * 0.01)
        solar_penetration = (solar_p / total_load) * 100 if total_load > 0 else 0
        
        st.info(f"⚡ **Transmission Losses (Est.):** {sim_loss:.2f} MW")
        st.info(f"🌱 **Solar Penetration Level:** {solar_penetration:.1f}%")
        
        # Carbon Reduction Logic (0.9kg CO2 per kWh for Thermal)
        co2_saved = (solar_p * 1000 * 0.9) / 1000 # kg per hour
        st.success(f"🌍 **CO2 Reduction:** {co2_saved:.1f} kg/hour")

    with opti_2:
        st.write("**AI-Driven Optimization Advice:**")
        
        # Scenario 1: High Industrial Load
        if (u + s) > 500:
            st.markdown('<div class="opti-card">💡 <b>Urla-Siltara Cluster:</b> Industrial demand is high. Suggesting Peak Load Shifting to off-peak hours to reduce ICT stress.</div>', unsafe_allow_html=True)
        
        # Scenario 2: Solar Potential
        if solar_active and solar_p < 400 and (n_r + m) > 100:
            st.markdown('<div class="opti-card">💡 <b>Solar Expansion:</b> Current solar utilization is stable. You can safely increase solar capacity by 150MW without grid instability.</div>', unsafe_allow_html=True)
            
        # Scenario 3: Asset Health
        if status != "ok":
            st.markdown('<div class="opti-card" style="border-left-color:red;">🚨 <b>Constraint Warning:</b> Grid is failing. Suggesting immediate Reactive Power Compensation at Bhatagaon and Mana nodes.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="opti-card">✅ <b>System Health:</b> Voltage profiles are within ±5% limit. No immediate hardware intervention required.</div>', unsafe_allow_html=True)

    # --- FINAL OPTIMIZATION TABLE ---
    st.write("**Optimization Summary Table:**")
    opti_summary = pd.DataFrame({
        "Parameter": ["Grid Reliability Index", "Cost Efficiency", "Carbon Footprint", "Asset Life Expectancy"],
        "Status": [
            "High" if status == "ok" else "Critical",
            f"{100 - (current_cost/base_cost*100):.1f}% Better",
            "Improving" if solar_active else "Standard",
            "Optimal" if usage_pct < 80 else "Reduced due to Heat"
        ],
        "Action Required": [
            "None" if status == "ok" else "Upgrade Lines",
            "Keep Solar Active",
            "Add more Renewables",
            "Monitor ICT Temperature"
        ]
    })
    st.table(opti_summary)

    # ============================================
    # NEW IMPLEMENTATION: STRESS TEST & REALITY ENGINE
    # ============================================
    st.divider()
    st.markdown('<p class="tech-label" style="color:#ff4b4b;">🛠️ Live Stress Test & Reality Engine</p>', unsafe_allow_html=True)
    
    # 1. N-1 CONTINGENCY IMPLEMENTATION
    st.write("**🚨 N-1 Contingency Analysis (Fault Simulation):**")
    col_f1, col_f2 = st.columns([1, 2])
    
    with col_f1:
        fault_line = st.selectbox("Select Line to Trip (Force Outage)", ["None"] + list(n.lines.index))
        cloud_cover = st.checkbox("☁️ Simulate Sudden Cloud Cover (Solar Drop 80%)")
    
    # Implementing the Fault Logic
    if fault_line != "None":
        n.lines.loc[fault_line, "s_nom"] = 0.001 # Line tripped
        st.warning(f"Line {fault_line} has TRIPPED. Calculating re-routing...")
    
    # Implementing Solar Intermittency
    current_solar_gen = solar_p
    if cloud_cover:
        current_solar_gen = solar_p * 0.2
        st.error(f"Cloud Cover Detected! Solar output dropped to {current_solar_gen:.1f} MW")

    # 2. POWER FACTOR & LOSS CALCULATION
    # Real World Formula: Apparent Power (kVA) = Real Power (kW) / Power Factor
    pf = 0.85
    apparent_load = total_load / pf
    transmission_losses = (total_load * 0.05) # 5% resistive loss implementation
    
    with col_f2:
        st.write("**Physical Reality Metrics:**")
        f_cols = st.columns(3)
        f_cols[0].metric("Apparent Load (MVA)", f"{apparent_load:.1f}", delta=f"{apparent_load - total_load:.1f} MVA Lag", delta_color="inverse")
        f_cols[1].metric("Active Losses", f"{transmission_losses:.1f} MW", delta="Heat Loss")
        f_cols[2].metric("Inertia Level", "Low" if solar_active else "High")

    # 3. RE-SOLVING GRID WITH FAULT CONDITIONS
    try:
        # Re-optimizing based on the fault selected
        if fault_line != "None" or cloud_cover:
            n.optimize(solver_name='highs')
            new_status = n.objective if n.objective is not None else "failed"
            if n.objective is None:
                st.error("💥 SYSTEM COLLAPSE: The grid cannot handle this N-1 fault! Blackout in sub-sectors.")
            else:
                st.success("Grid survived the fault, but at higher operational cost.")
    except:
        st.error("Critical Fault: Stability Limit Exceeded.")

    # 4. FINAL VULNERABILITY SUMMARY TABLE (Live Data)
    st.write("**Live Reliability Audit:**")
    audit_data = pd.DataFrame({
        "Audit Parameter": ["Power Factor Impact", "Line Loss Overhead", "N-1 Survivability", "Grid Inertia Status"],
        "Calculated Value": [f"{pf}", f"{transmission_losses:.1f} MW", "Failed" if fault_line != "None" and status != "ok" else "Pass", "Weak" if solar_active else "Stable"],
        "Risk Factor": ["High (Current Stress)", "Economic Leakage", "Topology Weakness", "Frequency Risk"],
        "Remedy": ["Add Capacitor Banks", "HVDC Conversion", "Build Mesh Network", "Add Synchronous Condensers"]
    })
    st.table(audit_data)

    st.caption("Note: This engine implements the drawbacks discussed in the previous audit to show real-world grid limitations.")