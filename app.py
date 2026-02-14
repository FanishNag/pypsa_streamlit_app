import streamlit as st
import pypsa
import pandas as pd
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Power Grid Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PREMIUM STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0e1117;
        color: #ffffff;
    }

    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }

    .stMetric:hover {
        transform: translateY(-5px);
        border-color: #00ffcc;
    }

    h1 {
        margin-top: 0rem !important;
        margin-bottom: 0.5rem !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }

    h1, h2, h3 {
        background: linear-gradient(90deg, #00ffcc, #0099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    .status-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #00ffcc;
        background: rgba(0, 255, 204, 0.1);
    }
</style>
""", unsafe_allow_html=True)

def run_simulation(demand_a, demand_b, demand_c, demand_d, demand_solar, demand_wind):
    # 1. Create the grid
    n = pypsa.Network()

    # 2. Add Buses (electrical nodes/junctions in the network)
    # Parameters:
    #   - v_nom: Nominal voltage in kV (kilovolts) - the standard operating voltage
    #   - x, y: Coordinates for visualization (arbitrary units, used for plotting)
    n.add("Bus", "City_A", v_nom=380, x=0, y=0)
    n.add("Bus", "City_B", v_nom=380, x=4, y=0)
    n.add("Bus", "City_C", v_nom=380, x=4, y=4)
    n.add("Bus", "City_D", v_nom=380, x=0, y=4)
    n.add("Bus", "Solar_Farm", v_nom=380, x=2, y=1)
    n.add("Bus", "Wind_Farm", v_nom=380, x=2, y=3)

    # 3. Add Lines (transmission lines connecting buses)
    # Parameters:
    #   - bus0, bus1: Names of the buses this line connects
    #   - x: Series reactance (per unit) - electrical impedance that opposes AC current flow
    #        Higher x = more resistance to power flow, causes voltage drops
    #   - r: Series resistance (per unit, optional) - resistive losses in the line
    #   - s_nom: Nominal apparent power capacity in MW - maximum power the line can carry
    #   - length: Physical length in km (optional) - used for loss calculations
    n.add("Line", "Line_A_B", bus0="City_A", bus1="City_B", x=0.1, s_nom=1000)
    n.add("Line", "Line_B_C", bus0="City_B", bus1="City_C", x=0.1, s_nom=1000)
    n.add("Line", "Line_C_D", bus0="City_C", bus1="City_D", x=0.1, s_nom=1000)
    n.add("Line", "Line_D_A", bus0="City_D", bus1="City_A", x=0.1, s_nom=1000)
    n.add("Line", "Line_S_A", bus0="Solar_Farm", bus1="City_A", x=0.1, s_nom=1000)
    n.add("Line", "Line_S_B", bus0="Solar_Farm", bus1="City_B", x=0.1, s_nom=1000)
    n.add("Line", "Line_W_C", bus0="Wind_Farm", bus1="City_C", x=0.1, s_nom=1000)
    n.add("Line", "Line_W_D", bus0="Wind_Farm", bus1="City_D", x=0.1, s_nom=1000)

    # 4. Add Generators (power generation plants)
    # Parameters:
    #   - bus: Name of the bus where this generator is connected
    #   - p_nom: Nominal power capacity in MW - maximum power this generator can produce
    #   - marginal_cost: Cost per MWh to generate power ($/MWh or ₹/MWh)
    #        Lower cost generators are dispatched first by the optimizer
    #        Typical order: Solar/Wind (0-10) < Coal (20-40) < Gas (40-80)
    #   - carrier: Type of fuel/technology (optional, e.g., 'solar', 'coal', 'gas')
    n.add("Generator", "Gas_Plant", bus="City_A", p_nom=500, marginal_cost=50) 
    n.add("Generator", "Coal_Plant", bus="City_B", p_nom=700, marginal_cost=30)
    n.add("Generator", "Solar_Plant", bus="Solar_Farm", p_nom=1000, marginal_cost=5)
    n.add("Generator", "Wind_Plant", bus="Wind_Farm", p_nom=800, marginal_cost=8)

    # 5. Add Loads (electricity demand/consumption points)
    # Parameters:
    #   - bus: Name of the bus where this load is connected
    #   - p_set: Power demand in MW - how much power this location needs
    #        This is the target consumption that must be met by generators
    n.add("Load", "City_A_Demand", bus="City_A", p_set=demand_a)
    n.add("Load", "City_B_Demand", bus="City_B", p_set=demand_b)
    n.add("Load", "City_C_Demand", bus="City_C", p_set=demand_c)
    n.add("Load", "City_D_Demand", bus="City_D", p_set=demand_d)
    n.add("Load", "Solar_Farm_Demand", bus="Solar_Farm", p_set=demand_solar)
    n.add("Load", "Wind_Farm_Demand", bus="Wind_Farm", p_set=demand_wind)

    # 6. Run the Optimization
    # The optimizer finds the least-cost way to:
    #   - Dispatch generators (decide how much each plant produces)
    #   - Route power through transmission lines
    #   - Meet all load demands while respecting:
    #       * Generator capacity limits (p_nom)
    #       * Line capacity limits (s_nom)
    #       * Physical power flow laws (Kirchhoff's laws)
    # Solver: 'highs' is an open-source linear programming solver
    try:
        n.optimize(solver_name='highs')
    except Exception as e:
        st.error(f"Solver Error: {e}. Please ensure 'highs' solver is installed.")
        return None

    return n

st.title("⚡ Power Grid Optimizer")
# st.markdown("### Interactive PyPSA Simulation Dashboard")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Scenario Controls")
st.sidebar.markdown("Adjust city demands to see how the grid reacts.")

d_a = st.sidebar.slider("City A Demand (MW)", 0, 1500, 800, step=50)
d_b = st.sidebar.slider("City B Demand (MW)", 0, 1500, 1000, step=50)
d_c = st.sidebar.slider("City C Demand (MW)", 0, 1500, 1200, step=50)
d_d = st.sidebar.slider("City D Demand (MW)", 0, 1500, 0, step=50)
d_solar = st.sidebar.slider("Solar Farm Demand (MW)", 0, 1500, 0, step=50)
d_wind = st.sidebar.slider("Wind Farm Demand (MW)", 0, 1500, 0, step=50)

with st.spinner("Running Power Flow Optimization..."):
    network = run_simulation(d_a, d_b, d_c, d_d, d_solar, d_wind)

if network and network.objective is not None:
    # --- RESULTS SECTION ---
    st.markdown("---")

    # Row 1: Generation Analytics WITH MARGINAL COST
    st.markdown("#### ⚡ Generation Capacity & Costs")
    gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)
    gcol1.metric("Total Capacity", f"{network.generators.p_nom.sum():.0f} MW", delta="System Generation", delta_color="off")
    
    gcol2.metric("Solar Plant", f"{network.generators.at['Solar_Plant', 'p_nom']:.0f} MW", 
                delta=f"${network.generators.at['Solar_Plant', 'marginal_cost']}/MWh", delta_color="normal")
    gcol3.metric("Wind Plant", f"{network.generators.at['Wind_Plant', 'p_nom']:.0f} MW", 
                delta=f"${network.generators.at['Wind_Plant', 'marginal_cost']}/MWh", delta_color="normal")
    gcol4.metric("Coal Plant", f"{network.generators.at['Coal_Plant', 'p_nom']:.0f} MW", 
                delta=f"${network.generators.at['Coal_Plant', 'marginal_cost']}/MWh", delta_color="off")
    gcol5.metric("Gas Plant", f"{network.generators.at['Gas_Plant', 'p_nom']:.0f} MW", 
                delta=f"${network.generators.at['Gas_Plant', 'marginal_cost']}/MWh", delta_color="inverse")

    # Row 2: Demand Analytics
    st.markdown("#### 💡 Demand Breakdown")
    dcol1, dcol2, dcol3, dcol4, dcol5, dcol6 = st.columns(6)
    
    total_demand = network.loads.p_set.sum()
    dcol1.metric("Total Demand", f"{total_demand:.0f} MW", delta="System Load", delta_color="off")
    
    dcol2.metric("City A", f"{network.loads.at['City_A_Demand', 'p_set']:.0f} MW")
    dcol3.metric("City B", f"{network.loads.at['City_B_Demand', 'p_set']:.0f} MW")
    dcol4.metric("City C", f"{network.loads.at['City_C_Demand', 'p_set']:.0f} MW")
    dcol5.metric("City D", f"{network.loads.at['City_D_Demand', 'p_set']:.0f} MW")
    dcol6.metric("Renewables", f"{network.loads.at['Solar_Farm_Demand', 'p_set'] + network.loads.at['Wind_Farm_Demand', 'p_set']:.0f} MW")

    st.markdown("---")
    
    # Row 3: MERGED Dispatch + Individual Costs
    st.markdown("#### 💰 Dispatch & Generation Economics")
    gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)

    # Calculate dispatch and individual costs
    solar_p = network.generators_t.p['Solar_Plant'].iloc[0]
    solar_cost = solar_p * network.generators.at['Solar_Plant', 'marginal_cost']
    wind_p = network.generators_t.p['Wind_Plant'].iloc[0]
    wind_cost = wind_p * network.generators.at['Wind_Plant', 'marginal_cost']
    coal_p = network.generators_t.p['Coal_Plant'].iloc[0]
    coal_cost = coal_p * network.generators.at['Coal_Plant', 'marginal_cost']
    gas_p = network.generators_t.p['Gas_Plant'].iloc[0]
    gas_cost = gas_p * network.generators.at['Gas_Plant', 'marginal_cost']
    total_gen_cost = solar_cost + wind_cost + coal_cost + gas_cost

    gcol1.metric("Total Cost", f"${network.objective:,.0f}", delta=f"({total_gen_cost:,.0f})", delta_color="inverse")

    gcol2.metric("Solar", f"{solar_p:.0f} MW", 
                delta=f"${solar_cost:,.0f}", delta_color="normal")
    gcol3.metric("Wind", f"{wind_p:.0f} MW", 
                delta=f"${wind_cost:,.0f}", delta_color="normal")
    gcol4.metric("Coal", f"{coal_p:.0f} MW", 
                delta=f"${coal_cost:,.0f}", delta_color="off")
    gcol5.metric("Gas", f"{gas_p:.0f} MW", 
                delta=f"${gas_cost:,.0f}", delta_color="inverse")
    
    # --- VISUALIZATION SECTION (Unified Grid Schematic) ---
    st.markdown("### 🗺️ Network Topology & Status")
    
    fig = go.Figure()

    # 1. Add Transmission Lines with Flow Labels at Midpoints
    for name, line in network.lines.iterrows():
        bus0 = network.buses.loc[line.bus0]
        bus1 = network.buses.loc[line.bus1]
        flow = network.lines_t.p0.at[network.snapshots[0], name]
        
        # Color based on flow direction/magnitude
        line_color = "rgba(255, 255, 255, 0.4)"
        
        # Draw the line
        fig.add_trace(go.Scatter(
            x=[bus0.x, bus1.x],
            y=[bus0.y, bus1.y],
            mode='lines',
            line=dict(color=line_color, width=2),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add the persistent Flow Label at midpoint
        mid_x = (bus0.x + bus1.x) / 2
        mid_y = (bus0.y + bus1.y) / 2
        fig.add_trace(go.Scatter(
            x=[mid_x], y=[mid_y],
            mode='text',
            text=[f"<b>{flow:+.0f} MW</b>"],
            textposition="top center",
            textfont=dict(size=10, color="white"),
            hoverinfo='none',
            showlegend=False
        ))

    # 2. Add Buses with Persistent Metrics
    bus_x = network.buses.x
    bus_y = network.buses.y
    bus_labels = []
    
    for bus in network.buses.index:
        plants = network.generators[network.generators.bus == bus].index.tolist()
        plant_names = ", ".join(plants) if plants else "No Plants"
        
        gen = network.generators_t.p[network.generators[network.generators.bus == bus].index].iloc[0].sum() if not network.generators[network.generators.bus == bus].empty else 0
        load = network.loads[network.loads.bus == bus].p_set.sum()
        bus_labels.append(f"<b>{bus}</b><br><span style='color: #ffa500; font-size: 9px;'>Plants: {plant_names}</span><br>Gen: {gen:.0f} | Load: {load:.0f}")

    fig.add_trace(go.Scatter(
        x=bus_x,
        y=bus_y,
        mode='markers+text',
        marker=dict(size=18, color='#00ffcc', line=dict(color='white', width=1)),
        text=bus_labels,
        textposition="bottom center",
        textfont=dict(size=11, color="#00ffcc"),
        hoverinfo='none',
        name="Grid Nodes"
    ))

    # Update layout for a clean, schematic look
    fig.update_layout(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font_color='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=100, r=100, t=100, b=50),
        height=900,
        title="Unified Grid Schematic"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- ASSET & TOPOLOGY SECTION ---
    st.markdown("### 📊 Assets & Topology")
    tab1, tab2 = st.tabs(["💡 Generator Dispatch", "🔌 Network Connections"])
    
    with tab1:
        # Map generators to their buses and show outputs
        gen_df = network.generators[['bus', 'p_nom', 'marginal_cost']].copy()
        gen_df['Dispatch (MW)'] = network.generators_t.p.iloc[0]
        gen_df['Utilization (%)'] = (gen_df['Dispatch (MW)'] / gen_df['p_nom']) * 100
        gen_df['Gen Cost ($)'] = gen_df['Dispatch (MW)'] * gen_df['marginal_cost']
        
        # Rename and reorder for clarity
        gen_df.columns = ['Bus Location', 'Capacity (MW)', 'Marginal Cost ($/MWh)', 'Dispatch (MW)', 'Utilization (%)', 'Gen Cost ($)']
        gen_df = gen_df[['Bus Location', 'Capacity (MW)', 'Dispatch (MW)', 'Gen Cost ($)', 'Marginal Cost ($/MWh)', 'Utilization (%)']]
        
        st.dataframe(
            gen_df.style.format({
                'Utilization (%)': '{:.1f}%', 
                'Marginal Cost ($/MWh)': '${:.0f}', 
                'Gen Cost ($)': '${:,.0f}'
            })
            .background_gradient(subset=['Utilization (%)'], cmap="YlOrRd")
            .background_gradient(subset=['Gen Cost ($)'], cmap="Blues"), 
            use_container_width=True
        )
        
        st.info("**Analysis:** Each generator's **Gen Cost** = Dispatch × Marginal Cost. PyPSA dispatches cheapest first (Solar → Wind → Coal → Gas).")
        st.caption("Complete economic dispatch with individual generation costs shown.")

    with tab2:
        # Show which bus is connected to which
        topology = []
        for name, line in network.lines.iterrows():
            topology.append({'Bus': line.bus0, 'Connected To': line.bus1, 'Line Name': name, 'Capacity': line.s_nom})
            topology.append({'Bus': line.bus1, 'Connected To': line.bus0, 'Line Name': name, 'Capacity': line.s_nom})
        
        topo_df = pd.DataFrame(topology)
        st.dataframe(topo_df.sort_values('Bus'), use_container_width=True, hide_index=True)
        st.caption("Mapping of all electrical connections between buses and their transmission capacities.")

    # --- CITY POWER SOURCING SECTION ---
    st.markdown("### 🔍 City Power Sourcing Analysis")
    selected_city = st.selectbox("Select a City to analyze its energy balance:", ["City A", "City B", "City C", "City D"])
    
    city_bus = selected_city.replace(" ", "_")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown(f"#### ⚡ Local Generation")
        local_gen = network.generators[network.generators.bus == city_bus].copy()
        if not local_gen.empty:
            local_gen['Output (MW)'] = network.generators_t.p[local_gen.index].iloc[0]
            st.table(local_gen[['p_nom', 'Output (MW)']])
            total_local = local_gen['Output (MW)'].sum()
        else:
            st.info("No local generators.")
            total_local = 0
            
    inflows = []
    outflows = []
    # Check all lines connected to this bus
    for name, line in network.lines.iterrows():
        flow = network.lines_t.p0.at[network.snapshots[0], name]
        if line.bus0 == city_bus:
            if flow < 0: # Entering bus
                inflows.append({'From/To': line.bus1, 'Link': name, 'MW': abs(flow)})
            else: # Leaving bus
                outflows.append({'From/To': line.bus1, 'Link': name, 'MW': flow})
        elif line.bus1 == city_bus:
            if flow > 0: # Entering bus
                inflows.append({'From/To': line.bus0, 'Link': name, 'MW': flow})
            else: # Leaving bus
                outflows.append({'From/To': line.bus0, 'Link': name, 'MW': abs(flow)})

    with col_b:
        st.markdown(f"#### 📥 Inflows (Imports)")
        if inflows:
            st.table(pd.DataFrame(inflows))
            total_inflow = sum(i['MW'] for i in inflows)
        else:
            st.info("No power imported.")
            total_inflow = 0

    with col_c:
        st.markdown(f"#### 📤 Outflows (Exports)")
        if outflows:
            st.table(pd.DataFrame(outflows))
            total_outflow = sum(i['MW'] for i in outflows)
        else:
            st.info("No power exported.")
            total_outflow = 0

    # Summary of the Balance
    city_demand = network.loads[network.loads.bus == city_bus].p_set.sum()
    st.info(f"**Energy Balance for {selected_city}:**")
    st.markdown(f"""
    *   **Demand**: `{city_demand:.0f} MW`
    *   **Formula**: `Demand = (Local Gen + Imports) - Exports`
    *   **Calculation**: `{city_demand:.0f} MW = ({total_local:.0f} + {total_inflow:.0f}) - {total_outflow:.0f}`
    """)

    st.success("Optimization completed successfully!")
else:
    total_demand = d_a + d_b + d_c + d_d + d_solar + d_wind
    total_capacity = 500+700+1000+800  # Fixed generator sum
    st.warning(f"⚠️ **Infeasible Grid State**: The regional demand ({total_demand} MW) exceeds available capacity ({total_capacity} MW). Please reduce city loads in the sidebar.")

st.markdown("---")
st.caption("Powered by PyPSA | Complete Economic Dispatch Analysis with Individual Costs")
