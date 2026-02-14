# ============================================
# STATE GRID OPTIMIZER
# ============================================
# This application simulates and optimizes the state electricity grid.
# It models real power plants, transmission lines, and demand centers to find
# the most cost-effective way to meet electricity demand while respecting
# physical constraints (line capacities, generator limits, power flow laws).

# --- IMPORTS ---
import streamlit as st  # Web app framework for interactive dashboards
import pypsa  # Python for Power System Analysis - core optimization engine
import pandas as pd  # Data manipulation for network results
import plotly.graph_objects as go  # Interactive visualizations (charts, network maps)
from datetime import datetime  # Timestamp for footer
import numpy as np  # Numerical operations

# --- PAGE CONFIGURATION ---
# Sets up the Streamlit app's browser tab title, icon, and layout
st.set_page_config(
    page_title="Grid Optimizer",  # Browser tab title
    page_icon="⚡",  # Browser tab icon
    layout="wide",  # Use full screen width for better data visualization
    initial_sidebar_state="expanded"  # Show sidebar by default (contains demand controls)
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

    .tech-label {
        color: #808495;
        font-size: 0.8rem;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        letter-spacing: 0.1em;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# UTILITY FUNCTIONS FOR GRID ANALYSIS
# ============================================

def calculate_transmission_losses(network):
    """
    Calculate total transmission losses across all power lines.
    """
    total_losses = 0
    # Robust check for flow data
    if 'p0' not in network.lines_t or network.lines_t.p0.empty:
        return 0.0

    for name, line in network.lines.iterrows():
        if name not in network.lines_t.p0.columns:
            continue
        flow = abs(network.lines_t.p0.at[network.snapshots[0], name])
        loss = flow * 0.025 * (line.length / 100)
        total_losses += loss
    return total_losses

def calculate_renewable_penetration(network):
    """
    Calculate the percentage of electricity generated from renewable sources.
    """
    total_gen = network.generators_t.p.iloc[0].sum()
    if total_gen == 0:
        return 0
    
    renewable_gen = 0
    renewable_plants = ['Solar_Raipur', 'Solar_Korba', 'Solar_Bilaspur', 'Hydro_Minimata_Bango']
    for plant in renewable_plants:
        if plant in network.generators_t.p.columns:
            renewable_gen += network.generators_t.p[plant].iloc[0]
            
    return (renewable_gen / total_gen * 100)

def calculate_coal_share(network):
    """
    Calculate the percentage of electricity generated from coal plants.
    """
    total_gen = network.generators_t.p.iloc[0].sum()
    if total_gen == 0:
        return 0
    
    coal_gen = 0
    coal_plants = ['Coal_NTPC_Sipat', 'Coal_NTPC_Korba', 'Coal_Hasdeo_BCCL', 
                   'Coal_Korba_West', 'Coal_LARA_STPP', 'Coal_Marwa_TPP']
    for plant in coal_plants:
        if plant in network.generators_t.p.columns:
            coal_gen += network.generators_t.p[plant].iloc[0]
            
    return (coal_gen / total_gen * 100)

def run_simulation(scenario_params, finalize=True):
    """
    State Grid Optimization
    """
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=1, freq="h"))
    
    # ... (Buses, Lines, Generators, Loads logic remains same but needs to be in this scope)
    # Re-adding the full body of run_simulation to include components
    
    # MAJOR CITIES
    n.add("Bus", "Raipur", v_nom=220, x=6, y=5, carrier="AC")
    n.add("Bus", "Bilaspur", v_nom=220, x=8, y=7, carrier="AC")
    n.add("Bus", "Durg_Bhilai", v_nom=220, x=5, y=4, carrier="AC")
    n.add("Bus", "Korba", v_nom=220, x=9, y=8, carrier="AC")
    n.add("Bus", "Raigarh", v_nom=132, x=10, y=9, carrier="AC")
    
    n.add("Bus", "Bhilai_Steel", v_nom=220, x=5, y=3, carrier="AC")
    n.add("Bus", "Korba_East", v_nom=220, x=10, y=8, carrier="AC")
    n.add("Bus", "Hasdeo_Basin", v_nom=132, x=8, y=9, carrier="AC")
    
    n.add("Bus", "Jagdalpur", v_nom=132, x=4, y=1, carrier="AC")
    n.add("Bus", "Dantewada", v_nom=132, x=3, y=0, carrier="AC")
    n.add("Bus", "Surguja", v_nom=132, x=7, y=10, carrier="AC")
    n.add("Bus", "Ambikapur", v_nom=132, x=6, y=11, carrier="AC")
    
    n.add("Bus", "Sipat_Power", v_nom=400, x=7, y=6, carrier="AC")
    n.add("Bus", "Korba_Power_Complex", v_nom=400, x=9, y=7.5, carrier="AC")
    
    # LINES
    n.add("Line", "Line_Raipur_Durg", bus0="Raipur", bus1="Durg_Bhilai", x=0.02, r=0.004, s_nom=2500, length=40)
    n.add("Line", "Line_Raipur_Bilaspur", bus0="Raipur", bus1="Bilaspur", x=0.04, r=0.008, s_nom=2500, length=120)
    n.add("Line", "Line_Raipur_Sipat", bus0="Raipur", bus1="Sipat_Power", x=0.03, r=0.006, s_nom=3500, length=80)
    n.add("Line", "Line_Raipur_Jagdalpur", bus0="Raipur", bus1="Jagdalpur", x=0.06, r=0.012, s_nom=600, length=300)
    n.add("Line", "Line_Durg_BhilaiSteel", bus0="Durg_Bhilai", bus1="Bhilai_Steel", x=0.01, r=0.002, s_nom=2500, length=15)
    n.add("Line", "Line_Durg_Jagdalpur", bus0="Durg_Bhilai", bus1="Jagdalpur", x=0.07, r=0.014, s_nom=500, length=350)
    n.add("Line", "Line_Bilaspur_Korba", bus0="Bilaspur", bus1="Korba", x=0.03, r=0.006, s_nom=3500, length=90)
    n.add("Line", "Line_Bilaspur_Sipat", bus0="Bilaspur", bus1="Sipat_Power", x=0.02, r=0.004, s_nom=3000, length=60)
    n.add("Line", "Line_Korba_KorbaPower", bus0="Korba", bus1="Korba_Power_Complex", x=0.01, r=0.002, s_nom=4000, length=20)
    n.add("Line", "Line_Korba_Raigarh", bus0="Korba", bus1="Raigarh", x=0.03, r=0.006, s_nom=1500, length=100)
    n.add("Line", "Line_Korba_KorbaEast", bus0="Korba", bus1="Korba_East", x=0.02, r=0.004, s_nom=1000, length=50)
    n.add("Line", "Line_KorbaEast_Hasdeo", bus0="Korba_East", bus1="Hasdeo_Basin", x=0.02, r=0.004, s_nom=800, length=70)
    n.add("Line", "Line_Hasdeo_Surguja", bus0="Hasdeo_Basin", bus1="Surguja", x=0.03, r=0.006, s_nom=600, length=80)
    n.add("Line", "Line_Bilaspur_Surguja", bus0="Bilaspur", bus1="Surguja", x=0.04, r=0.008, s_nom=700, length=150)
    n.add("Line", "Line_Surguja_Ambikapur", bus0="Surguja", bus1="Ambikapur", x=0.02, r=0.004, s_nom=500, length=60)
    n.add("Line", "Line_Jagdalpur_Dantewada", bus0="Jagdalpur", bus1="Dantewada", x=0.03, r=0.006, s_nom=400, length=90)
    n.add("Line", "Line_Sipat_KorbaPower", bus0="Sipat_Power", bus1="Korba_Power_Complex", x=0.02, r=0.004, s_nom=4500, length=70)
    n.add("Line", "Line_Raigarh_KorbaPower", bus0="Raigarh", bus1="Korba_Power_Complex", x=0.03, r=0.006, s_nom=2000, length=90)
    
    # GENERATORS
    n.add("Generator", "Coal_NTPC_Sipat", bus="Sipat_Power", p_nom=2980, marginal_cost=2.5, carrier="coal")
    n.add("Generator", "Coal_NTPC_Korba", bus="Korba_Power_Complex", p_nom=2600, marginal_cost=2.5, carrier="coal")
    n.add("Generator", "Coal_Korba_West", bus="Korba_Power_Complex", p_nom=2100, marginal_cost=2.6, carrier="coal")
    n.add("Generator", "Coal_LARA_STPP", bus="Raigarh", p_nom=1600, marginal_cost=2.7, carrier="coal")
    n.add("Generator", "Coal_Marwa_TPP", bus="Raigarh", p_nom=1000, marginal_cost=2.7, carrier="coal")
    n.add("Generator", "Coal_Hasdeo_BCCL", bus="Hasdeo_Basin", p_nom=600, marginal_cost=2.6, carrier="coal")
    n.add("Generator", "Solar_Raipur", bus="Raipur", p_nom=scenario_params['solar_raipur'], marginal_cost=0.0, carrier="solar")
    n.add("Generator", "Solar_Korba", bus="Korba", p_nom=scenario_params['solar_korba'], marginal_cost=0.0, carrier="solar")
    n.add("Generator", "Solar_Bilaspur", bus="Bilaspur", p_nom=scenario_params['solar_bilaspur'], marginal_cost=0.0, carrier="solar")
    n.add("Generator", "Hydro_Minimata_Bango", bus="Durg_Bhilai", p_nom=60, marginal_cost=0.8, carrier="hydro")
    
    # LOADS
    n.add("Load", "Load_Raipur_City", bus="Raipur", p_set=scenario_params['demand_raipur'])
    n.add("Load", "Load_Bilaspur", bus="Bilaspur", p_set=scenario_params['demand_bilaspur'])
    n.add("Load", "Load_Durg", bus="Durg_Bhilai", p_set=scenario_params['demand_durg'])
    n.add("Load", "Load_Korba", bus="Korba", p_set=scenario_params['demand_korba'])
    n.add("Load", "Load_Raigarh", bus="Raigarh", p_set=scenario_params['demand_raigarh'])
    n.add("Load", "Load_Bhilai_Steel_Plant", bus="Bhilai_Steel", p_set=scenario_params['demand_bhilai_steel'])
    n.add("Load", "Load_Korba_Industries", bus="Korba_East", p_set=scenario_params['demand_korba_industries'])
    n.add("Load", "Load_Jagdalpur", bus="Jagdalpur", p_set=scenario_params['demand_jagdalpur'])
    n.add("Load", "Load_Dantewada", bus="Dantewada", p_set=scenario_params['demand_dantewada'])
    n.add("Load", "Load_Surguja", bus="Surguja", p_set=scenario_params['demand_surguja'])
    n.add("Load", "Load_Ambikapur", bus="Ambikapur", p_set=scenario_params['demand_ambikapur'])
    n.add("Load", "Load_Hasdeo", bus="Hasdeo_Basin", p_set=scenario_params['demand_hasdeo'])
    
    if not finalize:
        return n, 0.0, None

    try:
        n.optimize(solver_name='highs')
        if n.objective is not None and not pd.isna(n.objective):
            return n, n.objective, None
        else:
            return None, 0.0, "Optimization failed - no valid solution"
    except Exception as e:
        return None, 0.0, str(e)

def run_simple_dispatch(scenario_params):
    """
    Non-optimized dispatch: Simple merit order (Renewable -> Largest Coal -> Smallest Coal).
    """
    n, _, error = run_simulation(scenario_params, finalize=False)
    if error:
        return None, 0.0, f"Network setup failed: {error}"
    
    # Initialize results DataFrames
    n.generators_t.p = pd.DataFrame(0.0, index=n.snapshots, columns=n.generators.index)
    total_demand = n.loads.p_set.sum()
    
    # Sorting logic: Largest Plants First (Dumb Dispatch)
    # This represents a legacy approach that ignores fuel costs/renewables
    sorted_gens = n.generators.sort_values(by=['p_nom'], ascending=[False])
    
    remaining_demand = total_demand
    dispatch = pd.Series(0.0, index=n.generators.index)
    
    for gen_id, gen in sorted_gens.iterrows():
        if remaining_demand <= 0: break
        p_out = min(gen.p_nom, remaining_demand)
        dispatch[gen_id] = p_out
        remaining_demand -= p_out
    
    n.generators_t.p.loc[n.snapshots[0]] = dispatch
    total_cost = (dispatch * n.generators.marginal_cost).sum()
    return n, total_cost, None

# ============================================
# STREAMLIT APP
# ============================================
st.markdown('<h1>GRID OPTIMIZER</h1>', unsafe_allow_html=True)
st.markdown('<div class="grid-status">🏭 Coal Capital of India • Power Surplus State • Real-time Dispatch Optimization</div>', unsafe_allow_html=True)

# --- SIDEBAR WITH REALISTIC DEFAULTS ---
st.sidebar.markdown("### 🎛️ LOAD CENTERS (MW)")
st.sidebar.markdown("---")

st.sidebar.markdown('<div class="district-header">URBAN CENTERS</div>', unsafe_allow_html=True)
demand_raipur = st.sidebar.slider("Raipur (Capital)", 0, 1500, 600, step=50)
demand_bilaspur = st.sidebar.slider("Bilaspur", 0, 1200, 500, step=50)
demand_durg = st.sidebar.slider("Durg-Bhilai", 0, 1000, 400, step=50)
demand_korba = st.sidebar.slider("Korba", 0, 800, 350, step=50)
demand_raigarh = st.sidebar.slider("Raigarh", 0, 800, 300, step=50)

st.sidebar.markdown('<div class="district-header">HEAVY INDUSTRIES</div>', unsafe_allow_html=True)
demand_bhilai_steel = st.sidebar.slider("SAIL Bhilai Steel Plant", 0, 2000, 800, step=100)
demand_korba_industries = st.sidebar.slider("Korba Industries & Mines", 0, 1200, 500, step=50)

st.sidebar.markdown('<div class="district-header">RURAL & TRIBAL DISTRICTS</div>', unsafe_allow_html=True)
demand_jagdalpur = st.sidebar.slider("Jagdalpur (Bastar)", 0, 400, 200, step=25)
demand_dantewada = st.sidebar.slider("Dantewada", 0, 300, 120, step=25)
demand_surguja = st.sidebar.slider("Surguja", 0, 400, 180, step=25)
demand_ambikapur = st.sidebar.slider("Ambikapur", 0, 300, 140, step=25)
demand_hasdeo = st.sidebar.slider("Hasdeo Basin", 0, 400, 150, step=25)

st.sidebar.markdown("---")
st.sidebar.markdown("### ☀️ SOLAR CAPACITY (MW)")

solar_raipur = st.sidebar.slider("Raipur Solar Parks", 0, 800, 400, step=50)
solar_korba = st.sidebar.slider("Korba Solar", 0, 600, 300, step=50)
solar_bilaspur = st.sidebar.slider("Bilaspur Solar", 0, 500, 250, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 QUICK SCENARIOS")

if st.sidebar.button("🌅 Normal Day Peak"):
    st.rerun()

if st.sidebar.button("🌙 Night / Off-Peak"):
    demand_raipur = 500
    demand_bilaspur = 400
    demand_bhilai_steel = 800
    solar_raipur = 0
    solar_korba = 0

if st.sidebar.button("🏭 Industrial Peak"):
    demand_bhilai_steel = 1600
    demand_korba_industries = 900
    demand_raigarh = 550

if st.sidebar.button("🌾 Agriculture Season"):
    demand_surguja = 250
    demand_hasdeo = 200
    demand_dantewada = 150

# Compile scenario
scenario_params = {
    'demand_raipur': demand_raipur,
    'demand_bilaspur': demand_bilaspur,
    'demand_durg': demand_durg,
    'demand_korba': demand_korba,
    'demand_raigarh': demand_raigarh,
    'demand_bhilai_steel': demand_bhilai_steel,
    'demand_korba_industries': demand_korba_industries,
    'demand_jagdalpur': demand_jagdalpur,
    'demand_dantewada': demand_dantewada,
    'demand_surguja': demand_surguja,
    'demand_ambikapur': demand_ambikapur,
    'demand_hasdeo': demand_hasdeo,
    'solar_raipur': solar_raipur,
    'solar_korba': solar_korba,
    'solar_bilaspur': solar_bilaspur
}

# Calculate totals
total_demand = sum([
    demand_raipur, demand_bilaspur, demand_durg, demand_korba, demand_raigarh,
    demand_bhilai_steel, demand_korba_industries,
    demand_jagdalpur, demand_dantewada, demand_surguja, demand_ambikapur, demand_hasdeo
])

total_capacity = (
    2980 + 2600 + 2100 + 1600 + 1000 + 600 +  # Coal: 10,880 MW
    60 +  # Hydro
    solar_raipur + solar_korba + solar_bilaspur  # Solar
)

# --- DISPATCH MODE SELECTION ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ DISPATCH STRATEGY")
dispatch_mode = st.sidebar.radio(
    "Select Mode",
    ["Optimized (Smart Grid)", "Non-Optimized (Simple)", "Comparison View"],
    help="Optimized uses PyPSA to find the lowest cost. Non-Optimized just uses largest plants first."
)

# Run simulation based on mode
with st.spinner("⚙️ Processing grid state..."):
    if dispatch_mode == "Optimized (Smart Grid)":
        network, total_cost, error = run_simulation(scenario_params)
    elif dispatch_mode == "Non-Optimized (Simple)":
        network, total_cost, error = run_simple_dispatch(scenario_params)
    else:  # Comparison View
        opt_net, opt_cost, opt_err = run_simulation(scenario_params)
        simple_net, simple_cost, simple_err = run_simple_dispatch(scenario_params)
        network = opt_net
        total_cost = opt_cost
        error = opt_err or simple_err

if network is not None:
    # KPI metrics logic (calculated for the primary active network)
    total_generation = network.generators_t.p.iloc[0].sum() if not network.generators_t.p.empty else 0.0
    renewable_pct = calculate_renewable_penetration(network)
    coal_pct = calculate_coal_share(network)
    transmission_losses = calculate_transmission_losses(network)
    
    # Pre-calculate gen_df for the primary network (used in charts and tabs)
    gen_data = []
    for gen_name in network.generators.index:
        output = network.generators_t.p[gen_name].iloc[0] if gen_name in network.generators_t.p.columns else 0.0
        cost = output * network.generators.at[gen_name, 'marginal_cost']
        carrier = network.generators.at[gen_name, 'carrier']
        capacity = network.generators.at[gen_name, 'p_nom']
        gen_data.append({
            'Plant': gen_name.replace('_', ' '),
            'Output (MW)': output,
            'Cost (₹/hr)': cost * 75,
            'Type': carrier,
            'Capacity (MW)': capacity,
            'Utilization (%)': (output / capacity * 100) if capacity > 0 else 0
        })
    gen_df = pd.DataFrame(gen_data).sort_values('Output (MW)', ascending=False)
    
    if dispatch_mode == "Comparison View":
        st.markdown("### ⚖️ OPTIMIZATION vs. SIMPLE DISPATCH")
        cc1, cc2, cc3 = st.columns(3)
        savings_hr = (simple_cost - opt_cost) * 75
        savings_pct = (simple_cost - opt_cost) / simple_cost * 100 if simple_cost > 0 else 0
        cc1.metric("Cost Savings", f"₹{savings_hr:,.0f}/hr", f"{savings_pct:.1f}% Savings")
        
        opt_ren = calculate_renewable_penetration(opt_net)
        sim_ren = calculate_renewable_penetration(simple_net)
        cc2.metric("Renewable Utilization", f"{opt_ren:.1f}%", f"{opt_ren - sim_ren:+.1f}% vs Simple")
        
        opt_loss = calculate_transmission_losses(opt_net)
        cc3.metric("Opt. Grid Losses", f"{opt_loss:.1f} MW", "Minimized by Solver")
        st.info(f"💡 **Optimization Value:** The optimizer saved **₹{savings_hr:,.0f} per hour** by prioritizing low-cost renewables and efficient plants. The 'Simple Dispatch' (Largest First) strategy wastes free solar energy by running large coal plants first.")
        st.markdown("---")
    # ============================================
    # RESULTS
    # ============================================
    st.markdown("---")
    
    # KPI ROW
    st.markdown("#### 📊 GRID STATUS")
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    
    kpi1.metric("State Demand", f"{total_generation:,.0f} MW", 
               help="Instantaneous Power demand (MW). At a 1-hour simulation, this equals energy in MWh.")
    
    kpi2.metric("Cost", f"₹{total_cost*75:,.0f}/hr", 
                f"₹{total_cost*75/total_demand:.2f}/MWh",
                help="Total system generation cost per hour.")
    
    kpi3.metric("Renewable", f"{renewable_pct:.1f}%")
    kpi4.metric("Coal", f"{coal_pct:.1f}%")
    
    if dispatch_mode == "Non-Optimized (Simple)":
        kpi5.metric("Grid Loss", "N/A", delta="Ignored in Simple Mode")
    else:
        kpi5.metric("Grid Loss", f"{transmission_losses:.0f} MW", 
                    delta=f"{transmission_losses/total_generation*100:.2f}%")
    
    kpi6.metric("Active Plants", "10", delta="Operational")
    
    st.markdown("---")
    
    # ============================================
    # GENERATION CAPACITY & COSTS
    # ============================================
    st.markdown("#### ⚡ Generation Capacity & Costs")
    gcol1, gcol2, gcol3, gcol4, gcol5, gcol6 = st.columns(6)
    
    total_capacity = network.generators.p_nom.sum()
    gcol1.metric("Total Capacity", f"{total_capacity:,.0f} MW", delta="System Generation", delta_color="off")
    
    # Coal plants
    coal_capacity = (network.generators.at['Coal_NTPC_Sipat', 'p_nom'] + 
                     network.generators.at['Coal_NTPC_Korba', 'p_nom'] + 
                     network.generators.at['Coal_Hasdeo_BCCL', 'p_nom'] + 
                     network.generators.at['Coal_Korba_West', 'p_nom'] + 
                     network.generators.at['Coal_LARA_STPP', 'p_nom'] + 
                     network.generators.at['Coal_Marwa_TPP', 'p_nom'])
    coal_avg_cost = 2.6  # Average marginal cost
    gcol2.metric("Coal Plants", f"{coal_capacity:,.0f} MW", 
                delta=f"₹{coal_avg_cost}/MWh", delta_color="off")
    
    # Solar
    solar_capacity = (network.generators.at['Solar_Raipur', 'p_nom'] + 
                      network.generators.at['Solar_Korba', 'p_nom'] + 
                      network.generators.at['Solar_Bilaspur', 'p_nom'])
    gcol3.metric("Solar Plants", f"{solar_capacity:,.0f} MW", 
                delta="₹0.0/MWh", delta_color="normal")
    
    # Hydro
    hydro_capacity = network.generators.at['Hydro_Minimata_Bango', 'p_nom']
    hydro_cost = network.generators.at['Hydro_Minimata_Bango', 'marginal_cost']
    gcol4.metric("Hydro Plant", f"{hydro_capacity:,.0f} MW", 
                delta=f"₹{hydro_cost}/MWh", delta_color="normal")
    
    # Top coal plants
    gcol5.metric("NTPC Sipat", f"{network.generators.at['Coal_NTPC_Sipat', 'p_nom']:,.0f} MW", 
                delta=f"₹{network.generators.at['Coal_NTPC_Sipat', 'marginal_cost']}/MWh", delta_color="off")
    gcol6.metric("NTPC Korba", f"{network.generators.at['Coal_NTPC_Korba', 'p_nom']:,.0f} MW", 
                delta=f"₹{network.generators.at['Coal_NTPC_Korba', 'marginal_cost']}/MWh", delta_color="off")
    
    st.markdown("---")
    
    # ============================================
    # DEMAND BREAKDOWN
    # ============================================
    st.markdown("#### 💡 Demand Breakdown")
    dcol1, dcol2, dcol3, dcol4, dcol5, dcol6, dcol7 = st.columns(7)
    
    dcol1.metric("Total Demand", f"{total_demand:,.0f} MW", delta="System Load", delta_color="off")
    
    dcol2.metric("Raipur", f"{scenario_params['demand_raipur']:,.0f} MW")
    dcol3.metric("Bilaspur", f"{scenario_params['demand_bilaspur']:,.0f} MW")
    dcol4.metric("Bhilai Steel", f"{scenario_params['demand_bhilai_steel']:,.0f} MW")
    dcol5.metric("Korba", f"{scenario_params['demand_korba']:,.0f} MW")
    dcol6.metric("Raigarh", f"{scenario_params['demand_raigarh']:,.0f} MW")
    dcol7.metric("Others", f"{scenario_params['demand_jagdalpur'] + scenario_params['demand_dantewada'] + scenario_params['demand_surguja']:,.0f} MW")
    
    st.markdown("---")
    
    # ============================================
    # DISPATCH & GENERATION ECONOMICS
    # ============================================
    st.markdown("#### 💰 Dispatch & Generation Economics")
    
    if dispatch_mode == "Comparison View":
        # Comparative Economics Table
        econ_data = []
        for gen_name in network.generators.index:
            opt_p = opt_net.generators_t.p[gen_name].iloc[0]
            sim_p = simple_net.generators_t.p[gen_name].iloc[0]
            mc = network.generators.at[gen_name, 'marginal_cost']
            
            econ_data.append({
                'Plant': gen_name.replace('_', ' '),
                'Opt. Dispatch (MW)': opt_p,
                'Simple Dispatch (MW)': sim_p,
                'Difference (MW)': opt_p - sim_p,
                'Cost Saving (₹/hr)': (sim_p - opt_p) * mc * 75
            })
        
        econ_df = pd.DataFrame(econ_data)
        st.dataframe(
            econ_df.style.format({
                'Opt. Dispatch (MW)': '{:.0f}',
                'Simple Dispatch (MW)': '{:.0f}',
                'Difference (MW)': '{:+.0f}',
                'Cost Saving (₹/hr)': '₹{:,.0f}'
            }).background_gradient(subset=['Cost Saving (₹/hr)'], cmap='Greens'),
            use_container_width=True
        )
    else:
        ecol1, ecol2, ecol3, ecol4, ecol5 = st.columns(5)
        # (Keep original ecol logic)
        coal_sipat_p = network.generators_t.p['Coal_NTPC_Sipat'].iloc[0]
        coal_sipat_cost = coal_sipat_p * network.generators.at['Coal_NTPC_Sipat', 'marginal_cost']
        coal_korba_p = network.generators_t.p['Coal_NTPC_Korba'].iloc[0]
        coal_korba_cost = coal_korba_p * network.generators.at['Coal_NTPC_Korba', 'marginal_cost']
        solar_total_p = sum([network.generators_t.p[g].iloc[0] for g in ['Solar_Raipur', 'Solar_Korba', 'Solar_Bilaspur']])
        hydro_p = network.generators_t.p['Hydro_Minimata_Bango'].iloc[0]
        
        ecol1.metric("Total Cost", f"₹{total_cost*75:,.0f}/hr", delta=f"₹{total_cost*75/total_demand:.2f}/MWh", delta_color="inverse")
        ecol2.metric("NTPC Sipat", f"{coal_sipat_p:.0f} MW", delta=f"₹{coal_sipat_cost*75:,.0f}/hr", delta_color="off")
        ecol3.metric("NTPC Korba", f"{coal_korba_p:.0f} MW", delta=f"₹{coal_korba_cost*75:,.0f}/hr", delta_color="off")
        ecol4.metric("Solar Total", f"{solar_total_p:.0f} MW", delta="₹0/hr", delta_color="normal")
        ecol5.metric("Hydro", f"{hydro_p:.0f} MW", delta=f"₹{hydro_p*0.8*75:,.0f}/hr", delta_color="normal")
    
    st.markdown("---")
    
    # ============================================
    # GENERATION DISPATCH
    # ============================================
    st.markdown("#### ⚡ POWER GENERATION DISPATCH")
    
    if dispatch_mode == "Comparison View":
        comp_col1, comp_col2 = st.columns(2)
        
        def create_gen_fig(n, title, color_theme):
            gen_data = []
            for gen_name in n.generators.index:
                output = n.generators_t.p[gen_name].iloc[0]
                carrier = n.generators.at[gen_name, 'carrier']
                capacity = n.generators.at[gen_name, 'p_nom']
                gen_data.append({
                    'Plant': gen_name.replace('_', ' '),
                    'Output (MW)': output,
                    'Type': carrier,
                    'Capacity (MW)': capacity,
                    'Utilization (%)': (output / capacity * 100) if capacity > 0 else 0
                })
            df = pd.DataFrame(gen_data).sort_values('Output (MW)', ascending=False)
            fig = go.Figure()
            carrier_colors = {'solar': '#ffd700', 'hydro': '#42a5f5', 'coal': color_theme}
            for _, row in df.iterrows():
                fig.add_trace(go.Bar(
                    y=[row['Plant']], x=[row['Output (MW)']], orientation='h',
                    marker_color=carrier_colors.get(row['Type'], '#90a4ae'),
                    text=f"{row['Output (MW)']:.0f} MW", textposition='inside',
                    showlegend=False
                ))
            fig.update_layout(title=title, height=400, margin=dict(l=10, r=10, t=30, b=10),
                             plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                             font=dict(color='white'))
            return fig

        with comp_col1:
            st.plotly_chart(create_gen_fig(opt_net, "✅ Optimized Dispatch (Least Cost)", "#ff6b35"), use_container_width=True)
            
            st.markdown("##### 🏭 Fuel Mix (Optimized)")
            f1, f2, f3 = st.columns(3)
            f1.metric("Solar", f"{sum([opt_net.generators_t.p[g].iloc[0] for g in ['Solar_Raipur', 'Solar_Korba', 'Solar_Bilaspur']]):,.0f} MW")
            f2.metric("Coal", f"{sum([opt_net.generators_t.p[g].iloc[0] for g in opt_net.generators[opt_net.generators.carrier=='coal'].index]):,.0f} MW")
            f3.metric("Hydro", f"{opt_net.generators_t.p['Hydro_Minimata_Bango'].iloc[0]:,.0f} MW")

        with comp_col2:
            st.plotly_chart(create_gen_fig(simple_net, "⚠️ Legacy Dispatch (Largest First)", "#546e7a"), use_container_width=True)
            
            st.markdown("##### 🏭 Fuel Mix (Legacy)")
            f4, f5, f6 = st.columns(3)
            f4.metric("Solar", f"{sum([simple_net.generators_t.p[g].iloc[0] for g in ['Solar_Raipur', 'Solar_Korba', 'Solar_Bilaspur']]):,.0f} MW")
            f5.metric("Coal", f"{sum([simple_net.generators_t.p[g].iloc[0] for g in simple_net.generators[simple_net.generators.carrier=='coal'].index]):,.0f} MW")
            f6.metric("Hydro", f"{simple_net.generators_t.p['Hydro_Minimata_Bango'].iloc[0]:,.0f} MW")

    else:
        gen_col1, gen_col2 = st.columns([2, 1])
        with gen_col1:
            fig_gen = go.Figure()
            carrier_colors = {'solar': '#ffd700', 'hydro': '#42a5f5', 'coal': '#ff6b35'}
            for _, row in gen_df.iterrows():
                fig_gen.add_trace(go.Bar(
                    y=[row['Plant']], x=[row['Output (MW)']], orientation='h',
                    marker_color=carrier_colors.get(row['Type'], '#90a4ae'),
                    text=f"{row['Output (MW)']:.0f} MW", textposition='inside',
                    showlegend=False
                ))
            fig_gen.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450, margin=dict(l=10, r=10, t=10, b=10), font=dict(color='white'))
            st.plotly_chart(fig_gen, use_container_width=True)
        
        with gen_col2:
            st.markdown("##### 🏭 Fuel Mix")
            fuel_gen = {}
            for gen_name in network.generators.index:
                carrier = network.generators.at[gen_name, 'carrier']
                output = network.generators_t.p[gen_name].iloc[0]
                fuel_gen[carrier] = fuel_gen.get(carrier, 0) + output
            for fuel, gen in sorted(fuel_gen.items(), key=lambda x: x[1], reverse=True):
                st.metric(f"{fuel.capitalize()}", f"{gen:,.0f} MW", delta=f"{gen/total_generation*100:.1f}%")
    
    st.markdown("---")
    
    # ============================================
    # NETWORK MAP
    # ============================================
    st.markdown('<p class="tech-label">Live Network Topology & Power Flow</p>', unsafe_allow_html=True)
    
    fig_network = go.Figure()
    
    # --- Transmission Lines (Pro Colors + Detailed Labels) ---
    for name, line in network.lines.iterrows():
        bus0 = network.buses.loc[line.bus0]
        bus1 = network.buses.loc[line.bus1]
        
        flow = network.lines_t.p0.at[network.snapshots[0], name] if name in network.lines_t.p0.columns else 0
        loading = (abs(flow)/line.s_nom)*100 if line.s_nom > 0 else 0
        color = "#f44336" if loading > 90 else "#ffa726" if loading > 70 else "#00ffcc"
        
        fig_network.add_trace(go.Scatter(
            x=[bus0.x, bus1.x, None],
            y=[bus0.y, bus1.y, None],
            mode='lines',
            line=dict(color=color, width=loading/10 + 1),
            hovertext=(f"<b>{name.replace('_', ' ')}</b><br>" +
                       f"Flow: {flow:.1f} MW<br>" +
                       f"Capacity: {line.s_nom:.0f} MW<br>" +
                       f"Loading: {loading:.1f}%"),
            hoverinfo="text",
            showlegend=False
        ))
        
        # Restore Flow labels for major lines
        if abs(flow) > 300:
            mid_x, mid_y = (bus0.x + bus1.x) / 2, (bus0.y + bus1.y) / 2
            fig_network.add_trace(go.Scatter(
                x=[mid_x], y=[mid_y], mode='text', text=[f"{abs(flow):.0f}"],
                textfont=dict(size=10, color='white', family='Inter', weight=700),
                hoverinfo='skip', showlegend=False
            ))
    
    # --- Buses (Pro Marker + Rich Data Tooltips) ---
    bus_x, bus_y, bus_text, bus_hover, bus_colors, bus_sizes = [], [], [], [], [], []
    
    for bus in network.buses.index:
        b_data = network.buses.loc[bus]
        bus_x.append(b_data.x); bus_y.append(b_data.y)
        bus_text.append(bus)
        
        gens = network.generators[network.generators.bus == bus]
        gen_out = network.generators_t.p[gens.index].iloc[0].sum() if not gens.empty and not network.generators_t.p.empty else 0
        load = network.loads[network.loads.bus == bus].p_set.sum()
        net = gen_out - load
        
        # Build very detailed hover
        hover = f"<b>{bus.replace('_', ' ')}</b><br>Net: {net:+.1f} MW<br>Load: {load:.1f} MW"
        if not gens.empty:
            hover += "<br>--- Plants ---"
            for gname in gens.index:
                out = network.generators_t.p[gname].iloc[0] if gname in network.generators_t.p.columns else 0
                cap = network.generators.at[gname, 'p_nom']
                hover += f"<br>• {gname.replace('_', ' ')}: {out:.0f}/{cap:.0f} MW"
        bus_hover.append(hover)
        
        # Size/Color based on Net Flow (Legacy app5 logic)
        bus_colors.append("#00ffcc" if net >= 0 else "#f44336")
        bus_sizes.append(15 + min(15, abs(net)/100))

    fig_network.add_trace(go.Scatter(
        x=bus_x, y=bus_y, mode='markers+text',
        text=bus_text, textposition='top center',
        hovertext=bus_hover, hoverinfo='text',
        marker=dict(size=bus_sizes, color="white", line=dict(width=2, color=bus_colors)),
        showlegend=False
    ))
    
    fig_network.update_layout(
        plot_bgcolor='#111', paper_bgcolor='rgba(0,0,0,0)', height=600,
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    st.plotly_chart(fig_network, use_container_width=True)
    
    # Grid health
    congested = 0
    if 'p0' in network.lines_t and not network.lines_t.p0.empty:
        for name, line in network.lines.iterrows():
            if name in network.lines_t.p0.columns:
                flow = abs(network.lines_t.p0.at[network.snapshots[0], name])
                if line.s_nom > 0 and flow / line.s_nom > 0.85:
                    congested += 1
    
    if congested > 3:
        st.markdown(f'<div class="alert-critical">⚠️ <b>CONGESTION ALERT</b>: {congested} corridors above 85% capacity.</div>', unsafe_allow_html=True)
    elif congested > 0:
        st.markdown(f'<div class="alert-warning">⚠️ <b>MONITOR</b>: {congested} line(s) approaching capacity.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-success">✅ <b>GRID HEALTHY</b>: All transmission corridors operating normally.</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # DETAILED TABS
    # ============================================
    st.markdown("#### 📋 DETAILED ANALYSIS")
    
    tab1, tab2, tab3 = st.tabs([
        "⚡ Power Plants", 
        "🔌 Transmission", 
        "📍 District Loads"
    ])
    
    with tab1:
        st.markdown("##### Power Plant Status")
        
        st.dataframe(
            gen_df[['Plant', 'Type', 'Capacity (MW)', 'Output (MW)', 'Utilization (%)', 'Cost (₹/hr)']].style.format({
                'Capacity (MW)': '{:.0f}',
                'Output (MW)': '{:.0f}',
                'Utilization (%)': '{:.1f}%',
                'Cost (₹/hr)': '₹{:,.0f}'
            })
            .background_gradient(subset=['Utilization (%)'], cmap='RdYlGn')
            .background_gradient(subset=['Cost (₹/hr)'], cmap='Reds'),
            hide_index=True,
            use_container_width=True
        )
        
    
    with tab2:
        st.markdown("##### Transmission Line Loading")
        
        line_status = []
        for name, line in network.lines.iterrows():
            flow = 0.0
            if 'p0' in network.lines_t and name in network.lines_t.p0.columns:
                flow = network.lines_t.p0.at[network.snapshots[0], name]
            
            utilization = abs(flow) / line.s_nom * 100 if line.s_nom > 0 else 0
            
            line_status.append({
                'Corridor': name.replace('Line_', '').replace('_', ' → '),
                'Flow (MW)': flow,
                'Capacity (MW)': line.s_nom,
                'Utilization (%)': utilization,
                'Length (km)': line.length
            })
        
        line_df = pd.DataFrame(line_status).sort_values('Utilization (%)', ascending=False)
        
        st.dataframe(
            line_df.style.format({
                'Flow (MW)': '{:+.0f}',
                'Capacity (MW)': '{:.0f}',
                'Utilization (%)': '{:.1f}',
                'Length (km)': '{:.0f}'
            }).background_gradient(subset=['Utilization (%)'], cmap='RdYlGn_r'),
            hide_index=True,
            use_container_width=True
        )
    
    with tab3:
        st.markdown("##### District-wise Power Balance")
        
        load_balance = []
        for bus in network.buses.index:
            gens = network.generators[network.generators.bus == bus]
            local_gen = 0.0
            if len(gens) > 0 and 'p' in network.generators_t:
                cols = [c for c in gens.index if c in network.generators_t.p.columns]
                if cols:
                    local_gen = network.generators_t.p[cols].iloc[0].sum()
            
            loads = network.loads[network.loads.bus == bus]
            load = loads.p_set.sum()
            
            net = local_gen - load
            
            load_balance.append({
                'District/Zone': bus.replace('_', ' '),
                'Local Gen (MW)': local_gen,
                'Demand (MW)': load,
                'Net (MW)': net,
                'Status': 'Exporting' if net > 100 else 'Importing' if net < -100 else 'Balanced'
            })
        
        load_df = pd.DataFrame(load_balance).sort_values('Demand (MW)', ascending=False)
        
        st.dataframe(
            load_df.style.format({
                'Local Gen (MW)': '{:.0f}',
                'Demand (MW)': '{:.0f}',
                'Net (MW)': '{:+.0f}'
            }).background_gradient(subset=['Net (MW)'], cmap='RdYlGn'),
            hide_index=True,
            use_container_width=True
        )

    # --- CITY POWER SOURCING SECTION ---
    st.markdown("### 🔍 City Power Sourcing Analysis")
    selected_city = st.selectbox("Select a City to analyze its energy balance:",
    [
    "Raipur",
    "Bilaspur",
    "Durg_Bhilai",
    "Korba",
    "Korba_East",
    "Raigarh",
    "Jagdalpur",
    "Ambikapur",
    "Sipat_Power",
    "Korba_Power_Complex",
    "Dantewada",
    "Hasdeo_Basin",
    "Surguja"
    ])

    city_bus = selected_city.replace(" ", "_")
    
    # Get city demand
    city_demand = network.loads[network.loads.bus==city_bus].p_set.sum()

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown(f"#### ⚡ Local Generation at {selected_city}")
        local_gen = network.generators[network.generators.bus == city_bus].copy()
        if not local_gen.empty:
            local_gen['Capacity (MW)'] = local_gen['p_nom']
            local_gen['Output (MW)'] = 0.0
            valid_cols = [c for c in local_gen.index if c in network.generators_t.p.columns]
            if valid_cols:
                local_gen.loc[valid_cols, 'Output (MW)'] = network.generators_t.p[valid_cols].iloc[0]
            local_gen['Utilization (%)'] = (local_gen['Output (MW)'] / local_gen['Capacity (MW)'] * 100).round(1)
            local_gen['Cost (₹/MWh)'] = local_gen['marginal_cost']
            local_gen['Type'] = local_gen['carrier']
            
            display_gen = local_gen[['Type', 'Capacity (MW)', 'Output (MW)', 'Utilization (%)', 'Cost (₹/MWh)']].copy()
            display_gen.index = display_gen.index.str.replace('_', ' ')
            
            st.dataframe(
                display_gen.style.format({
                    'Capacity (MW)': '{:.0f}',
                    'Output (MW)': '{:.0f}',
                    'Utilization (%)': '{:.1f}%',
                    'Cost (₹/MWh)': '₹{:.2f}'
                }).background_gradient(subset=['Utilization (%)'], cmap='RdYlGn'),
                use_container_width=True
            )
            total_local = local_gen['Output (MW)'].sum()
            total_capacity = local_gen['Capacity (MW)'].sum()
            
            st.metric("Total Local Capacity", f"{total_capacity:.0f} MW")
            st.metric("Total Local Dispatch", f"{total_local:.0f} MW", 
                     delta=f"{total_local/total_capacity*100:.1f}% utilized" if total_capacity > 0 else "0%")
        else:
            st.info("No local generators at this bus.")
            total_local = 0
            total_capacity = 0
            
    with col_b:
        st.markdown(f"#### Import/Export Details")
        
        # Find connected lines
        imports = []
        exports = []
        
        for name, line in network.lines.iterrows():
            flow = 0.0
            if line.bus0 == city_bus:
                if 'p0' in network.lines_t and name in network.lines_t.p0.columns:
                    flow = network.lines_t.p0.at[network.snapshots[0], name]
                
                if abs(flow) < 1e-3: continue # Skip near-zero flows
                
                if flow > 0:
                    exports.append({
                        'Line': name.replace('Line_', '').replace('_', ' → '),
                        'To': line.bus1.replace('_', ' '),
                        'MW': flow,
                        'Capacity': line.s_nom,
                        'Utilization (%)': abs(flow)/line.s_nom*100 if line.s_nom > 0 else 0
                    })
                else:
                    imports.append({
                        'Line': name.replace('Line_', '').replace('_', ' → '),
                        'From': line.bus1.replace('_', ' '),
                        'MW': abs(flow),
                        'Capacity': line.s_nom,
                        'Utilization (%)': abs(flow)/line.s_nom*100 if line.s_nom > 0 else 0
                    })
            
            elif line.bus1 == city_bus:
                if 'p1' in network.lines_t and name in network.lines_t.p1.columns:
                    flow = network.lines_t.p1.at[network.snapshots[0], name]
                
                if abs(flow) < 1e-3: continue # Skip near-zero flows
                
                if flow > 0:
                    exports.append({
                        'Line': name.replace('Line_', '').replace('_', ' → '),
                        'To': line.bus0.replace('_', ' '),
                        'MW': flow,
                        'Capacity': line.s_nom,
                        'Utilization (%)': abs(flow)/line.s_nom*100 if line.s_nom > 0 else 0
                    })
                else:
                    imports.append({
                        'Line': name.replace('Line_', '').replace('_', ' → '),
                        'From': line.bus0.replace('_', ' '),
                        'MW': abs(flow),
                        'Capacity': line.s_nom,
                        'Utilization (%)': abs(flow)/line.s_nom*100 if line.s_nom > 0 else 0
                    })

        total_import = sum([i['MW'] for i in imports])
        total_export = sum([e['MW'] for e in exports])
        
        # Import breakdown
        if imports:
            st.markdown("**Imports (Incoming Power):**")
            import_df = pd.DataFrame(imports)
            st.dataframe(
                import_df.style.format({
                    'MW': '{:.0f}',
                    'Capacity': '{:.0f}',
                    'Utilization (%)': '{:.1f}%'
                }).background_gradient(subset=['MW'], cmap='Blues'),
                hide_index=True,
                use_container_width=True,
                height=150
            )
            st.metric("Total Imports", f"{total_import:.0f} MW", delta=f"{len(imports)} lines")
        else:
            st.info("No imports - self-sufficient or exporting")
            
        # Export breakdown
        if exports:
            st.markdown("**Exports (Outgoing Power):**")
            export_df = pd.DataFrame(exports)
            st.dataframe(
                export_df.style.format({
                    'MW': '{:.0f}',
                    'Capacity': '{:.0f}',
                    'Utilization (%)': '{:.1f}%'
                }).background_gradient(subset=['MW'], cmap='Oranges'),
                hide_index=True,
                use_container_width=True,
                height=150
            )
            st.metric("Total Exports", f"{total_export:.0f} MW", delta=f"{len(exports)} lines", delta_color="inverse")
        else:
            st.info("No exports - consuming all local generation")
    
    # Energy Balance Summary
    st.markdown("---")
    st.markdown(f"#### 📊 {selected_city} Energy Balance Summary")
    
    bal_col1, bal_col2, bal_col3, bal_col4 = st.columns(4)
    
    bal_col1.metric("City Demand", f"{city_demand:.0f} MW", delta="Total Load")
    bal_col2.metric("Local Generation", f"{total_local:.0f} MW", 
                   delta=f"{total_local/city_demand*100:.1f}% of demand" if city_demand > 0 else "N/A")
    bal_col3.metric("Net Imports", f"{total_import - total_export:+.0f} MW",
                   delta="Importing" if (total_import - total_export) > 0 else "Exporting")
    bal_col4.metric("Self-Sufficiency", 
                   f"{min(100, total_local/city_demand*100):.1f}%" if city_demand > 0 else "N/A",
                   delta="Local gen / Demand")
    
    # Economic Dispatch Explanation
    if total_capacity > 0 and total_local == 0:
        st.markdown("---")
        st.markdown(f"#### 💡 Why isn't {selected_city} using its own generation?")
        
        st.markdown(f"""
        <div class="alert-warning">
        <b>⚙️ ECONOMIC DISPATCH EXPLANATION</b><br><br>
        
        {selected_city} has <b>{total_capacity:.0f} MW</b> of generation capacity but is currently dispatching <b>{total_local:.0f} MW</b>.<br><br>
        
        <b>Why? The optimizer found a cheaper solution:</b><br>
        • PyPSA minimizes <b>total system cost</b>, not individual city costs<br>
        • Local generators may have <b>higher marginal costs</b> than plants elsewhere<br>
        • It's cheaper to import power from low-cost plants (e.g., solar, cheaper coal) than run expensive local plants<br><br>
        
        <b>Example for Raigarh:</b><br>
        • Raigarh has Coal_LARA_STPP (₹2.7/MWh) and Coal_Marwa_TPP (₹2.7/MWh)<br>
        • But NTPC Sipat (₹2.5/MWh) and solar (₹0/MWh) are cheaper<br>
        • Optimizer imports from cheaper sources instead of running local expensive plants<br><br>
        
        <b>This is economically optimal</b> - the grid saves money by using the cheapest generation first!<br>
        To force local generation, you would need to either:<br>
        1. Reduce transmission capacity (force local use)<br>
        2. Lower local plant costs<br>
        3. Add "must-run" constraints (not implemented here)
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="alert-success">✅ <b>OPTIMIZATION SUCCESSFUL</b>: State grid operating optimally.</div>', unsafe_allow_html=True)

else:
    # Failed optimization
    capacity_surplus = total_capacity - total_demand
    
    if capacity_surplus >= 0:
        st.markdown(f'<div class="alert-critical">⚠️ <b>OPTIMIZATION FAILED</b><br><br>' +
                   f'Capacity ({total_capacity:,.0f} MW) exceeds demand ({total_demand:,.0f} MW) but solver failed.<br><br>' +
                   f'Likely transmission constraints. Increase line capacity.</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-critical">⚠️ <b>CAPACITY DEFICIT</b><br><br>' +
                   f'Demand ({total_demand:,.0f} MW) exceeds capacity ({total_capacity:,.0f} MW).<br><br>' +
                   f'<b>Deficit:</b> {abs(capacity_surplus):,.0f} MW</div>', 
                   unsafe_allow_html=True)
    
    if error:
        with st.expander("🔍 Technical Error"):
            st.code(error)

# Footer
st.markdown("---")
st.caption("🏭 State Electricity Grid")
st.caption("⚡ Coal Capital of India • 10,880 MW Thermal Capacity • Power Surplus State")
st.caption(f"⏱️ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")