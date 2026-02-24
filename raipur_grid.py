import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from sqlalchemy import create_engine
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Raipur Smart Meter & OH Line Mapper", layout="wide", page_icon="📍")

# --- DATABASE HELPERS ---
def get_db_connection(host, port, dbname, user, password):
    try:
        connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        engine = create_engine(connection_str)
        return engine
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None

import json

@st.cache_data
def load_infra_data(_engine, feeder_ids, feeder_names, table_name):
    if _engine is None or (len(feeder_ids) == 0 and len(feeder_names) == 0):
        return pd.DataFrame()
    
    try:
        if table_name == 'rpr_cable':
            # Use Fuzzy Matching for feeder names (Upper + Trim)
            name_list = "', '".join([str(n).strip().upper().replace("'", "''") for n in feeder_names])
            query = f"""
                SELECT gid, frompoleid, topoleid, feedername, feeder_id, n_voltage, 
                       ST_AsGeoJSON(geom) as geom_json
                FROM rpr_cable
                WHERE UPPER(TRIM(feedername)) IN ('{name_list}')
            """
        else:
            # Use Feeder ID for oh_line
            feeder_list = "', '".join([get_feeder_val(f) for f in feeder_ids])
            query = f"""
                SELECT gid, frompoleid, topoleid, feedername, feeder_id, n_voltage, 
                       ST_AsGeoJSON(geom) as geom_json
                FROM oh_line
                WHERE feeder_id IN ('{feeder_list}')
            """
            
        df = pd.read_sql(query, _engine)
        
        # Color mapping
        def get_color(voltage):
            v = str(voltage).strip().upper()
            if table_name == 'oh_line':
                if '33' in v: return [255, 51, 51, 200]
                return [255, 255, 51, 200]
            else:
                if '33' in v: return [0, 255, 255, 200]
                return [255, 153, 51, 200]
            
        df['line_color'] = df['n_voltage'].apply(get_color)
        df['infra_type'] = "Overhead" if table_name == 'oh_line' else "Underground Cable"
        
        df['geom_obj'] = df['geom_json'].apply(lambda x: json.loads(x))
        
        # Explode MultiLineStrings into separate rows so all segments are plotted
        exploded_rows = []
        for _, row in df.iterrows():
            g = row['geom_obj']
            if g['type'] == 'MultiLineString':
                for segment in g['coordinates']:
                    new_row = row.copy()
                    new_row['path'] = segment
                    exploded_rows.append(new_row)
            elif g['type'] == 'LineString':
                row['path'] = g['coordinates']
                exploded_rows.append(row)
        
        df_final = pd.DataFrame(exploded_rows)
        if df_final.empty: return pd.DataFrame()
        
        return df_final.dropna(subset=['path'])
    except Exception as e:
        st.error(f"Error querying {table_name}: {e}")
        return pd.DataFrame()

@st.cache_data
def derive_dt_data(df_meters):
    if df_meters.empty:
        return pd.DataFrame()
    
    # Aggregate by DT Code to derive transformer metrics
    dt_groups = df_meters.groupby('DT Code').agg({
        'LATITUDE': 'mean',
        'LONGITUDE': 'mean',
        'DT Name': 'first',
        'Feeder Code': 'first',
        'load_kw': 'sum',
        'Meter No2': 'count'
    }).reset_index()
    
    # Map to standardized columns
    dt_groups.columns = ['dtccode', 'lat', 'lon', 'name', 'feeder_id', 'capacity', 'meter_count']
    dt_groups = dt_groups[dt_groups['dtccode'] != 'No DT'].dropna(subset=['lat', 'lon']).copy()
    
    # Generate Box (Polygon) coordinates - fixed square around the center
    # 0.00015 degrees is ~16 meters, providing a clear "Box" visual on the map
    delta = 0.00015
    dt_groups['polygon'] = dt_groups.apply(lambda r: [
        [r['lon'] - delta, r['lat'] - delta],
        [r['lon'] + delta, r['lat'] - delta],
        [r['lon'] + delta, r['lat'] + delta],
        [r['lon'] - delta, r['lat'] + delta]
    ], axis=1)
    
    return dt_groups

def get_feeder_val(f):
    try:
        return str(int(float(f)))
    except:
        return str(f)

@st.cache_data
def load_meter_data(csv_path):
    print(f"Loading Smart Meter data from {csv_path}...")
    # Load specific columns for speed
    cols = ['Meter No2', 'LATITUDE', 'LONGITUDE', 'Sanctioned Load', 'Sanctioned Load unit', 'Consumer No', 'Name', 'Feeder Code', 'DT Code', 'DT Name']
    try:
        df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()
    
    # Clean Coordinates
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    
    # Filter for Raipur region bounds
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    df = df.query("21.0 < LATITUDE < 21.5 and 81.4 < LONGITUDE < 81.8").copy()
    
    # Process Load (Aggregate into kW)
    df['Sanctioned Load'] = pd.to_numeric(df['Sanctioned Load'], errors='coerce').fillna(0)
    df['unit'] = df['Sanctioned Load unit'].astype(str).str.upper().str.strip()
    df['load_kw'] = df['Sanctioned Load']
    df.loc[df['unit'] == 'MW', 'load_kw'] = df['Sanctioned Load'] * 1000
    df.loc[df['unit'] == 'W', 'load_kw'] = df['Sanctioned Load'] * 0.001
    
    return df

def render_meter_map(df, df_infra=pd.DataFrame(), df_dt=pd.DataFrame()):
    view_state = pdk.ViewState(latitude=21.25, longitude=81.63, zoom=12, pitch=0)

    layers = []
    
    # Meter Point Layer
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["LONGITUDE", "LATITUDE"],
        get_color="[0, 255, 204, 200]",
        get_radius=0.3,
        pickable=True,
    ))
    
    # Infrastructure Layer (OH and Cables)
    if not df_infra.empty:
        layers.append(pdk.Layer(
            "PathLayer",
            df_infra,
            get_path="path",
            get_color="line_color",
            get_width=0.5,
            pickable=True
        ))

    # DT Layer (Transformers) as Fixed-Size Boxes
    if not df_dt.empty:
        layers.append(pdk.Layer(
            "PolygonLayer",
            df_dt,
            get_polygon="polygon",
            get_fill_color="[0, 255, 0, 200]", # Bright Green
            get_line_color="[0, 128, 0, 255]", # Darker Green Outline
            line_width_min_pixels=2,
            stroked=True,
            filled=True,
            pickable=True,
        ))
    
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v9",
        tooltip={
            "html": """
                <div style='font-family: sans-serif;'>
                    <div style='display: {Meter No2 ? "block" : "none"}'>
                        <b>Consumer:</b> {Name}<br/>
                        <b>Meter No:</b> {Meter No2}<br/>
                        <b>Feeder:</b> {Feeder Code}<br/>
                        <b>Load:</b> {load_kw} kW
                    </div>
                    <div style='display: {frompoleid ? "block" : "none"}'>
                        <b style='color: #ffa500'>{infra_type} ({n_voltage})</b><br/>
                        <b>From Pole:</b> {frompoleid}<br/>
                        <b>To Pole:</b> {topoleid}<br/>
                        <b>Feeder:</b> {feedername} ({feeder_id})
                    </div>
                    <div style='display: {dtccode ? "block" : "none"}'>
                        <b style='color: #00ff00'>Distribution Transformer (DT)</b><br/>
                        <b>DT Code:</b> {dtccode}<br/>
                        <b>DT Name:</b> {name}<br/>
                        <b>Connected Feeder:</b> <span style='color: #ffff00'>{feeder_id}</span><br/>
                        <b>Total Load:</b> {capacity:.2f} kW<br/>
                        <b>Meters connected:</b> {meter_count}
                    </div>
                </div>
            """,
            "style": {"color": "white", "backgroundColor": "#222", "fontSize": "12px"}
        }
    ), use_container_width=True)

# --- UI ---
st.title("📍 Raipur Metropolitan Smart Meter & Grid Mapper")
st.markdown("Mapping consumer endpoints and network infrastructure from CSV & PostgreSQL.")

# Initialize Session State for persistence
if 'df_oh' not in st.session_state: st.session_state.df_oh = pd.DataFrame()
if 'df_cable' not in st.session_state: st.session_state.df_cable = pd.DataFrame()
if 'df_dt' not in st.session_state: st.session_state.df_dt = pd.DataFrame()

# Sidebar Configuration
with st.sidebar:
    st.header("🔌 Database Settings (pgAdmin)")
    db_host = st.text_input("Host", value="localhost")
    db_port = st.text_input("Port", value="5433")
    db_name = st.text_input("Database Name", value="gis_master_db")
    db_user = st.text_input("Username", value="postgres")
    db_pass = st.text_input("Password", value="2020", type="password")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sync_oh = st.button("Sync OH Lines")
    with col2:
        sync_cable = st.button("Sync Cables")
    with col3:
        sync_dt = st.button("Sync DTs")
        
    if st.button("Clear Grid Data"):
        st.session_state.df_oh = pd.DataFrame()
        st.session_state.df_cable = pd.DataFrame()
        st.session_state.df_dt = pd.DataFrame()
        st.rerun()
    
    st.divider()
    st.header("📊 Filter & Metrics")
    st.info("Mapping individual coordinates from CSV records.")

csv_file = "Copy of E.E CITY DN NORTH RAIPUR__ 1.csv"

start_time = time.time()
with st.spinner("Processing Data..."):
    df = load_meter_data(csv_file)

if not df.empty:
    # Handle Smart Meter Processing
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    df = df.query("21.0 < LATITUDE < 21.5 and 81.4 < LONGITUDE < 81.8").copy()
    
    df['Sanctioned Load'] = pd.to_numeric(df['Sanctioned Load'], errors='coerce').fillna(0)
    df['unit'] = df['Sanctioned Load unit'].astype(str).str.upper().str.strip()
    df['load_kw'] = df['Sanctioned Load']
    df.loc[df['unit'] == 'MW', 'load_kw'] = df['Sanctioned Load'] * 1000
    df.loc[df['unit'] == 'W', 'load_kw'] = df['Sanctioned Load'] * 0.001

    # Database Integration
    if sync_oh or sync_cable or sync_dt:
        engine = get_db_connection(db_host, db_port, db_name, db_user, db_pass)
        if engine:
            unique_feeder_ids = df['Feeder Code'].unique()
            # We need Feeder Names for the cable table join
            csv_full = pd.read_csv(csv_file, usecols=['Feeder Code', 'Feeder Name'], low_memory=False)
            unique_feeder_names = csv_full[csv_full['Feeder Code'].isin(unique_feeder_ids)]['Feeder Name'].dropna().unique()

            if sync_oh:
                with st.spinner("Syncing OH Lines..."):
                    st.session_state.df_oh = load_infra_data(engine, unique_feeder_ids, unique_feeder_names, 'oh_line')
            if sync_cable:
                with st.spinner("Syncing Underground Cables..."):
                    st.session_state.df_cable = load_infra_data(engine, unique_feeder_ids, unique_feeder_names, 'rpr_cable')
            if sync_dt:
                with st.spinner("Aggregating Transformers (DTs) from CSV..."):
                    st.session_state.df_dt = derive_dt_data(df)

    # Combine for map (Infrastructure only)
    frames = []
    if not st.session_state.df_oh.empty: frames.append(st.session_state.df_oh)
    if not st.session_state.df_cable.empty: frames.append(st.session_state.df_cable)
    df_infra = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    load_time = time.time() - start_time
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Smart Meters", f"{len(df):,}")
    m2.metric("Connected Feeders", f"{df['Feeder Code'].nunique()}")
    
    total_assets = len(df_infra) + len(st.session_state.df_dt)
    if total_assets > 0:
        m3.metric("Grid Assets Mapped", total_assets)
    else:
        m3.metric("Data Process Time", f"{load_time:.2f}s")
    
    st.divider()
    
    # Main Visualization
    # increate the height of visualzation
    render_meter_map(df, df_infra, st.session_state.df_dt)
    
    with st.expander("Meter Detailed Registry"):
        st.dataframe(df[['Meter No2', 'Feeder Code', 'Consumer No', 'Name', 'load_kw']].head(1000))
    
    if not df_infra.empty:
        with st.expander("Infrastructure Master Records (DB)"):
            st.dataframe(df_infra[['infra_type', 'feedername', 'n_voltage', 'frompoleid', 'topoleid']].head(1000))
else:
    st.error("No valid Smart Meter coordinates found in the Raipur region.")
