import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import geopandas as gpd
from shapely.geometry import Polygon
import h3
import contextily as ctx

# URLs to the data files in your GitHub repository
MASTER_URL = "https://raw.githubusercontent.com/krishm-htx/fdi-simulations/main/MasterGridObj.xlsx"
INSTANCES_URL = "https://raw.githubusercontent.com/krishm-htx/fdi-simulations/main/Instances_DATA.xlsx"

Methodology_URL = "https://raw.githubusercontent.com/krishm-htx/fdi-simulations/main/Methodology.png"
GIS_Steps1_URL = "https://raw.githubusercontent.com/krishm-htx/fdi-simulations/main/GIS_Steps1.png"
GIS_Steps2_URL = "https://raw.githubusercontent.com/krishm-htx/fdi-simulations/main/GIS_Steps2.png"

# Global password variable (can be replaced with environment variable for better security)
PASSWORD = "StormwaterPlanning@htx"

# Directory to store saved simulations
SAVE_DIR = "saved_simulations"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Authentication: Simple password protection
def authenticate():
    st.sidebar.header("Login")
    password = st.sidebar.text_input("Enter Password", type="password")
    
    if password == PASSWORD:
        return True
    else:
        st.sidebar.error("Incorrect Password")
        return False

# Function to dynamically adjust Ws and Wp sliders (Ws + Wp = 100)
def dynamic_sliders():
    ws = st.slider('Set Ws:', min_value=0, max_value=100, value=50, step=1)
    wp = 100 - ws
    st.write(f"Automatically adjusted Wp: {wp}")
    threshold_fdi = st.slider('Set FDI Threshold:', min_value=1.0, max_value=5.0, value=4.8, step=0.1)
    return ws, wp, threshold_fdi

# Function to calculate FDI
def calculate_fdi(W_s, I_s, I_p):
    W_p = 100 - W_s
    return (W_s * I_s + W_p * I_p) / 100

# Function to run FDI simulation
def run_simulation(df, W_s_range, threshold):
    df['FDI_Count'] = 0
    histogram_data = {}

    for index, row in df.iterrows():
        Is = row['Is']
        Ip = row['Ip']
        object_id = row['OBJECTID']

        count = 0
        for W_s in W_s_range:
            FDI = calculate_fdi(W_s, Is, Ip)
            if FDI > threshold:
                count += 1

        df.at[index, 'FDI_Count'] = count
        histogram_data[object_id] = count

    # Display histogram
    plt.figure(figsize=(10, 6))
    plt.bar(histogram_data.keys(), histogram_data.values(), color='skyblue')
    plt.xlabel('Object ID')
    plt.ylabel(f'FDI Frequency (Count of times FDI > {threshold})')
    plt.title('Histogram of FDI Frequency per Object')
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(plt)
    
    return df

# Function to merge with master file
def merge_with_master(fdi_df, master_df):
    merged_df = pd.merge(master_df, fdi_df[['GRID_ID', 'FDI_Count', 'Is', 'Ip']], on='GRID_ID', how='left')
    merged_df['FDI_Count'] = merged_df['FDI_Count'].fillna(0)
    merged_df['Is'] = merged_df['Is'].fillna(0)
    merged_df['Ip'] = merged_df['Ip'].fillna(0)
    return merged_df

# Function to plot hexagons on map
def plot_clusters_on_map(clustered_hexagons, top_10_hex):
    hex_polygons = []
    for hex_id in clustered_hexagons:
        hex_boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)
        hex_polygon = Polygon(hex_boundary)
        hex_polygons.append(hex_polygon)

    gdf = gpd.GeoDataFrame(geometry=hex_polygons, crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color='blue', alpha=0.5, edgecolor='black')

    # Highlight top 10 hexagons with highest FDI_Count
    for hex_id in top_10_hex:
        hex_boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)
        hex_polygon = Polygon(hex_boundary)
        gpd.GeoSeries([hex_polygon]).plot(ax=ax, color='red', edgecolor='black')

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    st.pyplot(fig)

# Function to save the simulation
def save_simulation(name, df, ws, wp, threshold, top_10_hex):
    save_path = os.path.join(SAVE_DIR, f"{name}.xlsx")
    df.to_excel(save_path, index=False)

    # Save metadata (Ws, Wp, threshold)
    meta_data = {"Ws": ws, "Wp": wp, "Threshold FDI": threshold, "Top 10 Hex": top_10_hex}
    with open(os.path.join(SAVE_DIR, f"{name}_meta.txt"), "w") as f:
        f.write(str(meta_data))

    st.success(f"Simulation '{name}' saved successfully.")

# Function to load saved simulations
def load_simulation():
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.xlsx')]
    if files:
        selected_file = st.selectbox("Select a saved simulation to load:", files)
        if st.button("Load Simulation"):
            loaded_df = pd.read_excel(os.path.join(SAVE_DIR, selected_file))
            with open(os.path.join(SAVE_DIR, f"{selected_file.replace('.xlsx', '_meta.txt')}"), "r") as f:
                meta_data = eval(f.read())
            st.write(f"Loaded simulation: {selected_file}")
            st.write(f"Ws: {meta_data['Ws']}, Wp: {meta_data['Wp']}, Threshold FDI: {meta_data['Threshold FDI']}")
            st.dataframe(loaded_df)
            
            # Show top 10 hexagons and plot the map
            top_10_hex = meta_data['Top 10 Hex']
            plot_clusters_on_map(loaded_df['GRID_ID'].tolist(), top_10_hex)

# Main Streamlit app function
def main():
    if authenticate():
        st.title("FDI Simulation and Clustering App")
        st.write("This app allows you to run FDI simulations and cluster H3 hexagons based on FDI values.")

        # Tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Import to ArcPro Help", "Saved Simulations", "Methodology"])

        # Import to ArcPro Help Tab
        with tab1:
            st.write("### Instructions to import the Excel file into ArcPro")
            st.image(ARC_HELP_IMG1, caption="Step 1: Load the file")
            st.image(ARC_HELP_IMG2, caption="Step 2: Import settings")

        # Saved Simulations Tab
        with tab2:
            st.write("### View and Load Saved Simulations")
            load_simulation()

        # Methodology Tab
        with tab3:
            st.write("### Methodology for FDI Calculations")
            st.image(METHODOLOGY_IMG, caption="FDI Calculation Methodology")

        # User inputs (Ws, Wp, threshold)
        ws, wp, threshold_fdi = dynamic_sliders()

        # Button to run simulation
        if st.button('Run FDI Simulation'):
            df = pd.read_excel("Instances_DATA.xlsx")  # Replace with actual file location
            master_df = pd.read_excel("MasterGridObj.xlsx")  # Replace with actual file location

            df = run_simulation(df, np.arange(ws, 101), threshold_fdi)
            merged_df = merge_with_master(df, master_df)

            # Calculate the top 10 hexagons by FDI_Count
            top_10_hex = df.nlargest(10, 'FDI_Count')['GRID_ID'].tolist()

            # Save the simulation
            sim_name = st.text_input("Enter a name for this simulation")
            if sim_name and st.button("Save Simulation"):
                save_simulation(sim_name, merged_df, ws, wp, threshold_fdi, top_10_hex)
                
            # Display the top 10 hexagons
            st.write("### Top 10 hexagons with highest FDI_Count")
            st.write(df.nlargest(10, 'FDI_Count')[['OBJECTID', 'GRID_ID', 'FDI_Count']])

            # Plot the hexagons on a map
            plot_clusters_on_map(df['GRID_ID'].tolist(), top_10_hex)


if __name__ == '__main__':
    main()
