import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from io import BytesIO
import geopandas as gpd
from shapely.geometry import Polygon
import h3
import contextily as ctx
import os
import requests 

# URLs to the data files in your GitHub repository
MASTER_URL = "https://raw.githubusercontent.com/krishm-htx/fdi-simulations/main/MasterGridObj.xlsx"
INSTANCES_URL = "https://raw.githubusercontent.com/krishm-htx/fdi-simulations/main/Instances_DATA.xlsx"

Methodology_URL = "https://github.com/krishm-htx/fdi-simulations/raw/main/FDI-Sims-method.pdf"
GIS_Steps_URL = "https://github.com/krishm-htx/fdi-simulations/raw/main/Excel_Import_to_ArcPro.pdf"

# Directory to store saved simulations
SAVE_DIR = "saved_simulations"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Function to dynamically adjust Ws and Wp sliders (Ws + Wp = 100)
def dynamic_sliders():
    ws_start = st.slider('Set Ws Start:', min_value=0, max_value=100, value=50, step=1)
    ws_end = st.slider('Set Ws End:', min_value=ws_start, max_value=100, value=100, step=1)
    wp_start = 100 - ws_end
    wp_end = 100 - ws_start
    st.write(f"Automatically adjusted Wp range: ({wp_start}, {wp_end})")  # Display Wp range
    threshold_fdi = st.slider('Set FDI Threshold:', min_value=1.0, max_value=5.0, value=4.8, step=0.1)
    return ws_start, ws_end, threshold_fdi

# Function to calculate FDI
def calculate_fdi(W_s, I_s, I_p):
    W_p = 100 - W_s
    return (W_s * I_s + W_p * I_p) / 100

# Function to run FDI simulation
def run_simulation(df, W_s_range, threshold): # Modified to accept W_s_range as argument
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

# Function to plot hexagons on map (Optimized) def hexagons_to_geodataframe(hex_ids):
    hex_polygons = []
    for hex_id in hex_ids:
        hex_boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)
        hex_polygon = Polygon(hex_boundary)
        hex_polygons.append(hex_polygon)

    gdf = gpd.GeoDataFrame(geometry=hex_polygons, crs="EPSG:4326")
    return gdf

def plot_clusters_on_map(clustered_hexagons):
    gdf = hexagons_to_geodataframe(clustered_hexagons)
    gdf = gdf.to_crs(epsg=3857)  # Reproject to Web Mercator

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot hexagons
    gdf.plot(ax=ax, color='blue', alpha=0.5, edgecolor='black')

    # Add basemap (OpenStreetMap)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Set title and show plot
    plt.title("Clustered H3 Hexagons Over Houston, TX")
    st.pyplot(plt)

# Function to handle file download for FDI results
def handle_file_download(merged_df):
    output_file = BytesIO()
    merged_df.to_excel(output_file, index=False)
    output_file.seek(0)
    st.download_button(label="Download FDI Results", data=output_file, file_name="updated_FDI_results_with_master.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Function to handle cluster download and display
def handle_cluster_download_and_display(df):
    df_filtered = df[df['FDI_Count'] > 0].copy()
    hexes_with_fdi = set(df_filtered['GRID_ID'])
    visited = set()

    def find_cluster(hex_id):
        cluster = []
        to_explore = [hex_id]
        while to_explore:
            current_hex = to_explore.pop()
            if current_hex not in visited:
                visited.add(current_hex)
                cluster.append(current_hex)
                neighbors = h3.k_ring(current_hex, 1)
                for neighbor in neighbors:
                    if neighbor in hexes_with_fdi and neighbor not in visited:
                        to_explore.append(neighbor)
        return cluster

    clusters = []
    for hex_id in hexes_with_fdi:
        if hex_id not in visited:
            cluster = find_cluster(hex_id)
            if len(cluster) >= 3:
                clusters.append(cluster)

    df['Cluster_Assigned'] = 0
    for cluster in clusters:
        df.loc[df['GRID_ID'].isin(cluster), 'Cluster_Assigned'] = 1

    # Plot clusters on the map
    clustered_hexagons = df[df['Cluster_Assigned'] == 1]['GRID_ID'].tolist()
    plot_clusters_on_map(clustered_hexagons)

    output_file = BytesIO()
    df.to_excel(output_file, index=False)
    output_file.seek(0)
    st.download_button(label="Download Clusters", data=output_file, file_name="updated_with_clusters.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Main Streamlit app function
def main():
    st.title("FDI Simulation App")
    st.write("Welcome to the FDI Simulation App!")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Run Simulations", "Saved Simulations", "Docs"])

    with tab1:
        st.write("### Run FDI Simulations")
        ws_start, ws_end, threshold_fdi = dynamic_sliders()
        W_s_range = np.arange(ws_start, ws_end + 1)  # Generate Ws range

        # Load data files
        master_df = pd.read_excel(requests.get(MASTER_URL, stream=True).raw)
        instances_df = pd.read_excel(requests.get(INSTANCES_URL, stream=True).raw)

        # Run simulation
        updated_df = run_simulation(instances_df, W_s_range, threshold_fdi)
        merged_df = merge_with_master(updated_df, master_df)

        # Display download buttons
        handle_file_download(merged_df)
        handle_cluster_download_and_display(updated_df)

    with tab2:
        st.write("### Saved Simulations")
        # TO DO: Implement saved simulations functionality

    with tab3:
        st.write("### Documentation")
        st.write("### Methodology")
        with st.expander("View PDF"):
            st.write(f'<iframe src="{Methodology_URL}" width="100%" height="600px"></iframe>', unsafe_allow_html=True)

        st.write("### Import to ArcPro Help")
        with st.expander("View PDF"):
            st.write(f'<iframe src="{GIS_Steps_URL}" width="100%" height="600px"></iframe>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
