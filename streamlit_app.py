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
MASTER_URL = "https://raw.githubusercontent.com/your_username/your_repo/main/MasterGridObj.xlsx"
INSTANCES_URL = "https://raw.githubusercontent.com/your_username/your_repo/main/Instances_DATA.xlsx"

# Global password variable (can be replaced with environment variable for better security)
PASSWORD = "yourpassword123"

# Authentication: Simple password protection
def authenticate():
    st.sidebar.header("Login")
    password = st.sidebar.text_input("Enter Password", type="password")
    
    if password == PASSWORD:
        return True
    else:
        st.sidebar.error("Incorrect Password")
        return False

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
        grid_id = row['GRID_ID']

        count = 0
        for W_s in W_s_range:
            FDI = calculate_fdi(W_s, Is, Ip)
            if FDI > threshold:
                count += 1

        df.at[index, 'FDI_Count'] = count
        histogram_data[object_id] = count

    # Display histogram
    st.write("### FDI Histogram")
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
def plot_clusters_on_map(clustered_hexagons):
    hex_polygons = []
    for hex_id in clustered_hexagons:
        hex_boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)
        hex_polygon = Polygon(hex_boundary)
        hex_polygons.append(hex_polygon)

    gdf = gpd.GeoDataFrame(geometry=hex_polygons, crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color='blue', alpha=0.5, edgecolor='black')
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    st.pyplot(fig)

# Function to handle clustering logic
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

    # Plot clusters on map
    clustered_hexagons = df[df['Cluster_Assigned'] == 1]['GRID_ID'].tolist()
    plot_clusters_on_map(clustered_hexagons)

    # Download clusters
    output_file = BytesIO()
    df.to_excel(output_file, index=False, engine='openpyxl')
    output_file.seek(0)
    st.download_button(label="Download Clustered Data", data=output_file, file_name="clustered_data.xlsx")

# Main Streamlit app function
def main():
    if authenticate():
        st.title("FDI Simulation and Clustering App")
        st.write("This app allows you to run FDI simulations and cluster H3 hexagons based on FDI values.")
        
        # Read the Excel files directly from the repository
        st.write("Loading data from repository...")
        df = pd.read_excel(INSTANCES_URL)
        master_df = pd.read_excel(MASTER_URL)
        
        # User input section
        ws_min = st.slider('Start of Ws range:', min_value=0, max_value=100, value=50, step=1)
        threshold = st.number_input('FDI Threshold:', value=4.8)
        W_s_range = np.arange(ws_min, 101)
        
        # Run simulation button
        if st.button('Run FDI Simulation'):
            df = run_simulation(df, W_s_range, threshold)
            merged_df = merge_with_master(df, master_df)
            
            # Provide download link for FDI results
            output = BytesIO()
            merged_df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            st.download_button(label="Download FDI Results", data=output, file_name="FDI_results.xlsx")
            
            # Handle clustering and map display
            handle_cluster_download_and_display(df)

if __name__ == '__main__':
    main()
