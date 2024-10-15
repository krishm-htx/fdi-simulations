import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
import h3
import contextily as ctx

# Load your PDFs and data from GitHub or local repo
PDF_METHOD_PATH = "https://github.com/krishm-htx/fdi-simulations/raw/main/FDI-Sims-method.pdf"
PDF_HELP_PATH = "https://github.com/krishm-htx/fdi-simulations/raw/main/Excel_Import_to_ArcPro.pdf"
INSTANCES_FILE_PATH ="https://raw.githubusercontent.com/krishm-htx/fdi-simulations/main/Instances_DATA.xlsx"
MASTER_FILE_PATH = "https://raw.githubusercontent.com/krishm-htx/fdi-simulations/main/MasterGridObj.xlsx"

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

    return df, histogram_data

# Function to create the histogram plot
def plot_histogram(histogram_data, threshold):
    plt.figure(figsize=(10, 6))
    plt.bar(histogram_data.keys(), histogram_data.values(), color='skyblue')
    plt.xlabel('Object ID')
    plt.ylabel(f'FDI Frequency (Count of times FDI > {threshold})')
    plt.title('Histogram of FDI Frequency per Object')
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(plt.gcf())

# Function to create a geodataframe of hexagons
def hexagons_to_geodataframe(hex_ids):
    hex_polygons = []
    for hex_id in hex_ids:
        # Convert hex_id to geo boundary (lat, lng pairs)
        hex_boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)  # This should work for the latest h3-py version
        hex_polygon = Polygon(hex_boundary)  # Convert to Shapely Polygon
        hex_polygons.append(hex_polygon)

    # Create a GeoDataFrame from the hexagons
    gdf = gpd.GeoDataFrame(geometry=hex_polygons, crs="EPSG:4326")
    return gdf

# Function to plot clustered hexagons on a map
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
    st.pyplot(fig)

# Function to cluster hexagons
def cluster_hexagons(df):
    clustered_hexagons = []
    for index, row in df.iterrows():
        hex_id = row['GRID_ID']
        fdi_count = row['FDI_Count']
        if fdi_count > 0:
            neighbors = []
            for other_index, other_row in df.iterrows():
                if other_index != index:
                    other_hex_id = other_row['GRID_ID']
                    # Use h3.k_ring_distances to check if they are neighbors (within distance 1)
                    k_ring = h3.k_ring_distances(hex_id, 1)  # Get hexagons within distance 1
                    if other_hex_id in k_ring[1]:  # k_ring[1] contains hexagons at distance 1
                        neighbors.append(other_hex_id)
            if len(neighbors) >= 2:  # at least 2 neighbors with FDI count > 0
                clustered_hexagons.append(hex_id)
    return clustered_hexagons



# Load the instances data and master data
@st.cache
def load_data():
    instances_df = pd.read_excel(INSTANCES_FILE_PATH)
    master_df = pd.read_excel(MASTER_FILE_PATH)
    return instances_df, master_df

# Main Streamlit App st.title('FDI Simulation App')

# Create Tabs
tab1, tab2, tab3 = st.tabs(["Run Simulation", "View Saved Results", "Methodology & Help"])

# Load data
df, master_df = load_data()

# Tab 1: Run Simulation
with tab1:
    st.header("Run FDI Simulation")

    # Sliders for W_s range: start point and end point
    ws_start = st.slider('Select W_s Start Point:', 0, 100, 50)
    ws_end = st.slider('Select W_s End Point:', ws_start, 100, 100)
    
    # Automatically calculate and display W_p as 100 - W_s
    wp_start = 100 - ws_start
    wp_end = 100 - ws_end
    st.write(f"W_p range: {wp_start} to {wp_end}")
    
    # Input for FDI threshold
    threshold = st.number_input('Set FDI Threshold:', value=4.8)

    # Run simulation when button is clicked
    if st.button("Run Simulation"):
        W_s_range = np.arange(ws_start, ws_end + 1)
        df, histogram_data = run_simulation(df, W_s_range, threshold)
        
        # Display results
        st.write(df)
        plot_histogram(histogram_data, threshold)

        # Download simulation results
        output_file = 'fdi_simulation_results.xlsx'
        df.to_excel(output_file, index=False)
        st.download_button('Download Simulation Results', data=open(output_file, 'rb'), file_name=output_file)

        # Optionally, plot clusters and allow download
        clustered_hexagons = cluster_hexagons(df)
        plot_clusters_on_map(clustered_hexagons)

# Tab 2: View Saved Results
with tab2:
    st.header("View Saved Results")
    saved_file = st.file_uploader("Upload saved simulation results", type=["xlsx"])
    if saved_file is not None:
        saved_df = pd.read_excel(saved_file)
        st.write(saved_df)

        # Plot histogram and clusters from saved results
        st.write("Histogram of Saved Results")
        plot_histogram(saved_df['FDI_Count'].to_dict(), threshold)

# Tab 3: Methodology & Help
with tab3:
    st.header("Documentation")
    st.write("Download the following PDFs for more information:")
    
    st.write("[Methodology PDF](%s)" % PDF_METHOD_PATH)
    st.write("[Help PDF for Importing to ArcGIS Pro](%s)" % PDF_HELP_PATH)
