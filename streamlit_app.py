import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
import h3
import contextily as ctx
import io
import folium
from streamlit_folium import folium_static

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
        hex_boundary = h3.cell_to_boundary(hex_id)
        hex_polygon = Polygon(hex_boundary)
        hex_polygons.append(hex_polygon)

    gdf = gpd.GeoDataFrame(geometry=hex_polygons, crs="EPSG:4326")
    return gdf

# Function to plot clustered hexagons on a map
def plot_clusters_on_map(clustered_hexagons):
    # Create a map centered on Houston
    m = folium.Map(location=[29.7604, -95.3698], zoom_start=10)

    # Create a GeoDataFrame of hexagons
    gdf = hexagons_to_geodataframe(clustered_hexagons)

    # Add hexagons to the map
    for _, row in gdf.iterrows():
        folium.GeoJson(
            row['geometry'],
            style_function=lambda x: {
                'fillColor': 'blue',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.5,
            }            
        ).add_to(m)

    # Display the map
    m

def cluster_hexagons(df):
    clustered_hexagons = []
    hexagons_checked = set()  # To avoid reprocessing already clustered hexagons
    
    # Filter for hexagons with FDI_Count > 1
    df_filtered = df[df['FDI_Count'] > 1].copy()
    
    for index, row in df_filtered.iterrows():
        hex_id = row['GRID_ID']
        
        # Skip already checked hexagons
        if hex_id in hexagons_checked:
            continue
        
        # Find neighbors for the current hexagon
        neighbors = []
        nearby_hexagons = h3.grid_disk(hex_id, 1)  # Get neighboring hexagons

        for other_index, other_row in df_filtered.iterrows():
            other_hex_id = other_row['GRID_ID']
            
            # Check if the other hexagon is a neighbor (distance 1)
            if other_index != index and other_hex_id in nearby_hexagons:
                neighbors.append(other_hex_id)
        
        # If it has 2 or more neighbors, it's a cluster
        if len(neighbors) >= 2:
            clustered_hexagons.append(hex_id)
            hexagons_checked.update(neighbors)  # Mark neighbors as checked
    return clustered_hexagons

# Load the instances data and master data
@st.cache_data
def load_data():
    try:
        instances_df = pd.read_excel(INSTANCES_FILE_PATH)
        master_df = pd.read_excel(MASTER_FILE_PATH)
        return instances_df, master_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Main Streamlit App st.title('FDI Simulation App')

# Create Tabs
tab1, tab2, tab3 = st.tabs(["Run Simulation", "View Saved Results", "Methodology & Help"])

# Load data
df, master_df = load_data()

def main():
    st.title('FDI Simulation App')

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["Run Simulation", "View Saved Results", "Methodology & Help"])

    # Load data
    df, master_df = load_data()
    if df is None or master_df is None:
        st.stop()

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
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)
            st.download_button('Download Simulation Results', data=output, file_name='fdi_simulation_results.xlsx')

            # Optionally, plot clusters and allow download
            clustered_hexagons = cluster_hexagons(df)
            if clustered_hexagons:
                st.subheader("Clustered Hexagons Over Houston, TX")
                plot_clusters_on_map(clustered_hexagons)
            else:
                st.write("No clusters found based on current criteria.")
    
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
        
if __name__ == "__main__":
    main()
