import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import geopandas as gpd
from shapely.geometry import Polygon
import h3
#import contextily as ctx
import io
import folium
from streamlit_folium import folium_static
import requests
from io import BytesIO
import time
import json
import os
import hashlib
import time
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Set page config at the very beginning of the script
st.set_page_config(layout="wide", page_title="FDI Simulation App")

# Load your PDFs and data from GitHub or local repo
PDF_METHOD_PATH = "https://github.com/krishm-htx/fdi-simulations/raw/main/FDI-Sims-method.pdf"
PDF_HELP_PATH = "https://github.com/krishm-htx/fdi-simulations/raw/main/Excel_Import_to_ArcPro.pdf"
INSTANCES_FILE_PATH ="https://raw.githubusercontent.com/krishm-htx/fdi-simulations/main/Instances_DATA.xlsx"
MASTER_FILE_PATH = "https://raw.githubusercontent.com/krishm-htx/fdi-simulations/main/MasterGridObj.xlsx"

def plot_sensitivity_histogram(df, W_s, threshold):
    plt.figure(figsize=(10, 6))
    FDI = calculate_fdi(W_s, df['Is'], df['Ip'])
    bins = np.arange(0, max(FDI) + 0.2, 0.2)
    plt.hist(FDI, bins=bins, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel('FDI Value')
    plt.ylabel('Frequency')
    plt.title(f'FDI Distribution (W_s = {W_s}, W_p = {100-W_s}, Threshold = {threshold})')
    st.pyplot(plt)

def plot_clustered_hexagons(df, W_s, threshold):
    df['FDI'] = calculate_fdi(W_s, df['Is'], df['Ip'])
    df['cluster'] = (df['FDI'] > threshold).astype(int)
    
    # Convert H3 indices to lat/lon
    df['lat'], df['lon'] = zip(*df['GRID_ID'].apply(lambda x: h3.cell_to_latlng(x)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df['lon'], df['lat'], c=df['cluster'], cmap='coolwarm', alpha=0.7)
    ax.set_title(f'Clustered Hexagons (W_s = {W_s}, Threshold = {threshold})')
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(fig)

def plot_oat_sensitivity(df, parameter, parameter_range, fixed_params):
    results = []
    for param_value in parameter_range:
        if parameter == 'W_s':
            fixed_params['W_s'] = param_value
            FDI = calculate_fdi(fixed_params['W_s'], df['Is'], df['Ip'])
        else:  # threshold
            FDI = calculate_fdi(fixed_params['W_s'], df['Is'], df['Ip'])
        results.append(np.mean(FDI > param_value))
    
    plt.figure(figsize=(10, 6))
    plt.plot(parameter_range, results)
    plt.xlabel(parameter)
    plt.ylabel('Proportion of hexagons above threshold' if parameter == 'threshold' else 'Mean FDI')
    plt.title(f'One-at-a-Time Sensitivity Analysis for {parameter}')
    st.pyplot(plt)

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

def login():
    if not st.session_state.logged_in:
        st.subheader("Login Section")
        username = st.text_input("User Name")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            hashed_pswd = make_hashes('StormwaterPlanning@htx')
            if username in ["Krish", "Paresh", "Mayuri", "Jesse", "Jordan"] and check_hashes(password, hashed_pswd):
                st.session_state.logged_in = True
                st.success("Logged In as {}".format(username))
                st.rerun()
            else:
                st.warning("Incorrect Username/Password")
    return st.session_state.logged_in
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
    plt.figure(figsize=(6, 4))
    plt.bar(histogram_data.keys(), histogram_data.values(), color='skyblue')
    plt.xlabel('Object ID')
    plt.ylabel(f'Times FDI > {threshold}')
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
def plot_clusters_on_map(df_filtered):
    center_lat = df_filtered['lat'].mean()
    center_lon = df_filtered['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, width = "100%", height="600px")

    # Color scale for FDI count
    def get_color(fdi_count):
        return 'purple'

    for _, row in df_filtered.iterrows():
        hexagon = h3.cell_to_boundary(row['GRID_ID'])
        folium.Polygon(
            locations=hexagon,
            popup=f"FDI Count: {row['FDI_Count']}",
            color='black',
            weight=1,
            fill=True,
            fill_color=get_color(row['FDI_Count']),
            fill_opacity=0.65
        ).add_to(m)

    # Add a legend
    legend_html = '''
         <div style="position: fixed; 
                     bottom: 50px; left: 70px; width: 150px; height: 120px; 
                     border:2px solid grey; z-index:9999; font-size:14px;
                     ">&nbsp; FDI Count <br>
             &nbsp; <i class="fa fa-map-marker fa-2x" style="color:green"></i>&nbsp; 0-25 <br>
             &nbsp; <i class="fa fa-map-marker fa-2x" style="color:yellow"></i>&nbsp; 26-50 <br>
             &nbsp; <i class="fa fa-map-marker fa-2x" style="color:red"></i>&nbsp; 51+ 
         </div>
         '''
    m.get_root().html.add_child(folium.Element(legend_html))

    folium_static(m)

def find_clusters(hex_list, min_cluster_size=3):
    clusters = []
    visited = set()

    for hex_id in hex_list:
        if hex_id not in visited:
            cluster = set([hex_id])
            to_explore = [hex_id]

            while to_explore:
                current_hex = to_explore.pop()
                neighbors = h3.grid_disk(current_hex, 1)

                for neighbor in neighbors:
                    if neighbor in hex_list and neighbor not in visited:
                        visited.add(neighbor)
                        cluster.add(neighbor)
                        to_explore.append(neighbor)

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

    return clusters

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

# Load data
df, master_df = load_data()

def main():
    st.title('FDI Simulation App')

    # Check if the user is logged in
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Display login form if not logged in
    if not st.session_state.logged_in:
        login()
    else:

        # If we reach here, the user is logged in
        # Main app content
        with st.sidebar:
            st.header("Simulation Parameters", help = "With an increment of 1 we calculate the FDI for each weight combination in the specified range. And if the FDI exceeds the threshold we add it to a score.")
            
            w_structural = st.slider(
                "Weight of Structural Flooding Instances",
                0, 100, (50, 100), 
                help="Slide to set the range for the weight of structural flooding instances."
            )
            st.write(f"Weight of Population Flooding Instances: {100 - w_structural[1]} to {100 - w_structural[0]}")
            
            threshold = st.number_input(
                'FDI Threshold:', 
                value=4.8, 
                help="Set the threshold for FDI calculations."
            )
            
            st.info("Set the range of weights you want to run the simulation for and also the threshold of FDI that should be analysed.")
    
        # Create Tabs
        tab1, tab2, tab3 = st.tabs(["Run Simulation", "Docs", "Sensitivity Analysis"])
    
        # Load data
        df, master_df = load_data()
        if df is None or master_df is None:
            st.stop()
    
        # Tab 1: Run Simulation
        with tab1:
            st.header("Run FDI Simulation")
            st.write("Please adjust the simulation parameters in the sidebar and click 'Run Simulation' to start.")
            st.info("If you need help understanding the methodology or how to open these results in ArcPro, please refer to the 'Docs' tab.")
            
            if st.button("Run Simulation", key="run_sim"):
                with st.spinner("Running simulation..."):
                    W_s_range = np.arange(w_structural[0], w_structural[1] + 1)
                    df, histogram_data = run_simulation(df, W_s_range, threshold)
                    
                    # Simulate a delay to show the spinner
                    time.sleep(2)
                
                st.success("Simulation completed successfully!")
                
                # Display results in an expander
                with st.expander("View Detailed Results", expanded=False):
                    st.dataframe(df)
    
                # Plot histogram
                st.subheader("FDI Frequency Distribution")
                # Download simulation results
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                output.seek(0)
                st.download_button(
                    'Download Simulation Results', 
                    data=output, 
                    file_name='fdi_simulation_results.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
                with st.spinner("Generating histogram..."):
                    plot_histogram(histogram_data, threshold)
                    time.sleep(1) 
    
                # Find and display clusters
                hexes = df[df['FDI_Count'] > 1]['GRID_ID'].tolist()
                clusters = find_clusters(hexes)
    
                df['cluster'] = df['GRID_ID'].apply(
                    lambda x: 1 if any(x in cluster for cluster in clusters) else 0
                )
    
                df_filtered = df[df['cluster'] > 0].copy()
                df_filtered['lat'], df_filtered['lon'] = zip(*df_filtered['GRID_ID'].apply(lambda x: h3 .cell_to_latlng(x)))
    
                # Plot clusters on map
                st.subheader("Clustered Hexagons Over Houston, TX")
                # Download clusters
                output = io .BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_filtered.to_excel(writer, index=False)
                output.seek(0)
                st.download_button(
                    'Download Clusters', 
                    data=output, 
                    file_name='fdi_clusters.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
                
                with st.spinner("Generating map..."):
                    plot_clusters_on_map(df_filtered)
                    time.sleep(1)
                st.info("click on the hexagon to know the number of times the FDI threshold was crossed!")
            
        with tab2:
            st.header("Documentation")
            st.write("I would suggest downloading the pdf than reading the steps on here.")
            # Methodology PDF
            st.subheader("Methodology Documentation")
            with st.expander("View Methodology Description", expanded=False):
                st.write("Extract Flooding Data:\n\nObtain total structural and population flooding data for each hexagonal area. This data should be exported into an Excel file with relevant fields such as Hex ID, structural instances, and population instances.\n\nAssign Instance Factors:\n\nDistribute the data into equal ranges and assign instance factors for both structural flooding (Is) and population flooding (Ip).\n\nUpload Data:\n\nThe Excel file with structural instances, population instances, and Hex IDs is uploaded to the tool or software (possibly GIS or similar processing tool).\n\nFDI Calculation:\n\nThe Flood Damage Index (FDI) is calculated using the formula:\n\n𝐹𝐷𝐼 = 𝑊𝑠 × 𝐼𝑠 + 𝑊𝑝 × 𝐼𝑝\n\nWhere:\n\n𝑊𝑠 = Weight for structural instances (percentage, ranging from 0% to 100%)\n\n𝐼𝑠 = Structural flooding instance factor\n\n𝑊𝑝 = Weight for population instances (percentage, equal to 100% minus 𝑊𝑠)\n\n𝐼𝑝 = Population flooding instance factor\n\nSet Weights and Threshold:\n\nChoose a specific 𝑊𝑠 value (weight for structural flooding) and calculate the corresponding FDI for each hexagon. You can iterate through different 𝑊𝑠 values to determine the appropriate threshold for FDI that you want to focus on.\n\nCluster Identification:\n\nAfter calculating the FDI, group neighboring hexagons with high FDI values (greater than the threshold, e.g., FDI > 4.8) into clusters. This clustering step can help identify significant areas of flooding impact.\n\nSave and Run Scenarios:\n\nThe scenario is saved, and additional scenarios can be run with adjusted weights or different parameters.\n\nExample:\n\nIf you set 𝑊𝑠 = 50% and 𝑊𝑝 = 50%, and you have structural and population instance factors: 𝐼𝑠 = 5 and 𝐼𝑝 = 3, you would calculate the FDI for various weight configurations. For instance, with 𝑊𝑠 = 99% and 𝑊𝑝 = 1%, if FDI > 4.8, it passes the threshold.")
                st.download_button(
                    'Download Methodology Documentation', 
                    data=BytesIO(requests.get(PDF_METHOD_PATH).content), 
                    file_name='FDI-Sims-method.pdf', 
                    mime='application/pdf'
                )
            
            # Help PDF
            st.subheader("Import to ArcPro")
            with st.expander("View Import Instructions", expanded=False):
                st.write("Step 1: Copy the Layer H3_R9\nAction: In the Drawing Order panel, right-click the layer H3_R9.\nSelect: From the context menu, choose Copy.\nPaste: Right-click in an empty area and select Paste to create a duplicate of the H3_R9 layer.\n\nStep 2: Add New Data\nNavigate: In the ribbon at the top, under the Map tab, select Add Data.\nFile Path: Go to your download folder and select the file updated_FDI_results_with_master.\nSheet Selection: Choose Sheet 1$ from the file to import the data.\n\nStep 3: Join Data to H3_R9_copy\nLocate: Find the newly created copy H3_R9_copy in the Drawing Order.\nRight-click: On the H3_R9_copy layer, select Joins and Relates and click Add Join.\n\nStep 4: Set Join Fields\nInput Field: Set the Input Table as H3_R9_copy and the Input Field as OBJECT ID.\nJoin Table: Select Sheet1 (imported in Step 2) as the Join Table.\nJoin Field: Ensure the Join Field is OBJECTID in both tables.\n\nStep 5: Symbology\n Open Symbology: Go to the H3_R9_copy layer, right-click, and choose Symbology.\nChoose Symbology Type: Set the type to Graduated Colors.\nSet Field: In the field options, select Cluster_Assigned to apply color gradation based on clusters.\n\nStep 6: View Results\nAfter applying the symbology, the map will display hexagons color-coded based on the cluster assignments, showing the distribution of the data visually across the region.")
                st.download_button(
                    'Download Import Instructions', 
                    data=BytesIO(requests.get(PDF_HELP_PATH).content), 
                    file_name='Excel_Import_to_ArcPro.pdf', 
                    mime='application/pdf'
                )
        with tab3:
            st.header("Sensitivity Analysis")
            
            # Histogram section
            st.subheader("FDI Distribution for Different W_s Values")
            W_s_range = range(0, 101, 5)
            threshold = 4.8
            
            selected_W_s = st.select_slider("Select W_s value", options=W_s_range)
            plot_sensitivity_histogram(df, selected_W_s, threshold)
            
            # Clustered hexagons
            st.subheader("Clustered Hexagons")
            plot_clustered_hexagons(df, selected_W_s, threshold)
            
            # One-at-a-Time Sensitivity Analysis
            st.subheader("One-at-a-Time Sensitivity Analysis")
            parameter = st.selectbox("Select parameter for OAT analysis", ["W_s", "threshold"])
            if parameter == 'W_s':
                parameter_range = st.slider(f"Select range for {parameter}", 0, 100, (0, 100))
            else:  # threshold
                parameter_range = st.slider(f"Select range for {parameter}", 0, 10, (0, 10))
            fixed_params = {"W_s": 50, "threshold": 4.8}
            plot_oat_sensitivity(df, parameter, parameter_range, fixed_params)

if __name__ == "__main__":
    main()
