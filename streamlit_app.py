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
import requests
from io import BytesIO
import time
import plotly.express as px
import plotly.graph_objects as go

# Set page config at the very beginning of the script
st.set_page_config(layout="wide", page_title="FDI Simulation App")

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
def plot_clusters_on_map(df):
    fig = go.Figure(go.Choroplethmapbox(
        geojson=df['geometry'].__geo_interface__,
        locations=df.index,
        z=df['FDI_Count'],
        colorscale="RdYlGn_r",
        zmin=df['FDI_Count'].min(),
        zmax=df['FDI_Count'].max(),
        marker_opacity=0.5,
        marker_line_width=0
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=df['lat'].mean(), lon=df['lon'].mean()),
            zoom=10
        ),
        height=600,
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    st.plotly_chart(fig, use_container_width=True)

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

# Main Streamlit App st.title('FDI Simulation App')

# Create Tabs
#tab1, tab2, tab3 = st.tabs(["Run Simulation", "View Saved Results", "Methodology & Help"])

# Load data
df, master_df = load_data()

def main():
    # Initialize session state
    if 'reset' not in st.session_state:
        st.session_state.reset = False

    st.title('FDI Simulation App')

    # Sidebar for inputs
    with st.sidebar:
        st.header("Simulation Parameters")
        
        if st.session_state.reset:
            # Reset values
            w_structural = (50, 100)
            threshold = 4.8
            st.session_state.reset = False
        else:
            # Use current values
            w_structural = st.session_state.get('w_structural', (50, 100))
            threshold = st.session_state.get('threshold', 4.8)

        st.subheader("Weights")
        w_structural = st.slider(
            "Weight of Structural Flooding Instances (W_s)",
            0, 100, w_structural, 
            help="Slide to set the range for the weight of structural flooding instances.",
            key="w_structural"
        )
        st.write(f"Weight of Population Flooding Instances (W_p): {100 - w_structural[1]} to {100 - w_structural[0]}")
        
        threshold = st.number_input(
            'FDI Threshold:', 
            value=threshold, 
            help="Set the threshold for FDI calculations.",
            key="threshold"
        )
        
        if st.button("Reset Parameters"):
            st.session_state.reset = True
            st.rerun()
        
        st.info("Adjust these parameters and click 'Run Simulation' to start.")

    
    # Reset logic
    if st.session_state.reset:
        st.session_state.w_structural = (50, 100)
        st.session_state.threshold = 4.8
        st.session_state.reset = False
        st.experimental_rerun()

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["Run Simulation", "View Saved Results", "Documentation"])

    # Load data
    df, master_df = load_data()
    if df is None or master_df is None:
        st.stop()

    # Tab 1: Run Simulation
    with tab1:
        st.header("Run FDI Simulation")
        st.write("Please adjust the simulation parameters in the sidebar and click 'Run Simulation' to start.")
        st.info("If you need help understanding what this app does or how to open these results in ArcPro, please refer to the 'Documentation' tab.")
        
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
            plot_histogram(histogram_data, threshold)

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

            # Find and display clusters
            hexes = df[df['FDI_Count'] > 1]['GRID_ID'].tolist()
            clusters = find_clusters(hexes)

            df['cluster'] = df['GRID_ID'].apply(
                lambda x: 1 if any(x in cluster for cluster in clusters) else 0
            )

# Data preparation
            df_filtered = df[df['cluster'] > 0].copy()
            df_filtered['lat'], df_filtered['lon'] = zip(*df_filtered['GRID_ID'].apply(lambda x: h3.cell_to_lat_lng(x)))
        
            # Convert H3 cells to polygons
            df_filtered['geometry'] = df_filtered['GRID_ID'].apply(lambda h: {
                'type': 'Polygon',
                'coordinates': [h3.cell_to_boundary(h, geo_json=True)]
            })
        
            # Add download button for cluster Excel file
            st.subheader("Download Cluster Data")
            
            # Create a BytesIO object to store the Excel file
            excel_buffer = io.BytesIO()
            
            # Save the DataFrame to the BytesIO object
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_filtered.to_excel(writer, index=False, sheet_name='Cluster_Data')
            
            # Create a download button
            st.download_button(
                label="Download Cluster Excel File",
                data=excel_buffer.getvalue(),
                file_name="cluster_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
            st.subheader("FDI Count Distribution Over Houston, TX")
            plot_clusters_on_map(df_filtered)

    # Tab 2: View Saved Results
    with tab2:
        st.header("View Saved Results")
        saved_file = st.file_uploader("Upload saved simulation results", type=["xlsx"])
        if saved_file is not None:
            saved_df = pd.read_excel(saved_file)
            st.dataframe(saved_df)
    
            st.subheader("Histogram of Saved Results")
            plot_histogram(saved_df['FDI_Count'].to_dict(), threshold)

    # Tab 3: Documentation
    with tab3:
        st.header("Documentation")
        
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
            st.write("Step 1: Copy the Layer H3_R9\nAction: In the Drawing Order panel, right-click the layer H3_R9.\nSelect: From the context menu, choose Copy.\nPaste: Right-click in an empty area and select Paste to create a duplicate of the H3_R9 layer.\n\nStep 2: Add New Data\nNavigate: In the ribbon at the top, under the Map tab, select Add Data.\nFile Path: Go to your download folder and select the file updated_FDI_results_with_master.\nSheet Selection: Choose Sheet 1$ from the file to import the data.\n\nStep 3: Join Data to H3_R9_copy\nLocate: Find the newly created copy H3_R9_copy in the Drawing Order.\nRight-click: On the H3_R9_copy layer, select Joins and Relates and click Add Join.\n\nStep 4: Set Join Fields\nInput Field: Set the Input Table as H3_R9_copy and the Input Field as OBJECT ID.\nJoin Table: Select Sheet1 (imported in Step 2) as the Join Table.\nJoin Field: Ensure the Join Field is OBJECTID in both tables.\n\nStep 5: Symbology\nOpen Symbology: Go to the H3_R9_copy layer, right-click, and choose Symbology.\nChoose Symbology Type: Set the type to Graduated Colors.\nSet Field: In the field options, select Cluster_Assigned to apply color gradation based on clusters.\n\nStep 6: View Results\nAfter applying the symbology, the map will display hexagons color-coded based on the cluster assignments, showing the distribution of the data visually across the region.")
            st .download_button(
                'Download Import Instructions', 
                data=BytesIO(requests.get(PDF_HELP_PATH).content), 
                file_name='Excel_Import_to_ArcPro.pdf', 
                mime='application/pdf'
            )

if __name__ == "__main__":
    main()
