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
def plot_clusters_on_map(df_filtered):
    center_lat = df_filtered['lat'].mean()
    center_lon = df_filtered['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # Color scale for FDI count
    def get_color(fdi_count):
        if fdi_count <= 25:
            return 'green'
        elif fdi_count <= 50:
            return 'yellow'
        else:
            return 'red'

    for _, row in df_filtered.iterrows():
        hexagon = h3.cell_to_boundary(row['GRID_ID'])
        folium.Polygon(
            locations=hexagon,
            popup=f"Cluster: {row['cluster']}<br>FDI Count: {row['FDI_Count']}",
            color='black',
            weight=1,
            fill=True,
            fill_color=get_color(row['FDI_Count']),
            fill_opacity=0.7
        ).add_to(m)

    # Add a legend
    legend_html = '''
         <div style="position: fixed; 
                     bottom: 50px; left: 50px; width: 120px; height: 90px; 
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

# Main Streamlit App st.title('FDI Simulation App')

# Create Tabs
#tab1, tab2, tab3 = st.tabs(["Run Simulation", "View Saved Results", "Methodology & Help"])

# Load data
df, master_df = load_data()

def main():
    st.title('FDI Simulation App')

    # Sidebar for inputs
    with st.sidebar:
        st.header("Simulation Parameters")
        
        st.subheader("Weights")
        w_structural = st.slider(
            "Weight of Structural Flooding Instances (W_s)",
            0, 100, (50, 100), 
            help="Slide to set the range for the weight of structural flooding instances."
        )
        st.write(f"Weight of Population Flooding Instances (W_p): {100 - w_structural[1]} to {100 - w_structural[0]}")
        
        threshold = st.number_input(
            'FDI Threshold:', 
            value=4.8, 
            help="Set the threshold for FDI calculations."
        )
        
        st.info("Adjust these parameters and click 'Run Simulation' to start.")

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["Run Simulation", "View Saved Results", "Methodology & Help"])

    # Load data
    df, master_df = load_data()
    if df is None or master_df is None:
        st.stop()

    # Tab 1: Run Simulation
    with tab1:
        st.header("Run FDI Simulation")
        st.write("Please adjust the simulation parameters in the sidebar and click 'Run Simulation' to start.")
        
        if st.button("Run Simulation", key="run_sim"):
            with st.spinner("Running simulation..."):
                W_s_range = np.arange(w_structural[0], w_structural[1] + 1)
                df, histogram_data = run_simulation(df, W_s_range, threshold)
            
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

            df_filtered = df[df['cluster'] > 0]
            df_filtered['lat'], df_filtered['lon'] = zip(*df_filtered['GRID_ID'].apply(lambda x: h3.cell_to_latlng(x)))

            st.subheader("Clustered Hexagons Over Houston, TX")
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

    # Tab 3: Methodology & Help
    with tab3:
        st.header("Documentation")
        
        # Methodology PDF
        st.subheader("Methodology Documentation")
        try:
            response = requests.get("https://github.com/krishm-htx/fdi-simulations/raw/main/FDI-Sims-method.pdf")
            if response.status_code == 200:
                st.download_button(
                    'Download Methodology Documentation', 
                    data=BytesIO(response.content), 
                    file_name='FDI-Sims-method.pdf', 
                    mime='application/pdf'
                )
            else:
                st.error("Failed to fetch the methodology documentation. Please try again later.")
        except Exception as e:
            st.error(f"An error occurred while fetching the methodology documentation: {str(e)}")
        
        # Help PDF
        st.subheader("Help Documentation")
        try:
            response = requests.get("https://github.com/krishm-htx/fdi-simulations/raw/main/Excel_Import_to_ArcPro.pdf")
            if response.status_code == 200:
                st.download_button(
                    'Download Help Documentation', 
                    data=BytesIO(response.content), 
                    file_name='Excel_Import_to_ArcPro.pdf', 
                    mime='application/pdf'
                )
            else:
                st.error("Failed to fetch the help documentation. Please try again later.")
        except Exception as e:
            st.error(f"An error occurred while fetching the help documentation: {str(e)}")
        
if __name__ == "__main__":
    main()
