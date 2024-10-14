import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
from google.colab import files  # For file downloads in Colab
import h3
import geopandas as gpd
from shapely.geometry import Polygon
import contextily as ctx


# Upload the file
master = files.upload()

# Load the Excel file into a DataFrame
file_name = list(master.keys())[0]  # Get the uploaded file name
df = pd.read_excel(file_name)

#df.head()  # Preview the first few rows
# Function to calculate FDI
def calculate_fdi(W_s, I_s, I_p):
    W_p = 100 - W_s
    return (W_s * I_s + W_p * I_p) / 100

# Function to run FDI simulation
def run_simulation(W_s_range, threshold):
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

    # Display the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(histogram_data.keys(), histogram_data.values(), color='skyblue')
    plt.xlabel('Object ID')
    plt.ylabel(f'FDI Frequency (Count of times FDI > {threshold})')
    plt.title('Histogram of FDI Frequency per Object')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    above_threshold = df[df['FDI_Count'] > threshold]
    return df, above_threshold

# Function to merge with the master file and prepare for download
def merge_with_master(fdi_df, master_df):
    merged_df = pd.merge(master_df, fdi_df[['GRID_ID', 'FDI_Count', 'Is', 'Ip']], on='GRID_ID', how='left')
    merged_df['FDI_Count'] = merged_df['FDI_Count'].fillna(0)
    merged_df['Is'] = merged_df['Is'].fillna(0)
    merged_df['Ip'] = merged_df['Ip'].fillna(0)
    return merged_df

# Function to plot H3 hexagons on a map
def hexagons_to_geodataframe(hex_ids):
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
    plt.show()

# Function to handle file download for FDI results
def handle_file_download(merged_df):
    output_file = 'updated_FDI_results_with_master.xlsx'
    merged_df.to_excel(output_file, index=False)

    download_button_fdi = widgets.Button(description="Download FDI Results")

    def on_fdi_download_clicked(b):
        files.download(output_file)

    download_button_fdi.on_click(on_fdi_download_clicked)
    display(download_button_fdi)

# Function to handle cluster download and display
def handle_cluster_download_and_display():
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

    output_file = "updated_with_clusters.xlsx"
    df.to_excel(output_file, index=False)

    download_button_clusters = widgets.Button(description="Download Clusters")

    def on_clusters_download_clicked(b):
        files.download(output_file)

    download_button_clusters.on_click(on_clusters_download_clicked)
    display(download_button_clusters)

# Function to get user input and run simulations interactively
def interactive_fdi_simulation():
    ws_min = widgets.IntSlider(value=50, min=0, max=100, step=1, description='Start of Ws:')
    threshold = widgets.FloatText(value=4.8, description='Threshold:')
    run_button = widgets.Button(description='Run Simulation')

    display(ws_min, threshold, run_button)

    def on_run_button_clicked(b):
        W_s_range = np.arange(ws_min.value, 101)
        threshold_value = threshold.value
        updated_df, above_threshold = run_simulation(W_s_range, threshold_value)
        merged_df = merge_with_master(updated_df, master)

        # After histogram, show the download buttons for FDI results
        handle_file_download(merged_df)

        # Display clusters on map and provide cluster download button
        handle_cluster_download_and_display()

        run_again = widgets.Button(description='SIMULATE AGAIN')
        done_button = widgets.Button(description='Done')

        def on_run_again_clicked(b):
            clear_output()
            interactive_fdi_simulation()

        def on_done_clicked(b):
            clear_output()
            print("Simulation complete.")

        display(run_again, done_button)

    run_button.on_click(on_run_button_clicked)

# Start of the workflow: Upload file and load it into a DataFrame
print("Upload your Excel file:")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_excel(file_name)

# Load the master file as well
master = pd.read_excel('MasterGridObj.xlsx')

# Start the interactive simulation
interactive_fdi_simulation()
