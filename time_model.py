import pandas as pd
import numpy as np
import plotly.express as px

file_address = "./shc.csv"

df = pd.read_csv(file_address)
df['OBSERVATION DATE'] = pd.to_datetime(df['OBSERVATION DATE'])
df['OBSERVATION COUNT'] = df['OBSERVATION COUNT'].fillna(1)
for col in ["LATITUDE", "LONGITUDE", "OBSERVATION COUNT"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

std_dev_split = 10 # Split each std dev into 10 for granularity

# Function to generate a heatmap for each month_year combo
def long_heatmap(name, df):
  species = df[df['SCIENTIFIC NAME'] == name]
  species["month"] = species["OBSERVATION DATE"].dt.month
  species["year"] = species["OBSERVATION DATE"].dt.year
  x_step = species['LONGITUDE'].std() / std_dev_split
  x_custom_breaks = np.arange(species['LONGITUDE'].min(), species['LONGITUDE'].max() + x_step, x_step)
  species['lng_bin'] = pd.cut(
      species['LONGITUDE'],
      # labels = False, - gets rid of lat/lng specifics and orders each cell by index
      bins=x_custom_breaks
      )
  y_step = species['LATITUDE'].std() / std_dev_split
  y_custom_breaks = np.arange(species['LATITUDE'].min(), species['LATITUDE'].max() + y_step, y_step)
  species['lat_bin'] = pd.cut(species['LATITUDE'], bins=y_custom_breaks)
  heatmap_df = (
    species.groupby(['lng_bin', 'lat_bin', 'month', 'year'], observed=False)
    .size()
    .reset_index(name='count')
  )
  heatmap_df['lat_mid'] = heatmap_df['lat_bin'].apply(lambda x: x.mid).astype(float)
  heatmap_df['lng_mid'] = heatmap_df['lng_bin'].apply(lambda x: x.mid).astype(float)

  return heatmap_df

long_heatmap_df = long_heatmap('Antigone canadensis', df)

fig = px.density_map(
    long_heatmap_df, 
    lat='lat_mid', 
    lon='lng_mid', 
    z='count',         # <--- Tells plotly to weight the cells by your count column
    radius=35,         # Expand radius to let binned cells touch visually
    center=dict(lat=long_heatmap_df['lat_mid'].mean(), lon=long_heatmap_df['lng_mid'].mean()), 
    zoom=12,
    map_style=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
    ),
    color_continuous_scale="YlOrRd"
)

fig.write_html("index.html", auto_open=True)