# data_aggregation.py

#Imports
import pandas as pd
import numpy as np
import folium
import branca.colormap as cm

# Grab data
file_address = "./shc.csv"
df = pd.read_csv(file_address)

# Clean data
df['OBSERVATION DATE'] = pd.to_datetime(
    df['OBSERVATION DATE'], errors='coerce'
)
df['OBSERVATION COUNT'] = pd.to_numeric(
    df['OBSERVATION COUNT'], errors='coerce'
).fillna(1)
for col in ["LATITUDE", "LONGITUDE"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

std_dev_split = 10

def create_bins(series, std_dev_split):
    minimum = series.min()
    maximum = series.max()
    std = series.std()

    if minimum == maximum:
        padding = max(abs(minimum) * 0.001, 0.001)
        return np.linspace(
            minimum - padding,
            maximum + padding,
            11
        )

    step = std / std_dev_split

    if not np.isfinite(step) or step <= 0:
        step = (maximum - minimum) / std_dev_split

    num_bins = max(
        1,
        int(np.ceil((maximum - minimum) / step))
    )

    return np.linspace(
        minimum,
        np.nextafter(maximum, np.inf),
        num_bins + 1
    )

def long_hashmap(name, df, std_dev_split=10):

    species = df[
        df['SCIENTIFIC NAME'] == name
    ].copy()

    species = species.dropna(
        subset=['OBSERVATION DATE', 'LATITUDE', 'LONGITUDE']
    )

    species = species[
        species['LATITUDE'].between(-90, 90) &
        species['LONGITUDE'].between(-180, 180)
    ].copy()

    if species.empty:
        raise ValueError(f"No valid observations for {name}")

    species['month'] = species['OBSERVATION DATE'].dt.month
    species['year'] = species['OBSERVATION DATE'].dt.year

    # Generate bin boundaries
    lng_edges = create_bins(
        species['LONGITUDE'], std_dev_split
    )

    lat_edges = create_bins(
        species['LATITUDE'], std_dev_split
    )

    # Assign integer indices instead of interval labels
    species['x'] = pd.cut(
        species['LONGITUDE'],
        bins=lng_edges,
        labels=False,
        include_lowest=True
    )

    species['y'] = pd.cut(
        species['LATITUDE'],
        bins=lat_edges,
        labels=False,
        include_lowest=True
    )

    species = species.dropna(subset=['x', 'y']).copy()

    species['x'] = species['x'].astype(int)
    species['y'] = species['y'].astype(int)

    # Abstracted heatmap dataframe
    heatmap_df = (
        species.groupby(
            ['x', 'y', 'month', 'year'],
            observed=True
        )
        .size()
        .reset_index(name='count')
    )

    # Build geographic lookup table
    # Every possible x/y combination is represented.
    coordinate_df = pd.MultiIndex.from_product(
        [
            range(len(lng_edges) - 1),
            range(len(lat_edges) - 1)
        ],
        names=['x', 'y']
    ).to_frame(index=False)

    # Translate integer indices into geographic boundaries
    coordinate_df['lng_min'] = coordinate_df['x'].map(
        lambda x: lng_edges[x]
    )

    coordinate_df['lng_max'] = coordinate_df['x'].map(
        lambda x: lng_edges[x + 1]
    )

    coordinate_df['lat_min'] = coordinate_df['y'].map(
        lambda y: lat_edges[y]
    )

    coordinate_df['lat_max'] = coordinate_df['y'].map(
        lambda y: lat_edges[y + 1]
    )

    # Geographic center of each bin
    coordinate_df['lat'] = (
        coordinate_df['lat_min'] +
        coordinate_df['lat_max']
    ) / 2

    coordinate_df['lng'] = (
        coordinate_df['lng_min'] +
        coordinate_df['lng_max']
    ) / 2

    return heatmap_df, coordinate_df

# Generate both DataFrames
long_heatmap_df, coordinate_df = long_hashmap(
    'Antigone canadensis',
    df,
    std_dev_split=10
)

def display_map(long_heatmap_df, coordinate_df):

  # Aggregate observations across all dates for initial display
  display_df = (
      long_heatmap_df.groupby(['x', 'y'])['count']
      .sum()
      .reset_index()
  )

  # Translate abstracted coordinates into geographic coordinates
  display_df = display_df.merge(
      coordinate_df,
      on=['x', 'y'],
      how='left',
      validate='one_to_one'
  )

  # Initialize map
  center_lat = display_df['lat'].mean()
  center_lng = display_df['lng'].mean()

  m = folium.Map(
      location=[center_lat, center_lng],
      zoom_start=4,
      tiles=(
          "https://server.arcgisonline.com/ArcGIS/rest/services/"
          "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
      ),
      attr="Tiles © Esri",
      control_scale=True
  )

  # Color scale
  max_count = display_df['count'].max()

  colormap = cm.LinearColormap(
      colors=[
          '#ffffcc',
          '#ffeda0',
          '#feb24c',
          '#fd8d3c',
          '#f03b20',
          '#bd0026'
      ],
      vmin=0,
      vmax=max_count
  )

  colormap.caption = 'Observation count'
  colormap.add_to(m)

  # Draw rectangles using translated geographic boundaries
  for _, row in display_df.iterrows():

      folium.Rectangle(
          bounds=[
              [row['lat_min'], row['lng_min']],
              [row['lat_max'], row['lng_max']]
          ],
          color=None,
          fill=True,
          fill_color=colormap(row['count']),
          fill_opacity=0.75,
          tooltip=(
              f"Grid: ({row['x']}, {row['y']})<br>"
              f"Observations: {row['count']}<br>"
              f"Latitude: {row['lat']:.4f}<br>"
              f"Longitude: {row['lng']:.4f}"
          )
      ).add_to(m)

  # Fit map to geographic boundaries
  m.fit_bounds([
      [
          display_df['lat_min'].min(),
          display_df['lng_min'].min()
      ],
      [
          display_df['lat_max'].max(),
          display_df['lng_max'].max()
      ]
  ])

  # Export
  m.save('index.html')

  import webbrowser
  webbrowser.open('index.html')

def generate_wide(long_heatmap_df):

    wide_heatmap_df = long_heatmap_df.pivot_table(
        index=['year', 'month'],
        columns=['x', 'y'],
        values='count',
        aggfunc='sum',
        fill_value=0
    )

    wide_heatmap_df.columns = [
        f"{i}" for i in range(0, wide_heatmap_df.columns.__len__())
    ]

    wide_heatmap_df = wide_heatmap_df.reset_index()

    # display_map(df_2011, coordinate_df)
    wide_heatmap_df.to_csv(path_or_buf="out.csv")