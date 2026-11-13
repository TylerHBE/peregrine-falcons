# data_visualization.py

# Imports
import pandas as pd
import numpy as np
import folium
import branca.colormap as cm

def display_long_map(long_heatmap_df, coordinate_df, file_name="index.html"):

  # Aggregate observations across all dates for initial display
  observation_counts = (
      long_heatmap_df.groupby(['x', 'y'])['count']
      .sum()
      .reset_index()
  )

  # Include every grid cell, including cells with no observations.
  display_df = coordinate_df.merge(
      observation_counts,
      on=['x', 'y'],
      how='left',
      validate='one_to_one'
  )
  display_df['count'] = display_df['count'].fillna(0).astype(int)

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
  m.save(file_name)

def display_map(display_df, file_name="index.html"):

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
  max_count = display_df['Count'].max()

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
          fill_color=colormap(row['Count']),
          fill_opacity=0.75,
          tooltip=(
              f"Grid: ({row['x']}, {row['y']})<br>"
              f"Observations: {row['Count']}<br>"
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
  m.save(f"./out/{file_name}")