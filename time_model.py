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

# Generate custom bin sizes
def create_bins(series, std_dev_split):
    """
    Generate coordinate bins based on standard deviation.
    Ensures that every coordinate, including the maximum,
    falls into a valid bin.
    """
    minimum = series.min()
    maximum = series.max()
    std = series.std()

    if minimum == maximum:
        padding = max(abs(minimum) * 0.001, 0.001)
        return np.linspace(
            minimum - padding, maximum + padding, 11
        )

    step = std / std_dev_split

    if not np.isfinite(step) or step <= 0:
        step = (maximum - minimum) / std_dev_split

    num_bins = max(1, int(np.ceil((maximum - minimum) / step)))

    # Explicitly extend the final boundary beyond the maximum
    return np.linspace(
        minimum,
        np.nextafter(maximum, np.inf),
        num_bins + 1
    )

# Function to generate transformed heatmap dataframe
def long_heatmap(name, df):

    species = df[df['SCIENTIFIC NAME'] == name].copy()

    species = species.dropna(
        subset=['OBSERVATION DATE', 'LATITUDE', 'LONGITUDE']
    )

    species = species[
        species['LATITUDE'].between(-90, 90) &
        species['LONGITUDE'].between(-180, 180)
    ].copy()

    if species.empty:
        raise ValueError(f"No valid observations for {name}")

    species["month"] = species["OBSERVATION DATE"].dt.month
    species["year"] = species["OBSERVATION DATE"].dt.year

    # Generate coordinate bins
    x_custom_breaks = create_bins(
        species['LONGITUDE'], std_dev_split
    )

    y_custom_breaks = create_bins(
        species['LATITUDE'], std_dev_split
    )

    species['lng_bin'] = pd.cut(
        species['LONGITUDE'],
        bins=x_custom_breaks,
        include_lowest=True
    )

    species['lat_bin'] = pd.cut(
        species['LATITUDE'],
        bins=y_custom_breaks,
        include_lowest=True
    )

    # Group by spatial bins and time
    heatmap_df = (
        species.groupby(
            ['lng_bin', 'lat_bin', 'month', 'year'],
            observed=True
        )
        .size()
        .reset_index(name='count')
    )

    # Calculate bin midpoints
    heatmap_df['lat_mid'] = heatmap_df['lat_bin'].apply(
        lambda x: x.mid
    ).astype(float)

    heatmap_df['lng_mid'] = heatmap_df['lng_bin'].apply(
        lambda x: x.mid
    ).astype(float)

    return heatmap_df

# Generate transformed dataframe
long_heatmap_df = long_heatmap(
    'Antigone canadensis',
    df
)


# ------------------------------------------
# VISUALIZATION
# ------------------------------------------

# Aggregate across months and years for display only.
# long_heatmap_df itself remains unchanged.
display_df = (
    long_heatmap_df.groupby(
        ['lng_bin', 'lat_bin'],
        observed=True
    )['count']
    .sum()
    .reset_index()
)

# Center map
center_lat = display_df['lat_bin'].apply(
    lambda x: float(x.mid)
).astype(float).mean()
center_lng = display_df['lng_bin'].apply(
    lambda x: float(x.mid)
).astype(float).mean()

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
    colors=['#ffffcc', '#ffeda0', '#feb24c',
            '#fd8d3c', '#f03b20', '#bd0026'],
    vmin=0,
    vmax=max_count
)

colormap.caption = "Observation count"
colormap.add_to(m)

# Draw each spatial bin as a colored rectangle
for _, row in display_df.iterrows():

    lat_bin = row['lat_bin']
    lng_bin = row['lng_bin']
    count = row['count']

    folium.Rectangle(
        bounds=[
            [lat_bin.left, lng_bin.left],
            [lat_bin.right, lng_bin.right]
        ],
        color=None,
        fill=True,
        fill_color=colormap(count),
        fill_opacity=0.75,
        tooltip=f"Observations: {count}"
    ).add_to(m)

# Automatically fit map to observation boundaries
min_lat = display_df['lat_bin'].apply(
    lambda x: x.left
).min()

max_lat = display_df['lat_bin'].apply(
    lambda x: x.right
).max()

min_lng = display_df['lng_bin'].apply(
    lambda x: x.left
).min()

max_lng = display_df['lng_bin'].apply(
    lambda x: x.right
).max()

m.fit_bounds([
    [min_lat, min_lng],
    [max_lat, max_lng]
])

# Save and open HTML
m.save("index.html")

import webbrowser
webbrowser.open("index.html")

print(long_heatmap_df.head())