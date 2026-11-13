# data_aggregation.py

#Imports
import pandas as pd
import numpy as np

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

def long_hashmap(name, df, std_dev_split):

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

def generate_wide(long_heatmap_df, coordinate_df):

    wide_heatmap_df = long_heatmap_df.pivot_table(
        index=['year', 'month'],
        columns=['x', 'y'],
        values='count',
        aggfunc='sum',
        fill_value=0
    )

    coordinate_index = pd.MultiIndex.from_frame(
        coordinate_df[['x', 'y']]
    )
    wide_heatmap_df = wide_heatmap_df.reindex(
        columns=coordinate_index,
        fill_value=0
    )

    wide_heatmap_df.columns = [
        f"{i}" for i in range(len(coordinate_df))
    ]

    wide_heatmap_df = wide_heatmap_df.reset_index()

    wide_heatmap_df.to_csv(path_or_buf="./data/out.csv")

    coordinate_df.rename_axis("pos").to_csv(path_or_buf="./data/translation.csv")