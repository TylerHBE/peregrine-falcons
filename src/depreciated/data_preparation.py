#Imports
import pandas as pd
import numpy as np
from src.depreciated.data_aggregation import long_hashmap,  generate_wide

# Grab data

def prep_main():

    file_address = "./data/shc.csv"

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

    std_dev_split = 5

    # Generate both DataFrames
    long_heatmap_df, coordinate_df = long_hashmap('Antigone canadensis', df, std_dev_split)

    generate_wide(long_heatmap_df, coordinate_df)
    long_heatmap_df.to_csv(path_or_buf="./data/long.csv")

    return long_heatmap_df, coordinate_df

long_heatmap_df, coordinate_df = prep_main()

co2_df_file_address = "./data/co2.csv"
co2_df = pd.read_csv(co2_df_file_address)
columns = ["datetime", "value", "latitude", "longitude"]
co2_df = co2_df[columns]

# generate a heatmap dataframe for CO2 data using the lng_min,lng_max,lat_min,lat_max columns from the coordinate_df
# specifically, for each row in the coordinate_df, we want to count the number of CO2 observations that fall within 
# the lng_min,lng_max,lat_min,lat_max bounds for that row. The resulting dataframe should have the same columns as 
# the coordinate_df, plus a "count" column that contains the average value of CO2 observations for each grid cell.
# Use pd.cut() to bin the CO2 observations into the grid cells defined by the coordinate_df, and then use groupby() to count the number of observations in each grid cell. Finally, merge the resulting counts back into the coordinate_df to create a new dataframe with the same structure as coordinate_df, but with an additional "count" column that contains the average value of CO2 observations for each grid cell.
co2_df['datetime'] = pd.to_datetime(co2_df['datetime'], errors='coerce')
# convert the latitude and longitude columns to numeric, coercing errors to NaN
co2_df['latitude'] = pd.to_numeric(co2_df['latitude'], errors='coerce')
co2_df['longitude'] = pd.to_numeric(co2_df['longitude'], errors='coerce')
co2_df = co2_df.dropna(subset=['datetime', 'value', 'latitude', 'longitude'])
co2_df = co2_df[(co2_df['latitude'].between(-90, 90)) & (co2_df['longitude'].between(-180, 180))]
co2_df['year'] = co2_df['datetime'].dt.year
co2_df['month'] = co2_df['datetime'].dt.month
co2_df = co2_df.rename(columns={"latitude": "lat", "longitude": "lng"})
# create bins for the CO2 data using the lng_min,lng_max,lat_min,lat_max columns from the coordinate_df
bins_lng = np.unique(np.concatenate([coordinate_df['lng_min'].values, coordinate_df['lng_max'].values]))
bins_lat = np.unique(np.concatenate([coordinate_df['lat_min'].values, coordinate_df['lat_max'].values]))
# use cut to bin the CO2 observations into the grid cells defined by the coordinate_df
# return lng_min, lng_max, lat_min, lat_max for each bin
co2_df['lng_bin'] = pd.cut(co2_df['lng'], bins=bins_lng, include_lowest=True)
co2_df['lat_bin'] = pd.cut(co2_df['lat'], bins=bins_lat, include_lowest=True)
co2_df = co2_df.dropna(subset=['lng_bin', 'lat_bin']).copy()
co2_df['lng_min'] = co2_df['lng_bin'].apply(lambda x: x.left)
co2_df['lng_max'] = co2_df['lng_bin'].apply(lambda x: x.right)
co2_df['lat_min'] = co2_df['lat_bin'].apply(lambda x: x.left)
co2_df['lat_max'] = co2_df['lat_bin'].apply(lambda x: x.right)
# merge 
# group by the bins and count the number of observations in each bin
co2_counts = co2_df.groupby(['lng_bin', 'lat_bin']).size().reset_index(name='count')

print(co2_df.head())