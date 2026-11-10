import pandas as pd
from data_visualization import display_map
import numpy as np

def get_dataframes():
    # Grab data
    file_address = "./data/out.csv"
    trans_addr = "./data/translation.csv"
    df = pd.read_csv(file_address)
    df_trans = pd.read_csv(trans_addr)
    return df, df_trans

def generate_comparison_html(p_file, g_file, output_file="comparison.html"):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Heatmap Comparison</title>
        <link rel="stylesheet" href="display.css">
    </head>
    <body>
        <div>
            <h2>Prediction Heatmap</h2>
            <iframe src="{p_file}" title="Prediction Heatmap"></iframe>
        </div>
        <div>
            <h2>Ground Truth Heatmap</h2>
            <iframe src="{g_file}" title="Ground Truth Heatmap"></iframe>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, "w") as f:
        f.write(html_content)

def make_heatmap_displays(model, df, transformer_df, X, y, number_of_predictions=1):
    # Generate prediction heatmaps for the specified number of predictions
    # select predictions randomly from the dataset
    random_indices = np.random.choice(X.index, size=number_of_predictions, replace=False)
    predictions = model.predict(X.loc[random_indices])

    copy = pd.DataFrame(columns=df.columns).drop(columns="Unnamed: 0")
    for i, index in enumerate(random_indices):
        year = X.loc[index, 'year']
        month = X.loc[index, 'month']
        copy.loc[len(copy)] = np.concatenate(([year, month], predictions[i]))

    copy = copy.melt(id_vars=['year', 'month'],var_name='pos', value_name='Count')
    copy['pos'] = pd.to_numeric(copy['pos'], errors='raise').astype('int64')
    copy = copy.merge(transformer_df, on="pos")

    # Create a new prediction + actual heatmap for each unique year and month combination
    for (year, month), group in copy.groupby(['year', 'month']):
        file_name_p = f"prediction_{year}_{month}.html"
        file_name_a = f"actual_{year}_{month}.html"
        display_map(group, file_name=file_name_p)
        # Create a heatmap for the actual data for the same year and month
        actual_group = df[(df['year'] == year) & (df['month'] == month)].drop(columns="Unnamed: 0")
        actual_group = actual_group.melt(id_vars=['year', 'month'],var_name='pos', value_name='Count')
        actual_group['pos'] = pd.to_numeric(actual_group['pos'], errors='raise').astype('int64')
        actual_group = actual_group.merge(transformer_df, on="pos")
        display_map(actual_group, file_name=file_name_a)
        generate_comparison_html(file_name_p, file_name_a, output_file=f"comparison_{year}_{month}.html")