import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"
import keras
from keras import layers
import numpy as np
from sklearn.model_selection import train_test_split

from src.depreciated.model_tools import make_heatmap_displays, get_dataframes

print(f"Using Keras version: {keras.__version__}")

# Grab data
df, df_trans = get_dataframes()

# transform the dataframe - every row will include the previous month's heatmap data, and the current month's heatmap data will be the target variable
transformed_df = df.copy()

base_columns = ["year","month"]
target_columns = [col for col in df.columns if col not in (["Unnamed: 0"] + base_columns)]
added_columns = [f"p_{col}" for col in target_columns]
# Add the previous month's heatmap data to the dataframe - must thus remove the first row, as it will have NaN values for the previous month's data
transformed_df[added_columns] = transformed_df[target_columns].shift(1)
transformed_df = transformed_df.dropna()
input_columns = base_columns + added_columns

X = transformed_df[input_columns]
y = transformed_df[target_columns]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize a sequential model
model = keras.Sequential([
    # Input layer: expects an array of 20 features
    layers.Input(shape=(input_columns.__len__(),)), 
    
    # Hidden layer 1: 64 neurons, ReLU activation function
    layers.Dense(4000, activation="relu"), 
    
    # Hidden layer 2: 32 neurons, ReLU activation function
    layers.Dense(4000, activation="relu"), 

    layers.Dense(4000, activation="relu"), 
    
    # Output layer: 1 neuron with Sigmoid activation for binary probability (0 to 1)
    layers.Dense(y.columns.__len__(), activation="relu") 
])

print(
    # View a breakdown of the network layers and parameters
    model.summary()
)

model.compile(
    optimizer="adam",                      # Adaptive Moment Estimation optimizer
    loss="mean_squared_error",            # Standard loss for binary classification
    metrics=["accuracy"]                   # Track accuracy during training
)

history = model.fit(
    X_train, 
    y_train, 
    epochs=40, 
    batch_size=32, 
    validation_data=(X_test, y_test)
)

# Evaluate model performance on validation data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

make_heatmap_displays(model, df, df_trans, X_test, y_test, number_of_predictions=5)