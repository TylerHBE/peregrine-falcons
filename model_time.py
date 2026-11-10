import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"
import keras
from keras import layers
import numpy as np
from sklearn.model_selection import train_test_split

from model_tools import make_heatmap_displays, get_dataframes

print(f"Using Keras version: {keras.__version__}")

# Grab data
df, df_trans = get_dataframes()

input_columns = ["year","month"]

X = df[input_columns]
y = df.drop(columns=(["Unnamed: 0"] + input_columns))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize a sequential model
model = keras.Sequential([
    # Input layer: expects an array of 20 features
    layers.Input(shape=(input_columns.__len__(),)), 
    
    # Hidden layer 1: 64 neurons, ReLU activation function
    layers.Dense(64, activation="relu"), 
    
    # Hidden layer 2: 32 neurons, ReLU activation function
    layers.Dense(128, activation="relu"), 

    layers.Dense(256, activation="relu"), 
    
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
    epochs=20, 
    batch_size=32, 
    validation_data=(X_test, y_test)
)

# Evaluate model performance on validation data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

make_heatmap_displays(model, df, df_trans, X_test, y_test, number_of_predictions=5)