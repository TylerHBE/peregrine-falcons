import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"
import keras
from keras import layers
import numpy as np

print(f"Using Keras version: {keras.__version__}")

# Grab data
file_address = "./out.csv"
df = pd.read_csv(file_address)

