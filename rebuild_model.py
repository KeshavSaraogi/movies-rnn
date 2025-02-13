import tensorflow as tf
from keras.models import load_model, Sequential
from keras.layers import SimpleRNN, Dense, Embedding
import h5py

# Define paths
old_model_path = "simpleRNN-IMDB-v2.h5"
new_model_path = "updated_model.keras"

# Load original model while ignoring unknown arguments
with h5py.File(old_model_path, "r") as f:
    if "model_config" in f.attrs:
        model_config = f.attrs["model_config"]
    else:
        raise ValueError("Could not find model config in the H5 file.")

# Convert model config to a new model
old_model = load_model(old_model_path, compile=False)
new_model = Sequential()

for layer in old_model.layers:
    if isinstance(layer, SimpleRNN):
        # Remove `time_major` argument if it exists
        new_model.add(SimpleRNN(layer.units, activation=layer.activation, return_sequences=layer.return_sequences))
    else:
        new_model.add(layer)

# Copy weights
new_model.set_weights(old_model.get_weights())

# Save in the new format
new_model.save(new_model_path)
print(f"New model saved as {new_model_path}")
