import h5py

# Open the H5 file
h5_path = "simpleRNN-IMDB.h5"

with h5py.File(h5_path, "r+") as f:
    # Locate the InputLayer configuration
    layers = f["model_weights"]
    
    # Check for batch_shape key and remove it
    for layer in layers.keys():
        if "batch_shape" in layers[layer].attrs:
            print(f"Removing batch_shape from layer: {layer}")
            del layers[layer].attrs["batch_shape"]

# Save the modified model
print("✅ Batch_shape attribute removed successfully!")
