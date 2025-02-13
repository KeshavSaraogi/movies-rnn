import h5py

# Open the H5 file in read+write mode
h5_path = "simpleRNN-IMDB.h5"

with h5py.File(h5_path, "r") as f:
    print("✅ Model structure:")
    print(list(f.keys()))  # List of model layers

    # Extract and save only model weights
    with h5py.File("simpleRNN-IMDB-weights.h5", "w") as weights_file:
        f.copy("model_weights", weights_file)

print("✅ Model weights saved successfully as 'simpleRNN-IMDB-weights.h5'")
