from tensorflow.keras.models import load_model

# Load the modified H5 model (now without batch_shape)
model = load_model("simpleRNN-IMDB.h5")  # Should work now

# Save a new clean model file
model.save("simpleRNN-IMDB-v2.h5")

print("✅ New model saved as simpleRNN-IMDB-v2.h5")
