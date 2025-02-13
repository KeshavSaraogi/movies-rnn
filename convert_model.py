import tensorflow as tf
from tensorflow.keras.models import load_model

# Try to load the model with a workaround for batch_shape error
try:
    model = load_model("simpleRNN-IMDB.h5", compile=False)
    print("✅ Model loaded successfully.")

    # Convert to SavedModel format
    model.save("simpleRNN-IMDB-savedmodel")
    print("✅ Model converted to TensorFlow SavedModel format.")

except TypeError as e:
    print("❌ Model loading failed due to batch_shape issue.")
    print(e)
