import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the same architecture (WITHOUT batch_shape)
max_features = 10000  # Make sure this matches what was used in training
max_len = 500

model = Sequential([
    Embedding(input_dim=max_features, output_dim=128, input_length=max_len),
    SimpleRNN(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model (necessary before loading weights)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the extracted weights
model.load_weights("simpleRNN-IMDB-weights.h5")

# Save the new model
model.save("simpleRNN-IMDB-v2.h5")

print("✅ Successfully recreated model and saved as 'simpleRNN-IMDB-v2.h5'")
