import tensorflow as tf
from keras.models import load_model, model_from_json
import json

# Load the existing model
model_path = "simpleRNN-IMDB-v2.h5"
model = load_model(model_path, compile=False)

# Convert model to JSON format
model_config = model.to_json()
model_config_dict = json.loads(model_config)

# Print model configuration for debugging
print("\n🔍 **BEFORE MODIFICATION:**")
print(json.dumps(model_config_dict, indent=2))  # Pretty print full config

# Iterate over layers and remove `time_major` if it exists
for layer in model_config_dict['config']['layers']:
    if 'time_major' in layer['config']:
        print(f"🛑 Found 'time_major' in {layer['class_name']} - Removing it!")
        del layer['config']['time_major']

# Print model configuration after modification
print("\n✅ **AFTER MODIFICATION:**")
print(json.dumps(model_config_dict, indent=2))

# Convert back to JSON and rebuild the model
updated_model_config_json = json.dumps(model_config_dict)
updated_model = model_from_json(updated_model_config_json)

# Save the updated model in a new format
updated_model.save("updated_model.keras", save_format="keras")

print("\n✅ Model successfully converted and saved as `updated_model.keras` without `time_major`.")
