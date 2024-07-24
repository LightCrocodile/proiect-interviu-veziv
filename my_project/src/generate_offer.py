import json
import numpy as np
from keras.models import load_model
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the custom tokenizer
from keras_nlp.my_modules.models.gemma.gemma_tokenizer import GemmaTokenizer

# Path to the model and tokenizer
model_path = r"D:\ardu\my_project\data\gemma2"
model_file = os.path.join(model_path, "model.weights.h5")
metadata_file = os.path.join(model_path, "metadata.json")

# Load the model
model = load_model(model_file)

# Load the custom tokenizer
tokenizer = GemmaTokenizer.from_pretrained(r"D:\ardu\my_project\data\gemma2")

def generate_offer(client_requirements, tokenizer, model):
    # Encode client requirements
    inputs = tokenizer.encode(client_requirements, return_tensors="np")
    input_ids = np.array(inputs["input_ids"])

    # Generate text using the Keras model
    outputs = model.predict(input_ids)

    # Decode the result
    predicted_ids = np.argmax(outputs, axis=-1)
    offer_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return offer_text

# Example usage
if __name__ == "__main__":
    client_requirements = "Clientul dorește o aplicație de tip Glovo..."
    print(generate_offer(client_requirements, tokenizer, model))
