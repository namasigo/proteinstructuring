import os
import warnings
from transformers import BertTokenizer, BertForSequenceClassification, AdapterConfig, AdapterType
import torch

# Suppress warnings (like the classifier init warning)
warnings.filterwarnings("ignore", category=UserWarning)

# Dynamically resolve the adapter path
base_path = os.path.dirname(__file__)
adapter_path = os.path.join(base_path, "..", "adapter_model")

# Confirm the config exists
config_path = os.path.join(adapter_path, "adapter_config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Adapter config not found in {adapter_path}. Make sure to save adapter properly after training.")

# Load tokenizer and base model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Load adapter
model.load_adapter(adapter_path, AdapterType.text_task, load_as="stability_adapter")
model.set_active_adapters("stability_adapter")
model.eval()

# Input sequence for prediction
sequence = "MKTFFVLLLTLVVVTIVCLDLGYT"

# Tokenize input
inputs = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

# Decode label
label_map = {0: "Unstable", 1: "Stable"}  # adjust if your label mapping is different
predicted_label = label_map.get(predicted_class, "Unknown")

# Output result
print(f"Sequence: {sequence}")
print(f"Predicted Stability: {predicted_label}")