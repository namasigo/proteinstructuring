# predict_adapter.py

from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
import torch
import os

# Paths
base_model_name = "bert-base-uncased"
adapter_path = "../adapter/adapter_model"  # Adjust if needed

# Check that adapter config exists
if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
    raise FileNotFoundError(f"Adapter config not found in {adapter_path}. Make sure to save adapter properly after training.")

# Load tokenizer and base model
tokenizer = BertTokenizer.from_pretrained(base_model_name)
base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=2)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Prediction function
def predict_stability(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "Stable" if prediction == 0 else "Unstable"

# Example usage
if __name__ == "__main__":
    test_sequence = "MKTFFVLLLTLVVVTIVCLDLGYT"  # Replace this with any sequence you want
    result = predict_stability(test_sequence)
    print(f"\n🧬 Sequence: {test_sequence}")
    print(f"🧪 Predicted Stability: {result}\n")