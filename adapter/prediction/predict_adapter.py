# predict_adapter.py

from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
import torch

# Load model and tokenizer
base_model_name = "bert-base-uncased"
model_path = "./adapter_model"  # Update if your fine-tuned model is elsewhere

tokenizer = BertTokenizer.from_pretrained(base_model_name)
base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

# Define prediction function
def predict_stability(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "Stable" if prediction == 0 else "Unstable"

# Example usage
if __name__ == "__main__":
    test_sequence = "MKTFFVLLLTLVVVTIVCLDLGYT"  # Replace with your own sequence
    result = predict_stability(test_sequence)
    print(f"Sequence: {test_sequence}")
    print(f"Predicted Stability: {result}")
if __name__ == "__main__":
    test_sequence = "MKTFFVLLLTLVVVTIVCLDLGYT"  # Example sequence
    result = predict_stability(test_sequence)
    print(f"Sequence: {test_sequence}")
    print(f"Predicted Stability: {result}")