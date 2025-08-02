from temberture import TemBERTure

model = TemBERTure()
sequences = ["MVLSPADKTNVKAAW", "GAVLILLLV"]
predicted_class, confidence_score = model.predict(sequences)

for seq, cls, score in zip(sequences, predicted_class, confidence_score):
    print(f"Sequence: {seq}")
    print(f"Predicted class: {cls}")
    print(f"Confidence score: {score:.2f}")
    print("-" * 30)