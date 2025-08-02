import torch
from transformers import BertTokenizer, BertModel
from adapter_transformers import BertAdapterModel

class TemBERTure:
    def __init__(self, adapter_path, device='cpu', batch_size=1, task='classification'):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.task = task

        # Load tokenizer and base model
        self.tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
        self.model = BertAdapterModel.from_pretrained('Rostlab/prot_bert')
        self.model.to(self.device)

        # Load adapter
        self.model.load_adapter(adapter_path, load_as=task)
        self.model.set_active_adapters(task)

        # Set model to evaluation mode
        self.model.eval()

    def preprocess(self, sequences):
        # Add spaces between amino acids
        spaced_seqs = [' '.join(list(seq)) for seq in sequences]
        tokens = self.tokenizer(spaced_seqs, return_tensors='pt', padding=True, truncation=True)
        return {k: v.to(self.device) for k, v in tokens.items()}

    def predict(self, sequences):
        inputs = self.preprocess(sequences)
        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.task == 'classification':
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            return preds.cpu().numpy(), probs[:, 1].cpu().numpy()  # Class and thermophilicity score

        elif self.task == 'regression':
            return outputs.logits.squeeze().cpu().numpy()  # Predicted Tm

        else:
            raise ValueError("Task must be 'classification' or 'regression'")