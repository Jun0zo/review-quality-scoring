import torch
from transformers import AutoModel


class BaseBERT():
    def __init__(self, model_type):
        self.model = None

        if model_type == "BERT":
            self.model = AutoModel.from_pretrained(
                "bert-base-multilingual-cased")
        elif model_type == "KoBERT":
            self.model = AutoModel.from_pretrained("snunlp/KR-Medium")
        else:
            raise Exception("Invalid model type")

    def save(self, path):
        torch.save(self.model.state_dict(), path)
