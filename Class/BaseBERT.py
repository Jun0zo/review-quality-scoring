import torch
from torch import nn
from transformers import AutoModel, BertModel


class BaseBERT(nn.Module):
    def __init__(self, bert_type, device='cpu'):
        super(BaseBERT, self).__init__()
        self.model = None
        self.bert_type = bert_type
        if bert_type == "BERT":
            self.model = AutoModel.from_pretrained(
                "bert-base-multilingual-cased").to(device)
        elif bert_type == "KoBERT":
            self.model = BertModel.from_pretrained("monologg/kobert").to(device)
        else:
            raise Exception("Invalid model type")
        
    def __call__(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def save(self, model_name):
        if self.model is not None:
            torch.save(self.model.state_dict(), f"model/{model_name}/base_bert.pt")
            print(f"Model saved to model/{model_name}/base_bert.pt")
        else:
            print("No model to save")

    def load(self, model_name):
        try:
            if self.model is not None:
                print("[*] model loading")
                self.model.load_state_dict(torch.load(f"model/{model_name}/base_bert.pt"))
                print(f"Model loaded from model/{model_name}/base_bert.pt")
            else:
                print("No model to load")
        except Exception as e:
            pass # print("Error ", e)
