import torch
import torch.nn as nn
import pandas as pd
from transformers import  AutoTokenizer, AutoModel

class Tokenizer():
    def __init__(self, model_type):
        self.tokenizer = None

        if model_type == "BERT":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        elif model_type == "KoBERT":
            self.tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-Medium") 
        else:
            raise Exception("Invalid model type")

class BaseBERT():
    def __init__(self, model_type):
        self.model = None

        if model_type == "BERT":
            self.model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        elif model_type == "KoBERT":
            self.model = AutoModel.from_pretrained("snunlp/KR-Medium") 
        else:
            raise Exception("Invalid model type")
        
class MCDO_BERT():
    def __init__(self, model_type="BERT"):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.tokenizer = Tokenizer(model_type).tokenizer
        self.base_bert = BaseBERT(model_type).model
        self.base_bert.to(self.device)

    def test(self):
        text = "안녕하세요, 이것은 KoBERT의 예시입니다."
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.base_bert(**inputs)

        print(outputs)
        pooler_output = outputs.pooler_output
        print(pooler_output)

m = MCDO_BERT()
m.test()