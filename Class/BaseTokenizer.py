from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertForSequenceClassification

class Tokenizer():
    def __init__(self, model_type):
        self.tokenizer = None

        if model_type == "BERT":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-multilingual-cased")
        elif model_type == "KoBERT":
            print()
            self.tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
        else:
            raise Exception("Invalid model type")
