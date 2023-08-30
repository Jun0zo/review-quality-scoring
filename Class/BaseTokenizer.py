from transformers import AutoTokenizer, AutoModel


class Tokenizer():
    def __init__(self, model_type):
        self.tokenizer = None

        if model_type == "BERT":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-multilingual-cased")
        elif model_type == "KoBERT":
            self.tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-Medium")
        else:
            raise Exception("Invalid model type")
