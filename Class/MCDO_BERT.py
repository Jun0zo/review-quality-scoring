import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModel

from .BottleNeck import BottleNeck
from .BaseBERT import BaseBERT
from .BaseTokenizer import Tokenizer
from .MontecarloMethod import MontecarloMethod


class MCDO_BERT():
    def __init__(self,
                 model_name,
                 bert_type=None,
                 montecarlo_num=None,
                 montecarlo_method='mean',
                 train_bottle_neck_stacks=None,
                 inference_bottle_neck_stacks=None,
                 device=None):

        self.model_name = model_name
        self.bert_type = bert_type
        # define device

        print("model name :", model_name)
        print("model type :", bert_type)
        if device == None:
            self.device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = Tokenizer(self.bert_type).tokenizer
        self.base_bert = BaseBERT(bert_type=self.bert_type, device=self.device)
        self.base_bert.load(self.model_name)

        
        self.bert_type = bert_type
        self.montecarlo_num = montecarlo_num
        self.train_bottle_neck_stacks = train_bottle_neck_stacks
        self.inference_bottle_neck_stacks = inference_bottle_neck_stacks
        
        self.montecarlo_method = MontecarloMethod(montecarlo_method)
        self.train_bottle_neck = BottleNeck(train_bottle_neck_stacks, device)
        self.inference_bottle_neck = BottleNeck(inference_bottle_neck_stacks, device)
        
        self.train_bottle_neck.to(self.device)
        self.inference_bottle_neck.to(self.device)

    def inference(self, text):
        # predict (base_bert + prediction_bottle_neck)
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device)
                      for key, value in inputs.items()}
            base_bert_outputs = self.base_bert(**inputs)
            bert_pooler_output = base_bert_outputs.pooler_output
            inference_bottle_neck_outputs = self.inference_bottle_neck.forward(
                bert_pooler_output)

        return torch.sum(inference_bottle_neck_outputs)

    def fine_tunning(self, dataset, epochs=3, batch_size=16):
        # define optimizer and loss function
        optimizer = torch.optim.AdamW(self.base_bert.parameters(), lr=5e-5)
        loss_fn = torch.nn.CrossEntropyLoss()

        # split dataset into train and test
        labels = dataset["label"].map({"높음": 2, "보통": 1, "낮음": 0})
        texts = dataset["text"]
        encoded_texts = self.tokenizer(
            texts.tolist(), return_tensors="pt", padding=True, truncation=True)
        input_dataset = TensorDataset(
            encoded_texts["input_ids"], encoded_texts["attention_mask"], torch.tensor(labels.tolist()))

        train_ratio = 0.8
        train_size = int(len(input_dataset) * train_ratio)
        test_size = len(input_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            input_dataset, [train_size, test_size])

        # define data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True)

        # train (base_bert + train_bottle_neck)
        for epoch in tqdm(range(epochs)):
            self.base_bert.train()
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(
                    self.device), attention_mask.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                # print("input_ids", input_ids.shape)
                # print("attention_mask", attention_mask.shape)
                # print('type : ', self.base_bert.bert_type)

                
                base_bert_outputs = self.base_bert(
                    input_ids, attention_mask=attention_mask)
                bert_pooler_output = base_bert_outputs.pooler_output
                # print bert_pooler_output is cpu or gpu
                # pass through train_bottle_neck

                # print("device :", self.train_bottle_neck.device)
                
                # print(self.train_bottle_neck.expected_moved_cuda_tensor.device)
                train_bottle_neck_outputs = self.train_bottle_neck.forward(
                    bert_pooler_output)  # Pass through train_bottle_neck

                loss = loss_fn(train_bottle_neck_outputs, labels)
                loss.backward()
                optimizer.step()

    def montecarlo_inference(self, text):
        montecarlo_outputs = []

        self.base_bert.eval()
        with torch.no_grad():
            # BERT outputs
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device)
                    for key, value in inputs.items()}
            try:
                outputs = self.base_bert(inputs["input_ids"], inputs["attention_mask"])
                base_bert_outputs = outputs.last_hidden_state[:, 0, :]
            except Exception as e:
                print("error ", e, inputs["input_ids"].shape, inputs["attention_mask"].shape, text)

            # Monte Carlo method
            for _ in range(self.montecarlo_num):
                # Pass through prediction_bottle_neck
                prediction_bottle_neck_outputs = self.inference_bottle_neck.forward(
                    base_bert_outputs)
                
                # Pass through montecarlo_method
                montecarlo_output = self.montecarlo_method(
                    prediction_bottle_neck_outputs)
                montecarlo_outputs.append(montecarlo_output)

        # Calculate mean and std
        montecarlo_outputs = torch.tensor(montecarlo_outputs)
        # print(montecarlo_outputs)
        # montecarlo_outputs = torch.cat(montecarlo_outputs, dim=0)
        montecarlo_mean = torch.mean(montecarlo_outputs, dim=0)
        montecarlo_std = torch.std(montecarlo_outputs, dim=0)

        return montecarlo_mean, montecarlo_std

    def description(self):
        print("MCDO_BERT")
        print("bert_type:", self.bert_type)
        print("montecarlo_num:", self.montecarlo_num)
        print("train_bottle_neck_stacks:")
        for stack in self.train_bottle_neck_stacks:
            print(stack)
        print("inference_bottle_neck_stacks:")
        for stack in self.inference_bottle_neck_stacks:
            print(stack)
        print("device:", self.device)
        print("tokenizer:", self.tokenizer)
        print("base_bert:", self.base_bert)
        print("montecarlo_method:", self.montecarlo_method)
        print("train_bottle_neck:", self.train_bottle_neck)
        print("inference_bottle_neck:", self.inference_bottle_neck)

    def save(self):
        os.makedirs(f"model/{self.model_name}", exist_ok=True)
        torch.save(self.base_bert.state_dict(),
                   f"model/{self.model_name}/base_bert.pt")
