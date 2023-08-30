import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModel


class BottleNeck(nn.Module):
    def __init__(self,
                 stacks,
                 device=(
                     torch.device("cuda") if torch.cuda.is_available(
                     ) else torch.device("cpu")
                 )
                 ):
        super(BottleNeck, self).__init__()
        self.layers = []
        self.device = device
        for idx, stack in enumerate(stacks):
            method = stack["method"]

            if method == "Linear":
                input_size = stack["input_size"]
                output_size = stack["output_size"]
                linear_layer = nn.Linear(input_size, output_size)
                linear_layer.to(self.device)
                self.layers.append(linear_layer)
                if idx < len(stacks) - 1:  # Apply ReLU for all but the last Linear layer
                    self.layers.append(nn.ReLU())
            elif method == "Dropout":
                dropout_rate = stack["dropout_rate"]
                self.layers.append(nn.Dropout(dropout_rate))
            else:
                raise Exception("Invalid method")

    def description(self):
        for layer in self.layers:
            print(layer)

    def save(self, path):
        state_dict = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                state_dict[f'linear_{idx}'] = layer.state_dict()

        torch.save(state_dict, path)

    def forward(self, bert_output):
        output = bert_output
        for layer in self.layers:
            output = layer(output)
        return output
