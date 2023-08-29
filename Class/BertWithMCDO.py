import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np
from typing import Tuple
from torchviz import make_dot
from torchsummary import summary


class BertWithMCDO(nn.Module):
    """
    BERT model for sequence classification with Monte Carlo Dropout
    """

    def __init__(self, dropout_rate=0.1, num_labels=3):
        super(BertWithMCDO, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, 512)  # Adding the first FC layer
        self.fc2 = nn.Linear(512, num_labels)  # Adding the second FC layer

    def load(self, path):
        """
        Load the model from the path
        :param path: path to load the model
        :return: None

        """
        self.load_state_dict(torch.load(path))

    def save(self, path):
        """
        Save the model to the path
        :param path: path to save the model
        :return: None

        """
        torch.save(self.state_dict(), path)

    def classifier_forward(self, pooled_output) -> torch.Tensor:
        """
        Feed input to BERT and the classifier to compute logits.
        :param input_ids: input ids
        :param attention_mask: attention mask
        :return: logits

        """
        fc_outputs = []
        dropout_output = self.dropout(pooled_output)
        fc1_output = self.fc1(dropout_output)
        fc_outputs.append(fc1_output)
        fc1_output = nn.functional.relu(fc1_output)
        dropout_output = self.dropout(fc1_output)
        fc2_output = self.fc2(dropout_output)
        fc_outputs.append(fc2_output)
        last_fc_output = fc2_output
        return last_fc_output, fc_outputs

    def forward(self, input_ids, attention_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        dropout -> fc1 -> relu -> dropout -> fc2
        Feed input to BERT and the classifier to compute logits.
        :param input_ids: input ids
        :param attention_mask: attention mask
        :return: logits and hidden_state

        """
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        hidden_state = bert_output.last_hidden_state[:, 0, :].cpu(
        ).detach().numpy()
        pooled_output = bert_output.pooler_output
        fc_output, fc_outputs = self.classifier_forward(pooled_output)
        return fc_output, hidden_state

    def monte_carlo_forward(self, input_ids, attention_mask, num_samples=100, embedding_idx=-1, method='norn2') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """
        dropout -> fc1 -> relu -> dropout -> fc2
        Feed input to BERT and the classifier to compute logits.
        :param input_ids: input ids
        :param attention_mask: attention mask
        :param num_samples: number of samples
        :return: prediction mean, prediction std, fc2_output, hidden_state
        """

        output_list = []
        monte_carlo_list = []

        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_hidden_state = bert_output.last_hidden_state[:, 0, :].cpu(
        ).detach().numpy()
        pooled_output = bert_output.pooler_output
        for _ in range(num_samples):
            with torch.no_grad():
                last_fc_output, fc_outputs = self.classifier_forward(
                    pooled_output)

                if method == 'norm2':
                    # calculate norm2
                    norm2 = torch.norm(fc_outputs[embedding_idx], dim=1)
                    output_list.append(norm2)

                elif method == 'max':
                    value, k = torch.max(fc_outputs[embedding_idx], dim=1)
                    output_list.append(value)

                monte_carlo_list.append(last_fc_output)

        monte_carlo_preds = torch.cat(monte_carlo_list, dim=0)
        prediction_mean = torch.mean(monte_carlo_preds, dim=0)
        prediction_std = torch.std(monte_carlo_preds, dim=0)
        _, predicted_label = torch.max(prediction_mean, dim=0)

        monte_carlo_preds = torch.cat(output_list, dim=0)
        prediction_mean = torch.mean(monte_carlo_preds, dim=0)
        prediction_std = torch.std(monte_carlo_preds, dim=0)

        return prediction_mean, prediction_std, predicted_label, bert_hidden_state


class BertWithMCDOLight(nn.Module):
    """
    BERT model for sequence classification with Monte Carlo Dropout
    """

    def __init__(self, dropout_rate=0.1, num_labels=3):
        super(BertWithMCDOLight, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, num_labels)

    def load(self, path):
        """
        Load the model from the path
        :param path: path to load the model
        :return: None

        """
        self.load_state_dict(torch.load(path))

    def save(self, path):
        """
        Save the model to the path
        :param path: path to save the model
        :return: None

        """
        torch.save(self.state_dict(), path)

    def classifier_forward(self, pooled_output) -> torch.Tensor:
        """
        Feed input to BERT and the classifier to compute logits.
        :param input_ids: input ids
        :param attention_mask: attention mask
        :return: logits

        """
        fc_outputs = []
        dropout_output = self.dropout(pooled_output)
        fc1_output = self.fc1(dropout_output)
        fc_outputs.append(fc1_output)
        last_fc_output = fc1_output
        return last_fc_output, fc_outputs

    def forward(self, input_ids, attention_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Feed input to BERT and the classifier to compute logits.
        :param input_ids: input ids
        :param attention_mask: attention mask
        :return: logits and hidden_state

        """
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        hidden_state = bert_output.last_hidden_state[:, 0, :].cpu(
        ).detach().numpy()
        pooled_output = bert_output.pooler_output
        fc_output, fc_outputs = self.classifier_forward(pooled_output)
        return fc_output, hidden_state

    def monte_carlo_forward(self, input_ids, attention_mask, num_samples=100, embedding_idx=-1, method='norm2') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Feed input to BERT and the classifier to compute logits.
        :param input_ids: input ids
        :param attention_mask: attention mask
        :param num_samples: number of samples
        :return: prediction mean, prediction std, fc2_output, hidden_state
        """

        output_list = []
        monte_carlo_list = []

        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_hidden_state = bert_output.last_hidden_state[:, 0, :].cpu(
        ).detach().numpy()
        pooled_output = bert_output.pooler_output
        for _ in range(num_samples):
            with torch.no_grad():
                last_fc_output, fc_outputs = self.classifier_forward(
                    pooled_output)

                if method == 'norm2':
                    # calculate norm2
                    norm2 = torch.norm(fc_outputs[embedding_idx], dim=1)
                    output_list.append(norm2)

                elif method == 'max':
                    value, k = torch.max(fc_outputs[embedding_idx], dim=1)
                    output_list.append(value)

                monte_carlo_list.append(last_fc_output)

        monte_carlo_preds = torch.cat(monte_carlo_list, dim=0)
        prediction_mean = torch.mean(monte_carlo_preds, dim=0)
        prediction_std = torch.std(monte_carlo_preds, dim=0)
        _, predicted_label = torch.max(prediction_mean, dim=0)

        monte_carlo_preds = torch.cat(output_list, dim=0)
        prediction_mean = torch.mean(monte_carlo_preds, dim=0)
        prediction_std = torch.std(monte_carlo_preds, dim=0)

        return prediction_mean, prediction_std, predicted_label, bert_hidden_state


class BertWithMCDOBase(nn.Module):
    """
    BERT model for sequence classification with Monte Carlo Dropout
    """

    def __init__(self, dropout_rate=0.1, num_labels=3):
        super(BertWithMCDOBase, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(768, num_labels)

    def load(self, path):
        """
        Load the model from the path
        :param path: path to load the model
        :return: None

        """
        self.load_state_dict(torch.load(path))

    def save(self, path):
        """
        Save the model to the path
        :param path: path to save the model
        :return: None

        """
        torch.save(self.state_dict(), path)

    def forward(self, input_ids, attention_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Feed input to BERT and the classifier to compute logits.
        :param input_ids: input ids
        :param attention_mask: attention mask
        :return: logits and hidden_state

        """
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        dropout_output = self.dropout(pooled_output)
        out = self.fc(dropout_output)
        return out, bert_output.last_hidden_state[:, 0, :].cpu().numpy()

    def monte_carlo_forward(self, input_ids, attention_mask, num_samples=100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Feed input to BERT and the classifier to compute logits.
        :param input_ids: input ids
        :param attention_mask: attention mask
        :param num_samples: number of samples
        :return: prediction mean, prediction std, fc2_output, hidden_state
        """
        output_list = []

        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        for _ in range(num_samples):
            with torch.no_grad():
                dropout_output = self.dropout(pooled_output)
                out = self.fc(dropout_output)

                output_list.append(out)

        monte_carlo_preds = torch.cat(output_list, dim=0)
        prediction_mean = torch.mean(monte_carlo_preds, dim=0)
        prediction_std = torch.std(monte_carlo_preds, dim=0)

        return prediction_mean, prediction_std, bert_output.last_hidden_state[:, 0, :].cpu().numpy()


if __name__ == "__main__":
    model = BertWithMCDO()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer("test", padding=True,
                       truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
