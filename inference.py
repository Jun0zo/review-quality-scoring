import pandas as pd
import torch
from tqdm import tqdm
import sys
from Class.MCDO_BERT import MCDO_BERT
import seaborn as sns
import matplotlib.pyplot as plt

# read data
dataset = pd.read_csv("data/train_data.csv")

# Create an instance of the class
config = {
    "model_name": "KoBERT-l2-d2_1",
    "bert_type": "KoBERT",
    "device": "cuda:0",
    "train_bottle_neck_stacks": [
        {
            "method": "Linear",
            "input_size": 768,
            "output_size": 3
        }
    ],
    "montecarlo_num": 100,
    "montecarlo_method": "std",
    "inference_bottle_neck_stacks": [
        {
            "method": "Dropout",
            "dropout_rate": 0.1,
        },
        {
            "method": "Dropout",
            "dropout_rate": 0.1,
        },
    ]
}
model = MCDO_BERT(**config)

l_values = []

for score in ('높음', '보통', '낮음'):
    print(f'진정성 {score}')
    
    filtered_dataset = dataset[dataset["label"] == score][:]
    l = []
    for datalist in tqdm(filtered_dataset.iloc):
        if len(datalist["text"]) > 500:
            continue
        montecarlo_mean, montecarlo_std = model.montecarlo_inference(datalist["text"])
        l.append(montecarlo_std)

    l = torch.tensor(l)
    l_values.append(l)
    # print(l, l.shape)
    print("mean std : ", torch.mean(l, dim=0))

    print("=========================")

# Create a boxplot for 'l' values
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
ax = sns.boxplot(data=l_values)
ax.set(xticklabels=['High', 'Medium', 'Low'])
ax.set_xlabel('Authenticity Score')
ax.set_ylabel('Standard Deviation (l values)')
ax.set_title('Boxplot of Standard Deviation for Different Authenticity Scores')
plt.savefig("sav")