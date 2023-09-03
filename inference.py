import pandas as pd
import torch
from tqdm import tqdm
import sys
import os
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


def save_boxplot(results, save_path):
   # Create a boxplot for 'l' values
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    ax = sns.boxplot(data=results)
    ax.set(xticklabels=['High', 'Medium', 'Low'])
    ax.set_xlabel('Authenticity Score')
    ax.set_ylabel('Standard Deviation (l values)')
    ax.set_title(
        'Boxplot of Standard Deviation for Different Authenticity Scores')
    plt.savefig(save_path)


def get_inference_results(model):
    results = []

    results_by_row = []

    for score in ('높음', '보통', '낮음'):
        print(f'진정성 {score}')

        filtered_dataset = dataset[dataset["label"] == score][:]
        montecarlo_results = []
        for datalist in tqdm(filtered_dataset.iloc):
            if len(datalist["text"]) > 500:
                continue
            montecarlo_mean, montecarlo_std = model.montecarlo_inference(
                datalist["text"])
            montecarlo_results.append(montecarlo_std)

            results_by_row.append([score, len(
                datalist["text"]), datalist["text"], montecarlo_std.item()])

        montecarlo_results = torch.tensor(montecarlo_results)
        results.append(montecarlo_results)

        print("mean std : ", torch.mean(montecarlo_results, dim=0))

        print("=========================")
    return results, results_by_row


def sort_by_score(results_by_row):
    new_df = pd.DataFrame(columns=["label", "text_length", "text", "score"])
    for idx, tmp in enumerate(results_by_row):
        new_df.loc[idx] = tmp
    new_df = new_df.sort_values(by="score", ascending=False)
    return new_df


def save_score_distib(sorted_df, save_path):
    div_n = int(len(sorted_df) / 3)
    for point, correct in [(div_n, '높음'), (div_n*2, '보통'), (div_n*3, '낮음')]:
        print("---------------------------", correct)
        div_df = sorted_df[:div_n]
        correct_counts = div_df['label'].value_counts()[correct]
        print(correct_counts, len(div_df), correct_counts / len(div_df))
        sorted_df = sorted_df[div_n:]


def save_experiments(config):
    base_path = f"{config['model_name']}/{config['montecarlo_num']}-{config['montecarlo_method']}-{str(config['inference_bottle_neck_stacks'][0]['dropout_rate'])}-{str(config['inference_bottle_neck_stacks'][1]['dropout_rate'])}"

    os.makedirs(base_path, exist_ok=True)

    model = MCDO_BERT(**config)
    results, results_by_row = get_inference_results(model)
    save_boxplot(
        results, f"{base_path}/boxplot.png")
    sorted_df = sort_by_score(results_by_row)
    sorted_df.to_csv(f"{base_path}/sorted.csv", index=False)

    save_score_distib(sorted_df, f"{base_path}/sorted.csv")

# for montecarlo_num in (10, 100, 1000):
#         for montecarlo_method in ("sum", "mean", "std", "norm2"):
#             for dropout_rate_1 in (0, 100, 10):
#                 for dropout_rate_2 in (0, 100, 10):
#                     config["montecarlo_num"] = montecarlo_num
#                     config["montecarlo_method"] = montecarlo_method
#                     config["inference_bottle_neck_stacks"][0]["dropout_rate"] = dropout_rate_1
#                     config["inference_bottle_neck_stacks"][1]["dropout_rate"] = dropout_rate_2


save_experiments(config)
