import pandas as pd
import sys
from Class.MCDO_BERT import MCDO_BERT


# read data
dataset = pd.read_csv("data/train_data.csv")

config = {
    "model_name": "KoBERT-l2-d2_ep10",
    "bert_type": "KoBERT",
    "device": "cpu",
    "train_bottle_neck_stacks": [
        {
            "method": "Linear",
            "input_size": 768,
            "output_size": 3
        }
    ],
    "montecarlo_num": 100,
    "montecarlo_method": "mean",
    "inference_bottle_neck_stacks": [
        {
            "method": "Dropout",
            "dropout_rate": 0.3,
        },
        # {
        #     "method": "Dropout",
        #     "dropout_rate": 0.3,
        # },
    ]
}

# Create an instance of the class
model = MCDO_BERT(**config)

model.description()

# Train the model
print("Fine tunning")
model.fine_tunning(dataset, epochs=10, batch_size=16)
model.save()
