import pandas as pd
import sys
from Class.BERT import MCDO_BERT


# read data
dataset = pd.read_csv("data/train_data.csv")

config = {
    "model_type": "BERT",
    "device": "cpu",
    "montecarlo_num": 10,
    "train_bottle_neck_stacks": [
        {
            "method": "Linear",
            "input_size": 768,
            "output_size": 3
        }
    ],
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

# Define a sample text for classification
sample_text = "이 영화 정말 재미있었어요!"
sample_text = "엥"

# Create an instance of the class
model = MCDO_BERT(**config)

# model.description()

# Predict the label of the sample text
for _ in range(10):
    res = model.inference(sample_text)
    print("res : ", res)

# Train the model
print("Fine tunning")
model.fine_tunning(dataset, epochs=3, batch_size=16)
