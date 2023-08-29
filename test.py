import pandas as pd
import sys
from Class.BERT import MCDO_BERT


# read data
dataset = pd.read_csv("data/train_data.csv")

config = {
    "model_type": "BERT",
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

# Create an instance of the class
model = MCDO_BERT(**config)

model.description()

# Predict the label of the sample text
predicted_label = model.inference(sample_text)

# Print the predicted label
print(predicted_label)

print("============================")
print('dataset 1', len(dataset))
dataset.reset_index(drop=True, inplace=True)
print('dataset 2', len(dataset))

# Train the model
model.fine_tunning(dataset, epochs=3, batch_size=16)
