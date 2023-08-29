import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter

if not os.path.isfile("ratings_total.csv"):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename="ratings_total.txt")
    df = pd.read_table('ratings_total.txt', names=['Ratings', 'Text'])
    df.to_csv('ratings_total.csv', index=False)
    os.remove("ratings_total.txt")
