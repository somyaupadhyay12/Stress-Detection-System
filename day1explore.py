import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = "data/raw/WESAD/S2/S2.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f, encoding='latin1')
print(data.keys())

print(type(data))
print(data.keys())

print(type(data['signal']))
print(type(data['label']))
print(type(data['subject']))


import numpy as np

print("Label shape:", data['label'].shape)
print("Unique labels:", np.unique(data['label']))
