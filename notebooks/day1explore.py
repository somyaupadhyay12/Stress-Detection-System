import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
file_path = PROJECT_ROOT / "Data" / "raw" / "WESAD" / "S2" / "S2.pkl"
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


