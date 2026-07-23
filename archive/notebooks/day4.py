# # upsample the EDA with labels
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# # Load data
# with open(PROJECT_ROOT / "Data" / "raw" / "WESAD" / "S2" / "S2.pkl", "rb") as f:
#     data = pickle.load(f, encoding="latin1")

# eda = data['signal']['wrist']['EDA'].flatten()
# labels = data['label']

# # Upsample EDA to match label length
# x_old = np.linspace(0, 1, len(eda))
# x_new = np.linspace(0, 1, len(labels))

# eda_up = np.interp(x_new, x_old, eda)

# # Plot full signal
# plt.figure(figsize=(20,6))
# plt.plot(eda_up, linewidth=0.5)

# # Label boundaries
# changes = np.where(np.diff(labels) != 0)[0]

# start = 0

# for change in changes:
#     middle = (start + change) // 2

#     plt.text(
#         middle,
#         np.max(eda_up) * 0.95,
#         str(int(labels[start])),   # shows 0,1,2,3...
#         ha='center'
#     )

#     plt.axvline(change, color='red', linestyle='--', alpha=0.5)

#     start = change + 1

# # Last section
# middle = (start + len(labels)) // 2

# plt.text(
#     middle,
#     np.max(eda_up) * 0.95,
#     str(int(labels[start])),
#     ha='center'
# )

# plt.title("Upsampled EDA with Labels")
# plt.xlabel("Samples")
# plt.ylabel("EDA")
# plt.grid(True)

# plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Load data
with open(PROJECT_ROOT / "Data" / "raw" / "WESAD" / "S2" / "S2.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

labels = data['label']

def plot_signal(signal, signal_name):
    
    signal = np.array(signal).flatten()

    # Upsample signal to label length
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, len(labels))

    signal_up = np.interp(x_new, x_old, signal)

    plt.figure(figsize=(20,6))
    plt.plot(signal_up, linewidth=0.5)

    changes = np.where(np.diff(labels) != 0)[0]

    start = 0

    for change in changes:
        middle = (start + change) // 2

        plt.text(
            middle,
            np.max(signal_up) * 0.95,
            str(int(labels[start])),
            ha='center'
        )

        plt.axvline(change,
                    color='red',
                    linestyle='--',
                    alpha=0.5)

        start = change + 1

    middle = (start + len(labels)) // 2

    plt.text(
        middle,
        np.max(signal_up) * 0.95,
        str(int(labels[start])),
        ha='center'
    )

    plt.title(signal_name)
    plt.xlabel("Samples")
    plt.ylabel(signal_name)
    plt.grid(True)
    plt.show()

