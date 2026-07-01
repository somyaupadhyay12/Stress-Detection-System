# import pandas as pd
# import matplotlib.pyplot as plt

# file_path= "data/raw/WESAD/S2/S2_E4_Data/ACC.csv"
# df1= pd.read_csv(file_path)
# print(df1.head())
# print(df1.columns)
# print(df1.shape)
# df1.plot()
# plt.show()
# file_path= "data/raw/WESAD/S2/S2_E4_Data/EDA.csv"
# df2= pd.read_csv(file_path)
# print(df2.head())
# print(df2.columns)
# print(df2.shape)
# df2.plot()
# plt.show()
# file_path= "data/raw/WESAD/S2/S2_E4_Data/HR.csv"
# df3= pd.read_csv(file_path)
# print(df3.head())
# print(df3.columns)
# print(df3.shape)
# df3.plot()
# plt.show()
# file_path= "data/raw/WESAD/S2/S2_E4_Data/TEMP.csv"
# df4= pd.read_csv(file_path)
# print(df4.head())
# print(df4.columns)
# print(df4.shape)
# df4.plot()
# plt.show()

# ALL PLOTS ARE INDIVIDUALLY PLOTTED
# import pandas as pd
# import matplotlib.pyplot as plt

# df1 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/ACC.csv")
# df2 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/BVP.csv")
# df3 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/EDA.csv")
# df4 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/HR.csv")
# df5 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/IBI.csv")
# df6 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/TEMP.csv")



# print(df1.head())
# print(df1.shape)

# print(df2.head())
# print(df2.shape)

# print(df3.head())
# print(df3.shape)

# print(df4.head())
# print(df4.shape)

# df1.plot(title="ACC")
# df2.plot(title="BVP")
# df3.plot(title="EDA")
# df4.plot(title="HR")
# df5.plot(title="IBI")
# df6.plot(title="TEMP")

# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/EDA.csv")

# plt.plot(df)

# plt.title("EDA Signal")
# plt.xlabel("Sample Number")
# plt.ylabel("Skin Conductance (µS)")

# plt.show()



# # Put all 4 graphs in one figure (recommended)
# import pandas as pd
# import matplotlib.pyplot as plt

# df1 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/ACC.csv")
# df2 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/BVP.csv")
# df3 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/EDA.csv")
# df4 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/HR.csv")
# df5 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/IBI.csv")
# df6 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/TEMP.csv")

# fig, axs = plt.subplots(6, 1, figsize=(12, 15))

# df1.plot(ax=axs[0], legend=False)
# axs[0].set_title("ACC")

# df2.plot(ax=axs[1], legend=False)
# axs[1].set_title("BVP")

# df3.plot(ax=axs[2], legend=False)
# axs[2].set_title("EDA")

# df4.plot(ax=axs[3], legend=False)
# axs[3].set_title("HR")

# df5.plot(ax=axs[4], legend=False)
# axs[4].set_title("IBI")

# df6.plot(ax=axs[5], legend=False)
# axs[5].set_title("TEMP")

# plt.tight_layout()
# plt.show()

                ## PLOTTING E4 DATA WITH COMMON TIME AXIS###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/ACC.csv")
df2 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/BVP.csv")
df3 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/EDA.csv")
df4 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/HR.csv")
df5 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/IBI.csv")
df6 = pd.read_csv("data/raw/WESAD/S2/S2_E4_Data/TEMP.csv")
time_acc = np.arange(len(df1)) / 32
time_bvp = np.arange(len(df2)) / 64
time_eda = np.arange(len(df3)) / 4
time_hr  = np.arange(len(df4)) / 1
time_temp = np.arange(len(df6)) / 4


fig, axs = plt.subplots(5,1,sharex=True)

axs[0].plot(time_acc, df1)
plt.xlabel("Time (seconds)")
axs[1].plot(time_bvp, df2)
plt.xlabel("Time (seconds)")
axs[2].plot(time_eda, df3)
plt.xlabel("Time (seconds)")
axs[3].plot(time_hr, df4)
plt.xlabel("Time (seconds)")
axs[4].plot(time_temp, df6)
plt.xlabel("Time (seconds)")
axs[0].set_title("ACC")
axs[1].set_title("BVP") 
axs[2].set_title("EDA")
axs[3].set_title("HR")      
axs[4].set_title("TEMP")

plt.xlabel("Time (seconds)")
plt.show()




        ##PLOTTING RESPIBAN DATA WITH SAME DATA AXIS AS OTHER SIGNALS###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load file
df = pd.read_csv(
    "data/raw/WESAD/S2/S2_respiban.txt",
    sep=r"\s+",
    comment="#",
    header=None
)

print(df.shape)
print(df.head())

# Respiban sampling frequency
fs = 700

# Common time axis
time = np.arange(len(df)) / fs

# Plot all channels on same time axis
fig, axs = plt.subplots(7, 1, figsize=(15, 12), sharex=True)

axs[0].plot(time, df.iloc[:, 2])
axs[0].set_title("ECG")

axs[1].plot(time, df.iloc[:, 3])
axs[1].set_title("EDA")

axs[2].plot(time, df.iloc[:, 4])
axs[2].set_title("EMG")

axs[3].plot(time, df.iloc[:, 5])
axs[3].set_title("TEMP")

# Accelerometer X Y Z
axs[4].plot(time, df.iloc[:, 6])
axs[4].set_title("ACC X")

axs[5].plot(time, df.iloc[:, 7])
axs[5].set_title("ACC Y")

axs[6].plot(time, df.iloc[:, 8])
axs[6].set_title("ACC Z")

plt.xlabel("Time (seconds)")
plt.tight_layout()
plt.show()


##plotting acc all together
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Respiban data
df = pd.read_csv(
    "data/raw/WESAD/S2/S2_respiban.txt",
    sep=r"\s+",
    comment="#",
    header=None
)

# Sampling frequency
fs = 700

# Common time axis
time = np.arange(len(df)) / fs

# Extract signals
ecg = df.iloc[:, 2]
eda = df.iloc[:, 3]
emg = df.iloc[:, 4]
temp = df.iloc[:, 5]

# Accelerometer axes
acc_x = df.iloc[:, 6]
acc_y = df.iloc[:, 7]
acc_z = df.iloc[:, 8]

# Combine X, Y, Z into one ACC signal
acc = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

# Plot
fig, axs = plt.subplots(5, 1, figsize=(15, 12), sharex=True)

axs[0].plot(time, ecg)
axs[0].set_title("ECG")

axs[1].plot(time, eda)
axs[1].set_title("EDA")

axs[2].plot(time, emg)
axs[2].set_title("EMG")

axs[3].plot(time, temp)
axs[3].set_title("TEMP")

axs[4].plot(time, acc)
axs[4].set_title("ACC (Combined XYZ)")
axs[4].set_xlabel("Time (seconds)")

plt.tight_layout()
plt.show()

