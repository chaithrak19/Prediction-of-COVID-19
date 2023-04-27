


import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('abc.jpg', 0)
colormap = plt.get_cmap('inferno')
heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

cv2.imshow('image', image)
cv2.imshow('heatmap', heatmap)
cv2.waitKey()


import pandas as pd
df = pd.read_csv('covid.csv')

#df = pd.DataFrame({'DOB': {0: '26/1/2016', 1: '26/1/2016'}})
print (df)

df['date12'] = pd.to_datetime(df.date12)
print (df)

df['date12'] = df['date12'].dt.strftime('%m/%d/%Y')
print (df)
reco = df['recovery']
date12 = df['date12']
num_cases = df["confirm cases"]
import matplotlib.pyplot as plt

# Plot
plt.plot_date(date12, num_cases, linestyle='solid')

plt.plot_date(date12, reco, linestyle='solid')

#plt.plot(df["confirm cases"], marker='o')

# Labelling

plt.xlabel("date")
plt.ylabel("confirm cases")
plt.title("Pandas Time Series Plot")

# Display

plt.show()
