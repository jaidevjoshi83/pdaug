

import matplotlib.pyplot as plt
import seaborn as sns, os
import pandas as pd


df  = pd.read_csv('out.tsv', sep="\t")


plt.figure(figsize=(10,12))

x_axis_labels = df.columns[1:]
y_axis_labels = df[df.columns[0]].tolist()

sns.set(font_scale=2)
sns.heatmap(df[df.columns[1:]], center=0,  yticklabels=y_axis_labels, xticklabels=x_axis_labels)

plt.xticks(rotation=45)
plt.yticks(rotation=45)


plt.show()



