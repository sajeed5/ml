import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pdb

# Load dataset from CSV file
df1 = pd.read_csv("dataset.csv")
print(df1)

# Extract 'Distance_Feature' and 'Speeding_Feature'
f1 = df1['Distance_Feature'].values
f2 = df1['Speeding_Feature'].values # Correct this line

# Stack 'f1' and 'f2' along axis=1
X = np.column_stack((f1, f2))

# Define the model
kmeans_model = KMeans(n_clusters=3)

# Fit the model to the data
kmeans_model.fit(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=kmeans_model.labels_)
plt.title('K-Means Clustering')
plt.xlabel('Distance Feature')
plt.ylabel('Speeding Feature')
plt.show()