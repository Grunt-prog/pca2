
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("balance-scale.data")
print(df.shape)
# Drop rows with missing values
df.dropna(inplace=True)

# Separate the last column into df1
df1 = df[df.columns[0]].copy()
df.drop(df.columns[0], axis=1, inplace=True)

category_mapping = {'L': 0, 'B': 1, 'R': 2}

# Map the categories to numerical values
df1 = df1.map(category_mapping)
# Drop the first column from df


# Standardize the data
scaler = StandardScaler()
df_standard = scaler.fit_transform(df)

# Compute the covariance matrix
df_cov = np.cov(df_standard.T)

# Calculate eigenvalues and eigenvectors
eig_values, eig_vectors = np.linalg.eig(df_cov)

# Sort eigenvalues in descending order
sorted_idx = np.argsort(eig_values)[::-1]
eig_values = eig_values[sorted_idx]
eig_vectors = eig_vectors[:, sorted_idx]

# Set the desired variance explained
variance_explained = 0.95
total_variance = np.sum(eig_values)

# Determine the number of dimensions (k) to retain
k = 0
variance_sum = 0
while (variance_sum / total_variance < variance_explained):
    variance_sum += eig_values[k]
    k += 1

print("No of dimensions: " + str(k))

# Select the top k eigenvectors
top_k_eigenvectors = eig_vectors[:, :k]

# Perform dimensionality reduction using PCA
X_pca = np.dot(df_standard, top_k_eigenvectors)

# Create a scatter plot of the transformed data
fig, ax = plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df1.values, cmap='coolwarm')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Scatter Plot of Transformed Data' + " no of dimensions: " + str(k))

plt.show()