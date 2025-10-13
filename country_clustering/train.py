"""

Model training

"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from EDA import *

# optional, run once to find good 'k'

# intertias = []
# K = range(2, 10)
# for k in K:
#     model = KMeans(n_clusters=k, init = "k-means++", n_init=50, max_iter=300, random_state=32)
#     model.fit(final_no_outliers_df)
#     intertias.append(model.inertia_)

# plt.plot(K, intertias, marker = "o", color = "red")
# plt.ylabel("Intertia")
# plt.xlabel("k")
# plt.title("Elbow Method")
# plt.grid(True)
# plt.show()

k=4

model = KMeans(n_clusters=k, init = 'k-means++', n_init=50,max_iter=3000,random_state=32)
model.fit(final_no_outliers_df)

labels = model.labels_

print(f"{model.inertia_ = }")

score = silhouette_score(final_no_outliers_df,labels)
print(score)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter the points (PCA1, PCA2, PCA3)
ax.scatter(
    final_no_outliers_df['PCA1'],
    final_no_outliers_df['PCA2'],
    final_no_outliers_df['PCA3'],
    c=labels,
    cmap='viridis',
    s=50,
    alpha=0.7
)

# Plot cluster centers
ax.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    model.cluster_centers_[:, 2],
    c='red',
    marker='X',
    s=200,
    label='Cluster centers'
)

ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
ax.legend()
plt.tight_layout()
plt.show()

# optional 
# can check relationship between silhouette_score and cluster counts

# for k in range(2, 10):
#     kmeans = KMeans(n_clusters=k, init="k-means++", n_init=50, max_iter=300, random_state=32)
#     kmeans.fit(final_no_outliers_df)
#     score = silhouette_score(final_no_outliers_df, kmeans.labels_)
#     print(f"Silhouette score for k={k}: {score}")
    