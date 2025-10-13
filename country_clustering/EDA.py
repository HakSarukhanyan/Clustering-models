"""

EDA part

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats
from functions import *

df = pd.read_csv("Country-data.csv")

df = df.drop('country',axis=1)

# for describing economic status of country is gdpp, 
# we need to calculate correlation between gdpp and other columns
drop_col = []

# for i in df.columns:
#     if(i != 'gdpp'):
#         correlations = float(calculate_corr(df,'gdpp',i,'spearman'))
#         print(f"gdpp|{i}: {correlations} ")
#         #plot_corr(df,'gdpp',i,'spearman')
#         if not(-0.7 < correlations < 0.7):
#             drop_col.append(i)

# df = df.drop(columns=drop_col)
#print(df.corr('spearman')) 


scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# PCA
# we will use K-means algorithm, so we don't need to have high dimensional data

pca = PCA(n_components = 3)
final_df = pd.DataFrame(pca.fit_transform(scaled_df), columns = ['PCA1','PCA2','PCA3'])

pca_plotting(final_df,3)

z_scores = np.abs(stats.zscore(final_df))
final_no_outliers_df = final_df[(z_scores < 3).all(axis=1)]


print("before: ", final_df.shape[0])
print("after: ", final_no_outliers_df.shape[0])

