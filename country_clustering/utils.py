"""

Useful functions 

"""

import pandas as pd
import matplotlib.pyplot as plt

def calculate_corr(df,col1,col2,method):
    """
    Calculating correlation between 2 columns of DataFrame
    
    Args:
        df(pd.DataFrame): The name of DataFrame
        col1 (str): The name of first column
        col2 (str): The name of second column
        method (str): The name of method: ('pearson', 'spearman','kendall')
        
    Returns:
        float: Correlation value between col1 and col2
    """
    return df[[col1,col2]].corr(method=method).iloc[0,1]

def plot_corr(df,col1,col2,method):
    """
    Plotting correlation between 2 columns of DataFrame

    Args:
        df(pd.DataFrame): The name of DataFrame
        col1 (str): The name of first column
        col2 (str): The name of second column
        method (str): The name of method: ('pearson', 'spearman','kendall')
    
    Returns: None
        Plot Correlation between col1 and col2
    """
    
    plt.scatter(df[col1],df[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.grid(True)
    plt.show()
    correlation_col1_col2 = calculate_corr(df,col1,col2,method)
    return None
    
def show_stat(df,col):
    """
    Print sorted, max, min and mean of column, to find outliers easier

    Args:
        df (pd.DataFrame): The name of DataFrame
        col (str): The name of column
        
    Returns:
        List: [sorted_values(list),max_value,min_value,mean_value]
    """
    sorted_column = df[col].sort_values().tolist()
    max_column = df[col].max()
    min_column = df[col].min()
    mean_column = df[col].mean()
    print(f"{col} -> max: {max_column}, min: {min_column}, mean: {mean_column}")
    return [sorted_column,max_column,min_column,mean_column]
    
def pca_plotting(df: pd.DataFrame,axis_count: int, title: str):
    """
    Scatter plot for 2D or 3D data
    

    Args:
        df (pd.DataFrame): The name of DataFrame
        axis_count (int): 2 for 2D scatter, 3 for 3D scatter
        title(str): The title of plot
    Returns: 
        None: Plot

    """
    
    if axis_count == 2:
        plt.scatter(df[df.columns[0]], df[df.columns[1]], color = 'blue', s = 50, alpha = 0.7)
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        plt.grid(True)
        plt.title(title)
        plt.show()
        
    elif axis_count == 3:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df[df.columns[0]], df[df.columns[1]],df[df.columns[2]], color = 'blue',
                    s = 50, alpha = 0.7)
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        plt.clabel(df.columns[2])
        plt.grid(True)
        plt.show()
    else:
        raise ValueError("axis_count must be 2 or 3")
    
def plot_clusters_3d(df, labels, model):
    """
    Plot clusters and their centers in 3D after PCA.

    Args:
        df (pd.DataFrame): DataFrame with 3 columns (PCA1, PCA2, PCA3)
        labels (array): Cluster labels from the model
        model (object): Fitted clustering model with `cluster_centers_` attribute
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df[df.columns[0]], df[df.columns[1]], df[df.columns[2]],
        c=labels,cmap='viridis',s=50,alpha=0.7)


    ax.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        model.cluster_centers_[:, 2],
        c='red',
        marker='X',
        s=200,
        label='Cluster centers'
    )

    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_zlabel(df.columns[2])
    ax.legend()
    plt.tight_layout()
    plt.show()