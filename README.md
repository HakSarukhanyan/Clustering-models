# Clustering-models

# KMeans Clustering on Country Data

This project demonstrates **unsupervised learning** using the **KMeans Clustering** algorithm on a dataset containing various economic and social indicators for different countries.  
The main goal is to group countries into clusters based on their economic similarities.

---

##  Project Structure

```
.
├── country_clustering/
│   ├── Country-data.csv       # Dataset
│   ├── EDA.py                 # Exploratory Data Analysis
│   ├── train.py               # Model training & clustering
│   ├── functions.py           # Helper functions for plotting, correlation, etc.
├── README.md
```

---

##  Features

  -  EDA (Exploratory Data Analysis):
  - Outlier detection using PCA visualization

 -**Dimensionality Reduction**:
  - PCA with 3 components to simplify clustering and visualize in 3D.

-**KMeans Clustering**:
  - Choosing cluster count
  - Evaluating with inertia and silhouette score
  - 2D and 3D visualization of clusters

-**Preprocessing**:
  - Handling outliers
  - Feature scaling (RobustScaler / MinMaxScaler)

---

##  Installation

Clone the repository and install dependencies:

```
git clone git@github.com:HakSarukhanyan/Clustering-models.git
cd Clustering-models
pip install -r requirements.txt
```

---

##  Requirements

```
pandas==2.3.3
numpy=2.3.3
matplotlib=3.10.7
scikit-learn=1.7.2
```

Install:
```
pip install pandas numpy matplotlib scikit-learn
```

---

##  Usage

###  1. Exploratory Data Analysis
Run:
```
python country_clustering/EDA.py
```
This will:
- Load and preprocess the data
- Apply scaling & PCA
- Plot 3D PCA for outlier analysis

---

###  2. Train KMeans Model
Run:
```
python country_clustering/train.py
```
This will:
- Fit KMeans with the chosen number of clusters
- Print inertia & silhouette score
- Plot cluster distribution in 3D

---

##  Evaluation Metrics

| Metric             | Description                                              |
|---------------------|-----------------------------------------------------------|
| **Inertia**         | Measures within-cluster sum of squares (lower = better).   |
| **Silhouette Score**| How well samples are clustered (closer to 1 = better).     |

---

##  Example Output

- **Inertia**: `≈ 7.0.1`  
- **Silhouette Score**: `≈ 0.35`  
- 3D visualization with `matplotlib` shows cluster separation.

---

##  Future Improvements

- Use **DBSCAN** or **Hierarchical Clustering** for comparison  

- Try **GPU acceleration** with RAPIDS cuML for large datasets  
- Add dashboard for interactive visualization

---


##  Author

**Hakob Sarukhanyan**  
📧 hakob0511@gmail.com  
🌐 [GitHub Profile](https://github.com/HakSarukhanyan)


