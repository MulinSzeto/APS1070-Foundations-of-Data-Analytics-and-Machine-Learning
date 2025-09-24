# APS1070 Projects

This repository contains four machine learning projects implemented in Python/Jupyter notebooks. Each project applies specific algorithms to a real dataset and explores its performance through structured tasks.

---

## Project 1 — KNN & Decision Trees on Housing Data
**Dataset:** [Boston Housing dataset](https://openml.org/search?type=data&status=any&id=43465)  
**Goal:** Classify whether the value of owner-occupied homes is above or below the median.  

**Techniques (by part):**  
- **Part 1:** Data exploration (features, median values, binary labels)  
- **Part 2:** Train/test split, visualization, effect of standardization  
- **Part 3:** Feature selection with Decision Tree importances, iterative KNN training with cross-validation, threshold-based stopping  
- Visualizations: standardized vs raw features, accuracy vs number of features, best `k` vs number of features  

**Learning Outcomes:**  
- Prepare data and evaluate the effect of scaling on distance-based models  
- Compare performance of KNN and Decision Trees  
- Implement wrapper-style feature selection using decision tree importances + KNN performance  
- Visualize how feature count impacts accuracy and optimal hyperparameters  
- Identify essential features for classification and justify based on CV performance  

---

## Project 2 — Fraud Detection with Gaussian Mixture Models
**Dataset:** [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Goal:** Detect fraudulent transactions in a highly imbalanced dataset (≈0.17% fraud).  

**Techniques (by part):**  
- **Part 1:** Data exploration, imbalance analysis, impact on classifiers  
- **Part 2:**  
  - Fit one-component Gaussian on single features  
  - Evaluate AUC for all features  
  - Threshold optimization for F1 Score  
  - Compare fitting on all data vs only non-fraud transactions  
- **Part 3:** Multi-feature Gaussian modeling, 2D scatter plots, selection of number of components, visual outlier detection  
- **Part 4:** Separate Gaussians for fraud vs non-fraud, tuning threshold \(c\) to maximize F1  

**Learning Outcomes:**  
- Understand how imbalance skews accuracy and why recall/precision are critical in fraud detection  
- Identify discriminative features via Gaussian distribution fit and AUC comparison  
- Implement anomaly detection with Gaussian Mixture Models at single- and multi-feature levels  
- Optimize thresholds systematically and analyze computational complexity  
- Visualize multi-dimensional feature spaces and mark fraud outliers  
- Compare modeling strategies (single vs dual Gaussians)  

---

## Project 3 — Dimensionality Reduction with PCA & SVD
**Dataset:** [Berkeley Earth Temperature dataset](https://berkeleyearth.org/data/)  
**Goal:** Analyze long-term temperature data (1992–2006) and extract dominant patterns.  

**Techniques (by part):**  
- **Part 2:** Compute covariance matrix, eigen-decomposition, scree plot, number of PCs to cover 99% variance  
- **Part 3:** Data reconstruction for selected cities using increasing numbers of PCs; plot residual error and RMSE vs number of components  
- **Part 4:** Use **Singular Value Decomposition (SVD)** instead of PCA; compare reconstructions and comment on results  

**Learning Outcomes:**  
- Compute covariance matrices and perform eigenvalue decomposition in NumPy  
- Evaluate variance explained and identify the cutoff for effective dimensionality reduction  
- Reconstruct time series from PCs, visualize incremental improvements, and calculate residual errors  
- Quantify reconstruction accuracy with RMSE as a function of PCs  
- Compare PCA vs SVD in practice and explain differences in requirements (e.g., standardization, covariance)  

---

## Project 4 — Gradient Descent & Regression
**Dataset:** [Online News Popularity dataset (UCI)](https://archive.ics.uci.edu/dataset/332/online+news+popularity)  
**Goal:** Predict article popularity (shares) on social media.  

**Techniques (by part):**  
- **Part 1:** Data preparation, train/test split, manual standardization, bias column handling  
- **Part 2:** Linear regression with closed-form (direct) solution  
- **Part 3:** Full-batch gradient descent implementation, convergence criteria, RMSE tracking, training time  
- **Part 4:** Mini-batch and stochastic gradient descent; compare convergence times and behaviors  
- **Part 5:** Experiment with learning rates across batch sizes, plot RMSE vs epoch/time, analyze results  

**Learning Outcomes:**  
- Implement linear regression both via closed-form and iterative gradient descent  
- Define and apply convergence thresholds relative to direct solution RMSE  
- Track and interpret training vs validation RMSE curves (detect under/overfitting)  
- Differentiate between epoch and iteration, especially in SGD/mini-batch settings  
- Analyze impact of learning rate and batch size on convergence speed and stability  
- Gain practical intuition for optimization trade-offs in regression models  

---
