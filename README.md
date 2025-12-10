# Classification of Microarray Gene Expression Data Using Machine Learning and Deep Learning Algorithms

This repository contains the source code (Jupyter Notebooks, `.ipynb`) and experiments from my undergraduate thesis *Classification of Microarray Gene Expression Data Using Machine Learning and Deep Learning Algorithms*.  

The project focuses on tumor classification using advanced computational methods applied to high-dimensional gene expression data derived from microarrays. The central goal is to perform a broad experimental comparison of Machine Learning (ML) and Deep Learning (DL) algorithms for cancer classification, using these models as an aid for medical diagnosis.

---

## Contents of the Notebooks (`.ipynb`)

The 5 notebooks in this repository correspond to detailed experiments for each standard cancer microarray dataset analyzed in the thesis.  

These datasets are characteristically challenging: they have a very large number of genes (features) and a small number of samples, and they differ in number of classes and degree of class imbalance.

**Datasets:**

1. Lung Cancer  
2. Prostate Cancer  
3. `11_Tumores` (multiclass)  
4. Leukemia  
5. SRBCT  

---

## Methodology and Computational Approach

The project implements a systematic comparison between six classification algorithms.

### Machine Learning (ML) Algorithms

- k-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Logistic Regression (LR)  
- Random Forest (RF)  

### Deep Learning (DL) Algorithms

- Multilayer Perceptron (MLP)  
- Convolutional Neural Network (CNN)  

### Key Techniques

- **Preprocessing and Dimensionality Reduction**  
  - Principal Component Analysis (PCA) to address high dimensionality and noise  
  - Feature scaling with `MinMaxScaler`  

- **Robust Validation**  
  - Stratified Cross-Validation (CV) to obtain more reliable and unbiased performance estimates  
  - Stratification was essential due to class imbalance, ensuring minority classes appeared in all folds  

- **Hyperparameter Optimization**  
  - Nested Cross-Validation for ML algorithms to reduce overfitting and select optimal hyperparameters  

---

## Key Results and Conclusions

The algorithms achieved excellent predictive performance, with accuracies up to 100% in many tests (especially with the Holdout method). The most robust models, across datasets, were:

1. **Logistic Regression (LR)** – ~96% (ML)  
2. **Convolutional Neural Network (CNN)** – ~95% (DL)  
3. **Support Vector Machine (SVM)** – ~94% (ML)  

The CNN achieved the best result on the challenging `11_Tumores` multiclass dataset (97% using Holdout) and consistently performed well by acting as a learned feature extractor. Overall, CNN, SVM, and LR delivered the best performance in most experiments.

### Critical Reflection

The thesis highlights that, although ML and DL techniques can reach very high accuracy (including 100% in some scenarios), these results may be overly optimistic due to the small sample size of microarray datasets.  

A key conclusion is that the goal in scientific work is not only to maximize accuracy with increasingly complex models, but also to prioritize interpretability and biological consistency. This motivates future work focused on interpretable models and deeper integration with the underlying biology.

---

## Tools and Libraries

- **Language:** Python  
- **Libraries:**  
  - Scikit-learn  
  - Keras  
  - TensorFlow  
