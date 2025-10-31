# 🌾 XAI-Crop: An Explainable Counterfactual Framework for Actionable Crop Recommendation

This project presents **XAI-Crop**, an interpretable and actionable machine learning framework for crop recommendation using **Explainable AI (XAI)** and **Counterfactual Analysis**.  
The system predicts optimal crop types based on soil and environmental parameters, while also explaining *why* a prediction was made and *how* to change inputs to achieve a desired output.

---

## 📋 Table of Contents

1. [Import Libraries](#import-libraries)  
2. [Load & Inspect Dataset & EDA](#load--inspect-dataset--eda)  
3. [Preprocessing](#preprocessing)  
4. [Utility Functions](#utility-functions)  
5. [Train All Models (Version-Safe)](#train-all-models-version-safe)  
   - RandomForest  
   - XGBoost  
   - LightGBM  
   - CatBoost  
6. [Global K-Fold Cross Validation Summary](#global-k-fold-cross-validation-summary)  
7. [Weighted Ensemble Model (Final Combination)](#weighted-ensemble-model-final-combination)  
8. [Final Model Comparison (Including Ensemble)](#final-model-comparison-including-ensemble)  
9. [Detailed 5-Fold Results per Model](#detailed-5-fold-results-per-model)  
10. [Ensemble Confusion Matrix + Error Analysis](#ensemble-confusion-matrix--error-analysis)  
11. [Model Accuracy Summary Visualization](#model-accuracy-summary-visualization)  
12. [Hyperparameter Tuning (Optuna) — Best Model](#hyperparameter-tuning-optuna--best-model)  
13. [SHAP Explainability](#shap-explainability)  
14. [Dimensionality Reduction: PCA, t-SNE, UMAP](#dimensionality-reduction-pca-t-sne-umap)  
15. [Manual Counterfactuals (Complete Analysis)](#manual-counterfactuals-complete-analysis)  
16. [Actionable Counterfactual Plan](#actionable-counterfactual-plan)  
17. [Model Fairness Check](#model-fairness-check)  
18. [Visualization: SHAP vs. Counterfactual Deltas](#visualization-shap-vs-counterfactual-deltas)

---

## 🚀 Overview

**XAI-Crop** combines multiple models — RandomForest, XGBoost, LightGBM, and CatBoost — into an ensemble that achieves high accuracy and interpretability.  
By integrating **SHAP** and **DiCE** (for counterfactual explanations), this framework not only predicts suitable crops but also suggests actionable changes in soil or environmental parameters to reach optimal outcomes.

---

## 🧠 Key Features

- Multi-model training and ensemble for improved accuracy  
- Global and fold-level performance comparison  
- **Optuna-based hyperparameter tuning** for the best model  
- **Explainable AI (SHAP)** integration for transparent predictions  
- **Counterfactual generation** for actionable insights  
- **Dimensionality reduction** using PCA, t-SNE, and UMAP
  <img width="723" height="547" alt="image" src="https://github.com/user-attachments/assets/708fcab4-8090-4cc3-a846-9a1e2f1eceb9" />
  <img width="712" height="528" alt="image" src="https://github.com/user-attachments/assets/dace4af6-3223-47ae-bac8-37e249b3306e" />
  <img width="712" height="528" alt="image" src="https://github.com/user-attachments/assets/d68d0b44-a070-4e28-8ba4-e1d8bd588356" />
- **Fairness and sensitivity checks** across features

---

## 📊 Dataset

- **Source:** [Crop Recommendation Dataset (Kaggle)](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset/data)  
- **Features:** N, P, K, Temperature, Humidity, pH, Rainfall  
- **Target:** Crop Type  

<img width="1073" height="470" alt="image" src="https://github.com/user-attachments/assets/b26d8160-021f-4e93-a85c-7208f89e6921" />

<img width="710" height="601" alt="image" src="https://github.com/user-attachments/assets/7443d543-b8a6-437b-8dad-7bd8525c58de" />

---

## 🧩 Technologies Used

- **Python**  
- **TensorFlow / Scikit-learn / CatBoost / XGBoost / LightGBM**  
- **Optuna** for hyperparameter optimization  
- **SHAP & LIME** for explainability  
- **DiCE** for counterfactual explanations  
- **Matplotlib / Seaborn / Plotly** for visualization  

---

## 🧾 Model Performance

### 🧮 Detailed Results of 5-Fold Validation (in %)

| Model           | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean  | Std  |
|-----------------|--------|--------|--------|--------|--------|-------|------|
| Random Forest   | 100.00 | 99.72  | 98.86  | 98.86  | 99.43  | 99.37 | 0.46 |
| XGBoost         | 99.15  | 98.86  | 99.72  | 99.15  | 98.86  | 99.15 | 0.31 |
| LightGBM        | 99.15  | 99.43  | 98.58  | 99.72  | 98.30  | 99.04 | 0.53 |
| CatBoost        | 99.15  | 99.72  | 98.01  | 98.86  | 99.15  | 98.98 | 0.56 |
| **Ensemble Model** | **100.00** | **100.00** | **99.15** | **99.43** | **99.15** | **99.55** | **0.30** |

---

### 📈 Comparative Performance Metrics of the Models

| Model           | Accuracy | Precision | Recall | F1-Score |
|-----------------|-----------|------------|---------|-----------|
| **Ensemble Model** | **0.9977** | **0.9978** | **0.9977** | **0.9977** |
| CatBoost        | 0.9977 | 0.9978 | 0.9977 | 0.9977 |
| RandomForest    | 0.9955 | 0.9957 | 0.9955 | 0.9955 |
| XGBoost         | 0.9886 | 0.9894 | 0.9886 | 0.9885 |
| LightGBM        | 0.9886 | 0.9891 | 0.9886 | 0.9886 |


<img width="1114" height="989" alt="image" src="https://github.com/user-attachments/assets/427c562f-e5b5-4e9e-a587-645c66c52dbf" />
<img width="1110" height="989" alt="image" src="https://github.com/user-attachments/assets/eef43c18-4a03-4227-a27a-20ac60e50d08" />


---

## 📈 Explainability

- **SHAP** visualizations highlight feature importance and model reasoning.
  <img width="756" height="547" alt="image" src="https://github.com/user-attachments/assets/a24602d2-3b72-43d5-bf0c-7524bca97102" />
  <img width="795" height="490" alt="image" src="https://github.com/user-attachments/assets/7088a02b-7246-4ce2-afba-fef65cefbd70" />
  <img width="718" height="490" alt="image" src="https://github.com/user-attachments/assets/17875d1b-e7e4-4fe5-a9b4-e67f9cfd5107" />

- **Counterfactual analysis** (DiCE) demonstrates how small changes in soil properties can alter crop recommendations.  
- Combined **SHAP vs. Counterfactual Delta** charts visualize the interpretability trade-offs.
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/1ddba5a1-e77b-4f76-9cc9-fff85c9db945" />

---

## 🔍 Notebook

Explore the full implementation here:  
[👉 Open in Google Colab](https://colab.research.google.com/drive/14FgHAon_9et9rjTe86BKtmzz5f2_tuH7#scrollTo=prElcSFjaB17)

---

## 👨‍💻 Author

**Abdullah Al Mahmud Joy**  
B.Sc. in Computer Science and Engineering  
Teaching Assistant at BUBT | AI & Deep Learning Enthusiast  

📫 [LinkedIn](https://www.linkedin.com/in/abdullah-al-mahmud-joy-359112202/)  
📧 abdullahalmahmudjoy39@gmail.com  

---

## 🏁 Citation

If you use this work, please cite as:
@misc{joy2025xaicrop,
title={XAI-Crop: An Explainable Counterfactual Framework for Actionable Crop Recommendation},
author={Abdullah Al Mahmud Joy},
year={2025},
url={https://colab.research.google.com/drive/14FgHAon_9et9rjTe86BKtmzz5f2_tuH7#scrollTo=prElcSFjaB17}

}

---

⭐ **If you find this project useful, please give it a star on GitHub!**


