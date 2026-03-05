# Liver Disease Prediction

A binary classification project that analyzes clinical blood test data from 30,691 patients to predict liver disease using Exploratory Data Analysis and machine learning — achieving 99.84% accuracy with a Random Forest classifier.

## Objective

Given 10 clinical features from blood tests, predict whether a patient has liver disease or not — enabling early detection to support clinical decision-making.

## Dataset

- Records: 30,691 patient records
- Features: 10 clinical variables (Age, Gender, Bilirubin, ALT, AST, Albumin, etc.)
- Target: Binary — Liver Disease (1) / No Disease (0)
- Class balance: 71.4% Liver Disease | 28.6% No Disease

## Tech Stack

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn

## Model Comparison

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 72.34% | 0.7631 |
| Gradient Boosting | 97.21% | 0.9956 |
| Extra Trees | 99.76% | 1.0000 |
| Random Forest | 99.84% | 1.0000 |

## Key Findings

- Alkaline Phosphotase, AST and ALT are the top 3 most important features
- Total and Direct Bilirubin levels are significantly elevated in liver disease patients
- Male patients show higher prevalence of liver disease
- Age group 40-60 has the highest concentration of liver disease cases

## Project Structure
```
Liver-Disease-Prediction/
├── Liver_Disease_Prediction.ipynb
├── Liver Patient Dataset (LPD)_train.csv
├── plot1_target_distribution.png
├── plot2_boxplots.png
├── plot3_correlation_heatmap.png
├── plot4_model_comparison.png
├── plot5_confusion_matrix.png
├── plot6_feature_importance.png
└── README.md
```

## How to Run
```bash
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
jupyter notebook Liver_Disease_Prediction.ipynb
```
