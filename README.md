# Liver Disease Prediction using Machine Learning

Predicted liver disease in patients using a Random Forest classification model. 
Performed end-to-end data cleaning, exploratory data analysis, and model evaluation 
achieving **91.56% accuracy**.

## Dataset
- **Records:** 583 patient records
- **Features:** Age, protein levels, bilirubin, enzyme markers, disease category
- **Source:** UCI / Kaggle Liver Patient Dataset

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 91.56% |
| Model | Random Forest Classifier |

## Features
- Data preprocessing & cleaning
- Categorical encoding
- Exploratory Data Analysis (EDA)
- Random Forest classification
- Feature importance analysis
- Visualizations:
  - Bar plot (average age by disease)
  - Strip plot (age distribution)
  - Box plot (bilirubin levels)
  - Feature importance chart
  - Correlation heatmap
  - Confusion matrix

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib

## How to Run
1. Clone the repository
```bash
git clone https://github.com/AdapakaGunaSekhar004/Liver-Disease-Prediction.git
```
2. Install dependencies
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```
3. Place the dataset file `project-data.csv` in the project directory
4. Run the script:
```bash
python liver_prediction.py
```

## Key Insights
- Bilirubin levels are the strongest indicator of liver disease
- Enzyme readings (SGPT, SGOT) show high correlation with disease presence
- Random Forest outperformed other models with 91.56% accuracy

## Author
**Adapaka Guna Sekhar**  
[LinkedIn](https://www.linkedin.com/in/guna-sekhar-adapaka-6903ab23b) | 
[GitHub](https://github.com/AdapakaGunaSekhar004) | 
[Portfolio](https://adapakagunasekhar004.github.io/portfolioper/)

> Note: This is an academic/portfolio project using sample medical data.
