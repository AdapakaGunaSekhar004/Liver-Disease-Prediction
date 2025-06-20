
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load data
data = pd.read_csv("project-data.csv", sep=';')

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Convert 'protein' to numeric
data['protein'] = pd.to_numeric(data['protein'], errors='coerce')

# Fill missing numeric values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Encode 'sex'
data['sex'] = data['sex'].map({'m': 0, 'f': 1})

# Encode 'category'
data['category'] = data['category'].str.strip()
le = LabelEncoder()
data['category_encoded'] = le.fit_transform(data['category'])

# Define features and target
X = data.drop(columns=['category', 'category_encoded'])
y = data['category_encoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy & report
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc*100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 📊 Bar Plot – Average Age per Disease
plt.figure(figsize=(8,5))
sns.barplot(x='category', y='age', data=data)
plt.xticks(rotation=45)
plt.title("Average Age by Disease Category")
plt.xlabel("Category")
plt.ylabel("Average Age")
plt.tight_layout()
plt.show()

# 🔴 Dot Plot – Age Distribution by Disease
plt.figure(figsize=(8,5))
sns.stripplot(x='category', y='age', data=data, jitter=True, palette="Set2")
plt.xticks(rotation=45)
plt.title("Age Distribution by Disease Category")
plt.xlabel("Category")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

# 📦 Box Plot – Bilirubin Levels by Category
plt.figure(figsize=(8,5))
sns.boxplot(x='category', y='bilirubin', data=data)
plt.xticks(rotation=45)
plt.title("Bilirubin Levels by Disease Category")
plt.xlabel("Category")
plt.ylabel("Bilirubin")
plt.tight_layout()
plt.show()

# 📉 Feature Importance Plot
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# 🔷 Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
