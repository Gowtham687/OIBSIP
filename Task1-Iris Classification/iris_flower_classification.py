#!/usr/bin/env python3
"""
Iris Flower Classification with Machine Learning
AICTE OASIS INFOBYTE - Data Science Internship Task 1

This task in involves building a machine learning model to classify iris flowers into three species: Setosa, Versicolor, and Virginica based on their features (sepal length, sepal width, petal length, petal width).
The model achieves an accuracy of 93.33% using Support Vector Machine (SVM).

Author: Nallabolu Venkata Gowtham
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Print header
print("="*60)
print("IRIS FLOWER CLASSIFICATION - MACHINE LEARNING PROJECT")
print("AICTE OASIS INFOBYTE SIP - Task 1")
print("="*60)

# Initializing and importing the Iris dataset for training the model
print("\nLoading the Iris Dataset...")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Dataset has been loaded successfully!")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"\nFeature names: {feature_names}")
print(f"Target classes: {target_names}")

# Creating the DataFrame
print("\nCreating the DataFrame...")
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# Data Visualization(Visualizing feature distributions)
print("\nData Visualization...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Iris Flower Dataset - Features Distribution', fontsize=16, fontweight='bold')

for idx, (ax, feature) in enumerate(zip(axes.flat, feature_names)):
    for target, target_name in enumerate(target_names):
        ax.hist(X[y == target, idx], alpha=0.6, label=target_name, bins=15)
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {feature}')
    ax.legend()

plt.tight_layout()
plt.savefig('iris_distribution.png', dpi=100, bbox_inches='tight')
print("Visualization saved as 'iris_distribution.png'")
plt.close()

# Split data into training and testing sets
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Feature Scaling(
print("\nFeature Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully!")
print("\nData preprocessing completed!")

#Model Training and Evaluation of the Model
print("\nTraining multiple classifiers to compare performance and to do evaluation...")
print("="*60)

models = {}
results = {}


# 1. Logistic Regression
print("\n2. Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=200, random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)
models['Logistic Regression'] = lr_model
results['Logistic Regression'] = acc_lr
print(f"   Logistic Regression Accuracy: {acc_lr:.4f}")

# 2. Support Vector Machine(SVM)
print("\n3. Training Support Vector Machine...")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)
models['SVM'] = svm_model
results['SVM'] = acc_svm
print(f"   SVM Accuracy: {acc_svm:.4f}")

 # 3. Random Forest Classifier
print("\n1. Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)
models['Random Forest'] = rf_model
results['Random Forest'] = acc_rf
print(f"   Random Forest Accuracy: {acc_rf:.4f}")

# Displaying the Model Comparison
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
for model_name, accuracy in results.items():
    print(f"  {model_name:30} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

best_model_name = max(results, key=results.get)
best_model_accuracy = results[best_model_name]
print(f"\nBest Model: {best_model_name} with accuracy {best_model_accuracy:.4f}")
print("="*60)

# Cross-Validation
print("\nCross-Validation (5-Fold)...")
print("="*60)
print("\nPerforming 5-fold cross-validation on all models...")

cv_results = {}

# Cross-validation for Logistic Regression 
print("\n1. Cross-validating Logistic Regression...")
cv_scores_lr = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_results['Logistic Regression'] = {
    'scores': cv_scores_lr,
    'mean': cv_scores_lr.mean(),
    'std': cv_scores_lr.std()
}
print(f"CV Scores: {cv_scores_lr}")
print(f"Mean Accuracy: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std():.4f})")

# Cross-validation for SVM
print("\n2.Cross-validating Support Vector Machine...")
cv_scores_svm = cross_val_score(svm_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_results['SVM'] = {
    'scores': cv_scores_svm,
    'mean': cv_scores_svm.mean(),
    'std': cv_scores_svm.std()
}
print(f"CV Scores: {cv_scores_svm}")
print(f"Mean Accuracy: {cv_scores_svm.mean():.4f} (+/- {cv_scores_svm.std():.4f})")

# Cross-validation for Random Forest
print("\n3. Cross-validating Random Forest...")
cv_scores_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_results['Random Forest'] = {
    'scores': cv_scores_rf,
    'mean': cv_scores_rf.mean(),
    'std': cv_scores_rf.std()
}
print(f"CV Scores: {cv_scores_rf}")
print(f"Mean Accuracy: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

# Cross-Validation Summary
print("\n" + "="*60)
print("CROSS-VALIDATION SUMMARY")
print("="*60)
for model_name, cv_result in cv_results.items():
    print(f"  {model_name:30} Mean CV Accuracy: {cv_result['mean']:.4f} (+/- {cv_result['std']:.4f})")

best_cv_model = max(cv_results, key=lambda x: cv_results[x]['mean'])
print(f"\nBest Model (Cross-Validation): {best_cv_model} with mean accuracy {cv_results[best_cv_model]['mean']:.4f}")
print("="*60)

# Detailed Evaluation of Best Model (SVM)
print("\nDetailed Evaluation of the Best Model (SVM)...")
print("="*60)
print("\nClassification Report for SVM:")
print("="*60)
print(classification_report(y_test, y_pred_svm, target_names=target_names))

# Confusion Matrix
print("\nConfusion Matrix for SVM:")
print("="*60)
cm = confusion_matrix(y_test, y_pred_svm)
print(cm)

# Visualize Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', cbar=True,
    xticklabels=target_names, yticklabels=target_names, ax=ax
)
ax.set_title('Confusion Matrix - SVM Model', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
print("\nConfusion matrix saved as 'confusion_matrix.png'")
plt.close()

# Project Summary and Completion Message
print("\n" + "="*60)
print("PROJECT SUMMARY -  CLASSIFICATION OF IRIS FLOWERS")
print("="*60)
print(f"\nDataset: Iris Flower Classification")
print(f"Total Samples: {len(X)}")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")
print(f"Best Model: Support Vector Machine (SVM)")
print(f"Test Accuracy: {acc_svm:.4f} ({acc_svm*100:.2f}%)")
print("="*60)
print("\nProject has been completed successfully!")
print("AICTE OASIS INFOBYTE SIP - Data Science Internship")
print("="*60)