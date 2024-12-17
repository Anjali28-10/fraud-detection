#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored on Dec 17, 2024

@author: Anjali
"""

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Set seaborn visual style for better aesthetics
sns.set()

# %% Data Loading and Visualization

data_path = "Data/synthetic-data-from-a-financial-payment-system/bs140513_032310.csv"
transaction_data = pd.read_csv(data_path)

# Display first 5 rows
transaction_data.head()

# Separate fraud and non-fraud records
data_fraud = transaction_data[transaction_data.fraud == 1]
data_non_fraud = transaction_data[transaction_data.fraud == 0]

# Count of fraudulent vs. non-fraudulent payments
sns.countplot(x="fraud", data=transaction_data)
plt.title("Fraudulent vs Non-Fraudulent Payments")
plt.show()

print("Non-Fraud Transactions: ", data_non_fraud.fraud.count())
print("Fraudulent Transactions: ", data_fraud.fraud.count())

# Average transaction amount by category
print("Mean transaction amount by category:\n", transaction_data.groupby('category')[['amount', 'fraud']].mean())

# Plot histograms of payment amounts for both classes
plt.hist(data_fraud.amount, alpha=0.5, label='Fraudulent', bins=100)
plt.hist(data_non_fraud.amount, alpha=0.5, label='Non-Fraudulent', bins=100)
plt.title("Distribution of Payment Amounts")
plt.ylim(0, 10000)
plt.xlim(0, 1000)
plt.legend()
plt.show()

# %% Data Preprocessing

# Drop columns with only one unique value
filtered_data = transaction_data.drop(['zipcodeOri', 'zipMerchant'], axis=1)

# Convert categorical features to category dtype
categorical_columns = filtered_data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    filtered_data[col] = filtered_data[col].astype('category')

# Convert categorical features to numeric codes
filtered_data[categorical_columns] = filtered_data[categorical_columns].apply(lambda x: x.cat.codes)

# Split data into features (X) and target (y)
X = filtered_data.drop(['fraud'], axis=1)
y = filtered_data['fraud']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# %% ROC-AUC Plot Function

def plot_roc_curve(actual, predictions):
    """
    Plot ROC curve given actual labels and prediction probabilities.
    """
    fpr, tpr, _ = roc_curve(actual, predictions)
    roc_auc_value = auc(fpr, tpr)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc_value)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', color='red')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

# Baseline performance to beat (predicting all non-fraud)
baseline_accuracy = data_non_fraud.fraud.count() / (data_non_fraud.fraud.count() + data_fraud.fraud.count())
print("Baseline accuracy to beat: {:.2f}%".format(baseline_accuracy * 100))

# %% K-Nearest Neighbors Classifier

knn_classifier = KNeighborsClassifier(n_neighbors=5, p=1)
knn_classifier.fit(X_train, y_train)

knn_predictions = knn_classifier.predict(X_test)

# Performance evaluation for KNN
print("KNN Classification Report:\n", classification_report(y_test, knn_predictions))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, knn_predictions))
plot_roc_curve(y_test, knn_classifier.predict_proba(X_test)[:, 1])

# %% Random Forest Classifier

rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

# Performance evaluation for Random Forest
print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))
plot_roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])

# %% XGBoost Classifier

xgb_classifier = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=400, objective="binary:hinge", random_state=42)
xgb_classifier.fit(X_train, y_train)

xgb_predictions = xgb_classifier.predict(X_test)

# Performance evaluation for XGBoost
print("XGBoost Classification Report:\n", classification_report(y_test, xgb_predictions))
print("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, xgb_predictions))
plot_roc_curve(y_test, xgb_classifier.predict_proba(X_test)[:, 1])

# %% Voting Classifier (Ensemble Model)

ensemble_estimators = [("KNN", knn_classifier), ("RF", rf_model), ("XGB", xgb_classifier)]
voting_model = VotingClassifier(estimators=ensemble_estimators, voting='soft', weights=[1, 4, 1])

voting_model.fit(X_train, y_train)
ensemble_predictions = voting_model.predict(X_test)

# Performance evaluation for Voting Classifier
print("Voting Classifier Report:\n", classification_report(y_test, ensemble_predictions))
print("Voting Classifier Confusion Matrix:\n", confusion_matrix(y_test, ensemble_predictions))
plot_roc_curve(y_test, voting_model.predict_proba(X_test)[:, 1])
