# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 21:30:18 2024

@author: guldogtas
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from scipy import stats

# Loading the data
veriler = pd.read_excel('BankData_SMOTE.xlsx')

# Detecting and filling missing data
print("Number of missing values (for each column):")
print(veriler.isnull().sum())

# Using SimpleImputer to fill numerical columns
sayisal_sutunlar = veriler.select_dtypes(include=['float64', 'int64']).columns
imputer_sayisal = SimpleImputer(strategy='mean')  # Filling with the mean
veriler[sayisal_sutunlar] = imputer_sayisal.fit_transform(veriler[sayisal_sutunlar])

# Using SimpleImputer to fill categorical columns
kategorik_sutunlar = veriler.select_dtypes(include=['object']).columns
imputer_kategorik = SimpleImputer(strategy='most_frequent')  # Filling with the most frequent value
veriler[kategorik_sutunlar] = imputer_kategorik.fit_transform(veriler[kategorik_sutunlar])

# After missing data check
print("\nAfter missing data check (for each column):")
print(veriler.isnull().sum())

# Converting categorical columns to numerical values
label_encoders = {}
for column in ['Cinsiyet', 'Meslek', 'Eğitim Düzeyi', 'Medeni Durum', 'Konut', 'Araç', 'Arsa']:
    le = LabelEncoder()
    veriler[column] = le.fit_transform(veriler[column].astype(str))
    label_encoders[column] = le  # Storing the encoders

# Determining independent variables (X) and dependent variable (y)
X = veriler.drop(columns=['Kredi Notu'])  # Taking all columns except 'Kredi Notu' as independent variables
y = veriler['Kredi Notu']  # Taking 'Kredi Notu' as the dependent variable

# Splitting the data into 75% training and 25% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Standardization (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Scaling the training data
X_test_scaled = scaler.transform(X_test)  # Scaling the test data

# Developing the Tree Model (Increasing the number of trees, increasing depth, optimizing min_samples_split and min_samples_leaf)
random_forest_model = RandomForestRegressor(
    random_state=0, 
    n_estimators=200,  # We increased the number of trees.
    max_depth=10,  # Instead of reducing the depth, we increased this value.
    min_samples_split=5,  # The min. split value is larger.
    min_samples_leaf=2,  # By increasing the min. leaf count, we created more robust trees.
    n_jobs=-1  # To fully utilize the processor cores.
)

# Training the model
random_forest_model.fit(X_train_scaled, y_train)

# Predicting credit scores with the test data
y_pred_rf = random_forest_model.predict(X_test_scaled)

# Evaluating model performance (Random Forest)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest - Test Data MSE (Standardized):", mse_rf)
print("Random Forest - Test Data R² Score (Standardized):", r2_rf)

# Creating and training the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predicting credit scores with the test data
y_pred_lr = linear_model.predict(X_test_scaled)

# Evaluating model performance (Linear Regression)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression - Mean Squared Error (MSE):", mse_lr)
print("Linear Regression - R-Squared (R²) Score:", r2_lr)

