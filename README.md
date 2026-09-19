# House Price Prediction

## Project Overview

This project is an end-to-end Machine Learning regression project that predicts house prices based on different property features.

The project covers the complete Machine Learning workflow, starting from data ingestion and validation to model training, evaluation, prediction, and deployment using Streamlit.

---

## Problem Statement

The objective of this project is to predict the selling price of a house based on its property characteristics.

Since the target variable `price` is a continuous numerical value, this is a **Regression Problem**.

---

## Dataset

The dataset contains **545 records and 13 columns**.

### Features

- area
- bedrooms
- bathrooms
- stories
- mainroad
- guestroom
- basement
- hotwaterheating
- airconditioning
- parking
- prefarea
- furnishingstatus

### Target Variable

- `price`

---

## Project Objectives

- Understand the complete regression workflow
- Perform data ingestion and validation
- Perform data preprocessing and transformation
- Implement and compare different regression algorithms
- Perform hyperparameter tuning
- Evaluate models using appropriate regression metrics
- Select the best-performing model
- Build a reusable prediction pipeline
- Create a user interface using Streamlit
- Deploy the Machine Learning application

---

## Machine Learning Models

The following regression algorithms were implemented and compared:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree Regression
5. Random Forest Regression
6. XGBoost Regression
7. AdaBoost Regression

---

## Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV** for the applicable models.

The purpose of hyperparameter tuning was to find suitable parameter combinations that improve model performance.

---

## Model Evaluation

The models were evaluated using:

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² Score

### Model Comparison

| Model | R² Score |
|---|---:|
| XGBoost | 0.6589 |
| Linear Regression | 0.6529 |
| Lasso Regression | 0.6529 |
| Ridge Regression | 0.6477 |
| Random Forest | 0.6152 |
| AdaBoost | 0.5403 |
| Decision Tree | 0.4637 |

Based on the model comparison, **XGBoost Regression** was selected as the final model.

### Final Model

**XGBoost Regression**

**R² Score:** 0.6589

---
## Technologies Used

# Programming Language
  **Python**

# Libraries
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
Seaborn
Joblib
PyYAML

# Application
  **Streamlit**

# Development Tools
Jupyter Notebook
VS Code
Git
GitHub