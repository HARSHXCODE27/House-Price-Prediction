# 🏠 California Housing Price Prediction
A machine learning project to predict housing prices in California districts using census data. Built while following the Hands-On Machine Learning book by Aurélien Géron (O'Reilly).

# What this project does
Trains and compares regression models to predict the median house value of a California district based on features like location, income, population, and housing attributes.

# Dataset
California Housing dataset from the O'Reilly Hands-On Machine Learning book — based on the 1990 California census. Contains ~20,000 districts with features like median income, ocean proximity, total rooms, and population.

# Approach
## Data preparation

- Used stratified sampling based on median income to ensure the train/test split is representative
- Built preprocessing pipelines using scikit-learn's Pipeline for clean, reproducible transformations
- Handled missing values and scaled numerical features inside the pipeline

# Models trained

Linear Regression — baseline model
Random Forest Regressor — performed significantly better

# Fine tuning

Used GridSearchCV to find the best hyperparameters for the Random Forest model


# Tech stack

Python, Jupyter Notebook
pandas, NumPy
scikit-learn (Pipeline, GridSearchCV, RandomForestRegressor)
matplotlib


# What I learned

How to build proper ML pipelines that prevent data leakage between train and test sets
Why stratified splits matter when your data has skewed distributions
How Random Forest outperforms Linear Regression on non-linear relationships
How GridSearchCV automates hyperparameter tuning instead of manual guessing
