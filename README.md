# Real Estate Investment Advisor

## Project Overview

The Real Estate Investment Advisor is a machine learning project designed to analyze residential property data and provide investment insights.

The project performs two main tasks:

1. Predict whether a property is a Good Investment.
2. Predict the estimated property price after 5 years.

## Dataset

The project uses an India housing prices dataset containing property details such as:

- State
- City
- Locality
- Property Type
- BHK
- Size in SqFt
- Price
- Price per SqFt
- Year Built
- Furnished Status
- Floor Details
- Nearby Schools and Hospitals
- Public Transport Accessibility
- Parking
- Security
- Amenities
- Facing
- Owner Type
- Availability Status

## Machine Learning

### Classification

The classification task predicts whether a property is a:

- Good Investment
- Not a Good Investment

Several classification models were evaluated, including:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

The selected classification model for deployment is Random Forest Classifier.

### Regression

The regression task predicts the estimated property price after 5 years.

Models evaluated include:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

The selected regression model for deployment is Linear Regression.

## Streamlit Application

The project includes a Streamlit application with three main sections:

### 1. Introduction

Provides an overview of the project, its purpose, and technologies used.

### 2. EDA Visualizations

Displays exploratory data analysis and visualizations covering:

- Property price distribution
- Property size
- Price per square foot
- State and city analysis
- Locality analysis
- BHK distribution
- Correlation analysis
- Furnishing status
- Parking
- Amenities
- Public transport
- Investment-related factors

### 3. Prediction

Users can enter property details and receive:

- Investment Decision
- Model Confidence
- Estimated Future Price after 5 Years
- Price Growth Percentage

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- Joblib
- MLflow

## Project Structure

```text
Real_Estate_Investment_Advisor/
│
├── data/
│   └── india_housing_prices.csv
│
├── models/
│   ├── classification_preprocessor.pkl
│   ├── regression_preprocessor.pkl
│   ├── random_forest_classifier.pkl
│   └── linear_regression_model.pkl
│
├── notebooks/
│   └── 01_eda_and_preprocessing(7).ipynb
│
├── app.py
├── mlflow.db
├── requirements.txt
└── README.md