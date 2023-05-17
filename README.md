# Heart Disease Classification

## Introduction

This project focuses on the classification of heart disease based on various features. The goal is to develop machine learning models that can accurately predict the presence or absence of heart disease based on the given dataset.

The project consists of two components:

- Python Notebook: The Python notebook heart_disease_classification.ipynb contains the data analysis, model training, and evaluation process. It utilizes the heart.csv dataset for building and evaluating different machine learning models.

- Web App: The web app allows users to input custom data and make predictions using the trained models. It uses WebAssembly (WASM) to execute Python functions in the browser.

## Dataset

The dataset used in this project is named `heart.csv`. It contains various features related to heart health, such as age, sex, chest pain type, blood pressure, cholesterol levels, and more. The target variable indicates the presence or absence of heart disease.

It can be downloaded from kaggle from link.

## Notebook

The Python notebook `heart_disease_classification.ipynb` is the main component of this project. It provides detailed data analysis, model training, and evaluation steps. Here are some key sections covered in the notebook:

    - Data exploration and visualization
    - Preprocessing and feature engineering
    - Train-test split
    - Model training using various algorithms:
        - Logistic Regression
        - Naive Bayes
        - Support Vector Machine (SVM)
        - K-Nearest Neighbors (KNN)
        - Decision Tree
        - Random Forest
        - XGBoost
        - Neural Network
    - Evaluation of trained models
    - Comparison of model accuracies

## Web App

The web app component of the project allows users to interact with the trained models and make predictions based on custom input data. It utilizes WebAssembly (WASM) to execute Python functions in the browser, enabling real-time predictions without requiring server-side processing.

The web app is implemented using HTML, CSS, and JavaScript, with the Python functions  which `py-script` WebAssembly Framework for execution in the browser.

## Usage

To use this project, follow the instructions below:

- Clone the repository:

```
git clone https://github.com/basicfunc/heart-disease-prediction
```
    
- Open the Python notebook heart_disease_classification.ipynb in Jupyter Notebook or any compatible environment.

- Execute the notebook cells to run the data analysis, model training, and evaluation process.

- To use the web app, open the index.html file in a modern web browser.

Note: Make sure you have an active internet connection as the web app may require loading external dependencies.

- Enter the required input data in the web app and click the "Predict" button to obtain predictions from the trained models.

- Explore the notebook and web app to gain insights into heart disease classification and use the provided code as a reference or starting point for further customization.

## License

The source code and dataset in this project are licensed under the MIT License. Feel free to modify and adapt the code for your own purposes.
