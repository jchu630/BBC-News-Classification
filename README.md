# BBC-News-Classification

[BBC-News-Classification](

https://jchu630.github.io/BBC-News-Classification/BBC_News_Classification.html)


This project is a comparative study of supervised learning algorithms for text classification, applied to BBC news articles. The goal is to classify articles into two categories: Tech and Entertainment, using modern feature extraction techniques and multiple machine learning models.

## Project Overview

- **Objective:** Compare the performance of four popular supervised learning algorithms on a text classification task.
- **Algorithms Implemented:**
  - Naive Bayes (Multinomial)
  - k-Nearest Neighbors (kNN)
  - Support Vector Machines (SVM) – Linear and RBF kernels
  - Neural Networks (MLP)
- **Feature Engineering:** TF-IDF vectorization for model training; CountVectorizer for term frequency analysis.
- **Evaluation Metric:** Weighted F1 Score
- **Hyperparameter Tuning:** Cross-validation for key parameters (e.g., alpha for NB, k for kNN, C/gamma for SVM, hidden units for NN).

## Key Results

| Rank | Model | Best Hyperparameters | Test Weighted F1 Score | 
| --- | --- | --- | --- | 
| 1 | Neural Network | 1 hidden layer, units=45, activation=logistic | 0.9812 |
| 1 | SVM (linear) | C=1 | 0.9812 |
| 3 | kNN | k=7, Euclidean | 0.9811 |
| 4 | Naive Bayes | alpha=0.1 | 0.9718 |
| 5 | SVM (RBF) | C=10, gamma='scale' | 0.8873 |

## Features

- **Exploratory Data Analysis:** Term frequency analysis, class distribution visualization.
- **Model Development:** Full code for training, evaluation, and vizualization of decision boundaries.
- **Hyperparameter Analysis:** Impact of training size and parameter tuning on performance.
- **Final Comparison:** Summary table and discussion of results.

## Files in This Repository

- BBC_News_Classification.html – Full report with code, plots, and analysis.
- BBC_News_Classification.ipynb – Original Jupyter Notebook.
- train.csv, test.csv – Dataset files.

## Notes

- Dataset: Provided by Dr Jingfeng Zhang
- Environment: Python 3.12, scikit-learn, pandas, matplotlib, numpy.
