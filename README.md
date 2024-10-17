# Spam Email Classifier using Naive Bayes

This project implements a **spam email classifier** using the **Naive Bayes** algorithm in **MATLAB**. The system classifies emails as either spam or non-spam using a suitable dataset, providing performance evaluation through a confusion matrix, ROC curve, and key metrics such as precision, recall, F1-score, and accuracy.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Visualization](#visualization)
- [Performance Evaluation](#performance-evaluation)
- [Results](#results)
  
## Overview
The spam email classifier uses the Naive Bayes algorithm, a probabilistic classifier, to predict whether an email is spam or not. This project includes training the model on a dataset, testing it, and evaluating its performance with various metrics. Two key visualizations are also included: the confusion matrix and the ROC curve.

## Features
- **Naive Bayes Classifier**: A simple, yet effective machine learning model that works well with high-dimensional data like email content.
- **Model Evaluation**: Provides metrics such as accuracy, precision, recall, and F1-score.
- **Visualizations**: Includes a confusion matrix and ROC curve to give a graphical view of the model’s performance.

## Dataset
The classifier uses the **SpamBase** dataset, a widely-used dataset for email spam detection. You can download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/spambase).

- **Features**: Various attributes of email content (such as word frequency).
- **Target**: Binary labels (`1` for spam and `0` for non-spam).

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ahmdmohamedd/spam-email-classifier.git
   cd spam-email-classifier
   ```

2. **Download the Dataset**:
   - Download the `spambase.data` file from the [UCI Repository](https://archive.ics.uci.edu/ml/datasets/spambase) and place it in the repository folder.

3. **Run MATLAB**:
   Open MATLAB, navigate to the cloned repository folder, and run the script.

## Usage
To train and evaluate the Naive Bayes classifier, follow these steps:

1. **Run the Script**: 
   Load the dataset, split it into training and testing sets, and train the Naive Bayes model.
   ```matlab
   run spam_classifier.m
   ```

2. **Output**: The script outputs key performance metrics and visualizations. The model evaluates the accuracy, precision, recall, and F1-score of the spam classifier.

## Visualization
The following visualizations are generated and saved:

1. **Confusion Matrix**:
   The confusion matrix shows the number of true positive, true negative, false positive, and false negative predictions.
   - Saved as: `confusion_matrix.png`
   
2. **ROC Curve**:
   The ROC curve plots the true positive rate (TPR) against the false positive rate (FPR) for different thresholds, providing insight into the model's discriminatory ability.
   - Saved as: `roc_curve.png`

## Results
Upon running the script, the system outputs the following:
- **Confusion Matrix**: Displays the breakdown of predictions.
- **Accuracy**: 0.81957
- **Precision**: 0.70319
- **Recall**: 0.95405
- **F1-Score**: 0.80963
- **Visualizations**: Confusion matrix and ROC curve are saved in PNG format for easy inspection.
