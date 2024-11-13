# Project 1 - Clustering Analysis Project

This project focuses on clustering analysis as an initial phase of a larger AI or data science initiative. It aims to find a suitable clustering algorithm, visualize clusters, and evaluate clustering performance.

## Project Overview

1. **Clustering and Visualization**  
   - Identify and apply a clustering algorithm that best fits the dataset.
   - Visualize clusters by plotting data points with distinct colors.
   - Adjust clustering parameters to assess the quality of the clusters visually.

2. **Domain Identification**  
   - Determine the number of distinct domains represented by the data.
   - Analyze which domain each cluster signifies, helping to inform further data processing steps.

3. **Cluster Evaluation**  
   - Evaluate clusters using labeled data through train-test splits.
   - Assess clustering performance by measuring accuracy against test data labels.

# Project 2 - Multi-Stage Classification with Random Forest

This project focuses on a two-stage classification process using the Random Forest algorithm. The goal is to classify a dataset of digits (0-9) across five distinct domains, then evaluate the impact of cross-domain data on classification accuracy.

## Project Overview

1. **Primary Classification**
   - Use Random Forest to classify numbers (0-9) across the dataset, disregarding domain initially.
   - Assign each classified number to one of the five specific domains.

2. **Domain-Specific Classification**
   - After identifying the domain, apply a secondary classification for each domain to determine the specific type of number.

3. **Cross-Domain Training Evaluation**
   - Test the effect of training with data from other domains:
     - Investigate whether using cross-domain data in training improves classification performance.
     - Compare results for models trained with and without cross-domain data to assess the benefits.

## Evaluation Metrics

- Accuracy of primary and secondary classifications.
- Comparison of results for models with and without cross-domain training data.

This project highlights the benefits of a two-stage classification approach and investigates the influence of cross-domain data on model performance.

# Project 3 - MLP-SVM Feature Selection and Availability Score Prediction

This project combines a Multi-Layer Perceptron (MLP) neural network with feature selection techniques to determine the best features for predicting an availability score using the Support Vector Machine (SVM) algorithm. The goal is to optimize the MLP network to find the best weights and features, then use the SVM algorithm to predict the availability score based on the selected features and the amount of penalty applied.

## Project Overview

1. **MLP Network**: 
   - Train an MLP model to learn the weights of the input features.
   - The MLP is trained to capture the relationship between the input features and the output variable (availability score).

2. **Feature Selection**: 
   - Use feature selection methods to identify the most relevant features for the task.
   - The aim is to reduce the dimensionality and improve the prediction performance by selecting the most informative features.

3. **SVM for Availability Score**: 
   - After selecting the best features, train an SVM model to predict the availability score.
   - The SVM model uses the selected features and the penalty (regularization) parameter to make predictions.
   
4. **Penalty (C parameter)**: 
   - Adjust the penalty term in the SVM algorithm to control the trade-off between achieving a low error on the training data and maintaining a simple model to avoid overfitting.


A report of each phase is included in the Report folder

## Getting Started

1. **Clone the Repository**: `git clone <repository-url>`
2. **Run Clustering Scripts**: Execute clustering and visualization with the provided scripts.
3. **Evaluate Clustering**: Use labeled data for testing and validating clustering results.
