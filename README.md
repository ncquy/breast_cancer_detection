# Breast Cancer Classification Using Multiple Models

This project demonstrates how to classify breast cancer data using various machine learning models. The dataset used is the **Wisconsin Diagnostic Breast Cancer** dataset (wdbc), and a variety of classifiers from different libraries are employed to evaluate their performance through cross-validation. The project also includes a visual representation of the classification results (`wdbc_classification_cv.png`), which can be used to compare the model performance graphically.

## Table of Contents
- Project Structure
- Models
- Installation
- Usage
- Results
- Requirements
- Contact

## Project Structure

The project consists of the following components:

- **`main.py`**: The primary Python script that loads the dataset, trains multiple models, and evaluates them.
- **`requirements.txt`**: A file containing the required Python packages for this project.
- **`wdbc_classification_cv.png`**: An image summarizing the cross-validation results for the different models used in this project.

## Models

The following machine learning models are evaluated:

- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting Classifier
- XGBoost Classifier
- k-Nearest Neighbors (KNN)
- Logistic Regression
- AdaBoost Classifier
- LightGBM Classifier
- CatBoost Classifier
- Multilayer Perceptron (MLP)

Each model is loaded using a separate function to keep the code modular. The cross-validation procedure is performed using 5-fold cross-validation to evaluate both training and testing accuracy.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ncquy/breast_cancer_detection.git
   cd breast_cancer_detection

2. Install the required packages:
   ```bash
   pip install -r requirements.txt


## Usage
1. Run the Python script:
2. The script will evaluate the performance of all models on the breast cancer dataset using 5-fold cross-validation. Results will be printed to the terminal for each model, displaying:
   - Training Accuracy
   - Test Accuracy
   - Custom score based on test accuracy.
3. The evaluation results are saved as `wdbc_classification_cv.png`, providing a visual summary of the model performances.

## Results
The output for each model includes:
- Training accuracy
- Test accuracy
- A custom score based on the formula:
  ```scss
  Your score = max(10 + 100 * (accuracy_test - 0.9), 0)
The results after testing different models are summarized in the table below.
<p align='center'>
  <img width="800px" src="https://github.com/ncquy/breast_cancer_detection/blob/main/wdbc_classification_cv.png" />
  <br/>
  <i> The Breast Cancer Classification results after testing different models.</i>
</p>

### Authors
* [Nguyen Cong Quy](https://github.com/ncquy)

