from sklearn import datasets, model_selection
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

# Model loading functions
def load_random_forest():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def load_svm():
    return SVC(kernel='linear', random_state=42)

def load_gradient_boosting():
    return GradientBoostingClassifier(n_estimators=100, random_state=42)

def load_xgboost():
    return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42) 

def load_knn():
    return KNeighborsClassifier(n_neighbors=5)

def load_logistic_regression():
    return LogisticRegression(max_iter=1000, random_state=42)

def load_adaboost():
    return AdaBoostClassifier(n_estimators=100, random_state=42)

def load_lightgbm():
    return lgb.LGBMClassifier(n_estimators=100, random_state=42)

def load_catboost():
    return CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, silent=True)

def load_mlp():
    return MLPClassifier(max_iter=1000, random_state=42)

if __name__ == '__main__':
    # Load the breast cancer dataset
    wdbc = datasets.load_breast_cancer()

    # List of models to evaluate
    model_list = {
        'Random Forest': load_random_forest(),
        'SVM': load_svm(),
        'Gradient Boosting': load_gradient_boosting(),
        'XGBoost': load_xgboost(),
        'KNN': load_knn(),
        'Logistic Regression': load_logistic_regression(),
        'AdaBoost': load_adaboost(),
        'LightGBM': load_lightgbm(),
        'CatBoost': load_catboost(),
        'MLP': load_mlp()
    }

    # Evaluate each model
    for model_name, model in model_list.items():
        cv_results = model_selection.cross_validate(model, wdbc.data, wdbc.target, cv=5, return_train_score=True)

        # Calculate accuracies
        acc_train = np.mean(cv_results['train_score'])
        acc_test = np.mean(cv_results['test_score'])

        # Print the results
        print(f'{model_name} Results:')
        print(f'* Accuracy @ training data: {acc_train:.3f}')
        print(f'* Accuracy @ test data: {acc_test:.3f}')
        print(f'* Your score: {max(10 + 100 * (acc_test - 0.9), 0):.0f}')
        print('-' * 30)
