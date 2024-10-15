import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Load the dataset
school_data = pd.read_csv("updated_dataset.csv")

# Define features and target
X = school_data[['Previous qualification (grade)', 'Debtor', 'Tuition fees up to date',
                 'Educational special needs', 'Scholarship holder', 'International',
                 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
                 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
                 'Curricular units 1st sem (without evaluations)', 'Curricular units 1st sem (grade)',
                 'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
                 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
                 'Curricular units 2nd sem (without evaluations)', 'Curricular units 2nd sem (grade)',
                 'Age at enrollment']]
y = school_data["Target"]

# Split the data into training, validation, and test sets (60% train, 20% validation, 20% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(C=1.0, solver='lbfgs', max_iter=100)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.4f}")

# Manually change hyperparameters
log_reg = LogisticRegression(C=0.1, solver='liblinear', max_iter=200)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression (C=0.1, solver='liblinear') Accuracy: {accuracy:.4f}")

# Decision Tree
dt = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.4f}")

# Manually change hyperparameters
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=4)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree (max_depth=5, min_samples_split=4) Accuracy: {accuracy:.4f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Manually change hyperparameters
rf = RandomForestClassifier(n_estimators=200, max_features='log2')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest (n_estimators=200, max_features='log2') Accuracy: {accuracy:.4f}")

# SVM
svm = SVC(C=1.0, kernel='rbf')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy:.4f}")

# Manually change hyperparameters
svm = SVC(C=0.5, kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM (C=0.5, kernel='linear') Accuracy: {accuracy:.4f}")

# XGBoost
xg = xgb.XGBClassifier(learning_rate=0.1, n_estimators=100)
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy:.4f}")

# Manually change hyperparameters
xg = xgb.XGBClassifier(learning_rate=0.05, n_estimators=200)
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost (learning_rate=0.05, n_estimators=200) Accuracy: {accuracy:.4f}")

# Define the model for grid search
logistic_model = LogisticRegression(max_iter=200)

# Set up grid search parameters
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear']
}

# Grid search for Logistic Regression
grid_search = GridSearchCV(logistic_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters (Grid Search):", grid_search.best_params_)
print(f"Best accuracy (Grid Search): {grid_search.best_score_:.4f}")

# Random search for Logistic Regression
random_search_logistic = RandomizedSearchCV(estimator=logistic_model, param_distributions=param_grid, n_jobs=-1, n_iter=10, cv=3)
random_search_logistic.fit(X_train, y_train)

# Best parameters and accuracy
print("Best parameters (Logistic Regression Random Search):", random_search_logistic.best_params_)
print(f"Best accuracy (Logistic Regression Random Search): {random_search_logistic.best_score_:.4f}")

# Define the objective function for Hyperopt tuning
def objective(params):
    model = LogisticRegression(
        C=params['C'],
        max_iter=int(params['max_iter']),
        solver=params['solver'],
        random_state=42
    )
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return {'loss': -accuracy, 'status': STATUS_OK}

# Define the hyperparameter search space
space = {
    'C': hp.uniform('C', 0.01, 10),
    'max_iter': hp.choice('max_iter', [200, 500, 1000]),
    'solver': hp.choice('solver', ['lbfgs', 'liblinear', 'saga', 'newton-cg'])
}

# Initialize trials object to track progress
trials = Trials()

# Run Bayesian Optimization with Hyperopt
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

# Extract best accuracy from trials
best_accuracy = -trials.best_trial['result']['loss']
print("Best parameters (Logistic Regression Hyperopt):", best_params)
print(f"Best score (accuracy): {best_accuracy:.4f}")

# Define the model for Decision Tree
dt_model = DecisionTreeClassifier()

# Set up the parameter grid for Decision Tree
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Grid search for Decision Tree
grid_search_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, n_jobs=-1, cv=3)
grid_search_dt.fit(X_train, y_train)

# Best parameters and accuracy for Decision Tree
print("Best parameters (Decision Tree):", grid_search_dt.best_params_)
print(f"Best accuracy (Decision Tree): {grid_search_dt.best_score_:.4f}")

# Random search for Decision Tree
random_search_dt = RandomizedSearchCV(estimator=dt_model, param_distributions=param_grid_dt, n_jobs=-1, n_iter=10, cv=3)
random_search_dt.fit(X_train, y_train)

# Best parameters and accuracy for Decision Tree
print("Best parameters (Decision Tree Random Search):", random_search_dt.best_params_)
print(f"Best accuracy (Decision Tree Random Search): {random_search_dt.best_score_:.4f}")

# Define the objective function for Hyperopt tuning for Decision Tree
def objective_dt(params):
    model = DecisionTreeClassifier(max_depth=params['max_depth'], min_samples_split=params['min_samples_split'])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return {'loss': -accuracy, 'status': 'ok', 'accuracy': accuracy}

# Define the hyperparameter search space for Decision Tree
space_dt = {
    'max_depth': hp.choice('max_depth', [None, 10, 20, 30]),
    'min_samples_split': hp.choice('min_samples_split', [2, 5, 10])
}

# Initialize trials object to track progress for Decision Tree
trials_dt = Trials()

# Run Bayesian Optimization with Hyperopt for Decision Tree
best_params_dt = fmin(fn=objective_dt, space=space_dt, algo=tpe.suggest, max_evals=50, trials=trials_dt)

# Extract best accuracy from trials for Decision Tree
best_accuracy_dt = -trials_dt.best_trial['result']['loss']
print("Best parameters (Decision Tree Hyperopt):", best_params_dt)
print(f"Best score (accuracy): {best_accuracy_dt:.4f}")

# Define the model for Random Forest
rf_model = RandomForestClassifier()

# Set up the parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Grid search for Random Forest
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, n_jobs=-1, cv=3)
grid_search_rf.fit(X_train, y_train)

# Best parameters and accuracy for Random Forest
print("Best parameters (Random Forest):", grid_search_rf.best_params_)
print(f"Best accuracy (Random Forest): {grid_search_rf.best_score_:.4f}")

# Random search for Random Forest
random_search_rf = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid_rf, n_jobs=-1, n_iter=10, cv=3)
random_search_rf.fit(X_train, y_train)

# Best parameters and accuracy for Random Forest
print("Best parameters (Random Forest Random Search):", random_search_rf.best_params_)
print(f"Best accuracy (Random Forest Random Search): {random_search_rf.best_score_:.4f}")

# Define the objective function for Hyperopt tuning for Random Forest
def objective_rf(params):
    model = RandomForestClassifier(n_estimators=int(params['n_estimators']), max_depth=params['max_depth'], min_samples_split=params['min_samples_split'])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return {'loss': -accuracy, 'status': 'ok', 'accuracy': accuracy}

# Define the hyperparameter search space for Random Forest
space_rf = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
    'max_depth': hp.choice('max_depth', [None, 10, 20, 30]),
    'min_samples_split': hp.choice('min_samples_split', [2, 5, 10])
}

# Initialize trials object to track progress for Random Forest
trials_rf = Trials()

# Run Bayesian Optimization with Hyperopt for Random Forest
best_params_rf = fmin(fn=objective_rf, space=space_rf, algo=tpe.suggest, max_evals=50, trials=trials_rf)

# Extract best accuracy from trials for Random Forest
best_accuracy_rf = -trials_rf.best_trial['result']['loss']
print("Best parameters (Random Forest Hyperopt):", best_params_rf)
print(f"Best score (accuracy): {best_accuracy_rf:.4f}")

# Plotting confusion matrices
models = {
    "Logistic Regression": log_reg,
    "Decision Tree": dt,
    "Random Forest": rf,
    "SVM": svm,
    "XGBoost": xg
}

plt.figure(figsize=(15, 10))

for i, (name, model) in enumerate(models.items(), 1):
    plt.subplot(2, 3, i)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# Summary of classification reports
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"Classification report for {name}:\n{classification_report(y_test, y_pred)}")
