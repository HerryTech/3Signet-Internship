import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU, BatchNormalization
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """Load the dataset."""
    return pd.read_csv(filepath)

def preprocess_data(school_data):
    """Preprocess the data."""
    X = school_data[['Previous qualification (grade)', 'Debtor', 'Tuition fees up to date', 
                     "Educational special needs", 'Scholarship holder', 'International',
                     'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
                     'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 
                     'Curricular units 1st sem (without evaluations)', 'Curricular units 1st sem (grade)',
                     'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 
                     'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 
                     'Curricular units 2nd sem (without evaluations)', 'Curricular units 2nd sem (grade)', 
                     'Age at enrollment']]
    y = school_data["Target"]
    return X, y

def split_data(X, y):
    """Split the data into training, validation, and test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """Train Decision Tree model."""
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """Train SVM model."""
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """Train XGBoost model."""
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def build_and_train_nn(X_train, y_train, epochs=50, batch_size=32):
    """Build and train a Neural Network model."""
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    return model, history

def build_and_train_deep_nn(X_train, y_train, epochs=50, batch_size=32):
    """Build and train a Deep Neural Network with multiple hidden layers."""
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    return model, history

def build_and_train_nn_with_dropout(X_train, y_train, epochs=50, batch_size=32):
    """Build and train a Neural Network with Dropout layers."""
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    return model, history

def build_and_train_nn_with_batch_norm(X_train, y_train, epochs=50, batch_size=32):
    """Build and train a Neural Network with Batch Normalization."""
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    return model, history

if __name__ == "__main__":
    # Load and preprocess the data
    filepath = "updated_dataset.csv"
    school_data = load_data(filepath)
    X, y = preprocess_data(school_data)
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Train models
    lr_model = train_logistic_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    svm_model = train_svm(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # Train Neural Networks
    sl_model, history_sl = build_and_train_nn(X_train, y_train)
    ml_model, history_ml = build_and_train_deep_nn(X_train, y_train)
    nd_model, history_nd = build_and_train_nn_with_dropout(X_train, y_train)
    nb_model, history_nb = build_and_train_nn_with_batch_norm(X_train, y_train)

    print("Model training completed.")
