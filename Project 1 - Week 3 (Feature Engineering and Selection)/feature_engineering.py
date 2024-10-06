import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import chi2, RFE
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.utils import resample

# Load dataset
school_data = pd.read_csv("cleaned_transformed_dataset.csv")

# List of numerical variables
numerical_variables = [
    "Application order", "Previous qualification (grade)", "Admission grade", "Age at enrollment",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)", 
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)", 
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)", 
    "Unemployment rate", "Inflation rate", "GDP", 'Total Curricular Units 1st Semester',
    'Total Curricular Units 2nd Semester', 'Total Credits Earned', 'Total Units Enrolled',
    'Weighted Grade 1st Semester', 'Weighted Grade 2nd Semester', 'GPA'
]

# Step 1: Polynomial Feature Generation
poly = PolynomialFeatures(degree=2, interaction_only=True)
poly_features = poly.fit_transform(school_data[numerical_variables])

# Convert to DataFrame for easier viewing
poly_feature_names = poly.get_feature_names_out(numerical_variables)
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

# Step 2: Calculate skewness for numerical variables
skewness = school_data[numerical_variables].skew()

# Threshold for high skewness
threshold = 1

# List of highly skewed numerical variables
high_skewed_variables = skewness[abs(skewness) > threshold].index.tolist()

# Step 3: Apply log transformation to highly skewed variables
for col in high_skewed_variables:
    school_data[col] = np.log1p(school_data[col])  # log(1 + x) to avoid log(0)

# Step 4: Correlation-based feature selection
corr_matrix = school_data.corr()
target_corr = corr_matrix['Target']
selected_features = target_corr[abs(target_corr) >= 0.47].index.tolist()
selected_features.remove('Target')  # Exclude the target variable

print("Selected Features from Correlation:", selected_features)

# Define target variable
target = 'Target'
X = school_data.drop(columns=[target])  # Features
y = school_data[target]  # Target variable

# Step 5: Chi-Square test
chi2_values, p_values = chi2(X, y)

# Create a DataFrame for Chi-Square results
chi2_results = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Stat': chi2_values,
    'p-value': p_values
}).sort_values(by='Chi2 Stat', ascending=False)

# Features with p-value < 0.05
alpha = 0.05
significant_features = chi2_results[chi2_results['p-value'] < alpha]
print("\nSignificant Features (p-value < 0.05):\n", significant_features)

# Step 6: RFE with Logistic Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize model
model = LogisticRegression(max_iter=200)

# Initialize and fit RFE
rfe = RFE(estimator=model, n_features_to_select=5)
rfe.fit(X_scaled, y)

# Get selected features
rfe_selected_features = X.columns[rfe.support_]
print("Selected Features from RFE:\n", rfe_selected_features)

# Step 7: Lasso for feature selection
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

# Get Lasso-selected features
lasso_importance = pd.Series(lasso.coef_, index=X.columns)
lasso_selected_features = lasso_importance[lasso_importance != 0]
print("Selected Features from Lasso:\n", lasso_selected_features)

# Step 8: Stability Selection with Lasso
def stability_selection(X, y, n_bootstrap=100, threshold=0.5, alpha=0.01):
    """
    Perform stability selection using Lasso (L1-regularization).
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - n_bootstrap: Number of bootstrap iterations
    - threshold: Proportion of times a feature must be selected to be considered stable
    - alpha: Regularization strength for Lasso
    
    Returns:
    - Selected features (boolean mask)
    """
    n_samples, n_features = X.shape
    selected_counts = np.zeros(n_features)  # Track how many times each feature is selected
    
    for _ in range(n_bootstrap):
        # Resample the dataset
        X_resampled, y_resampled = resample(X, y, random_state=None)
        
        # Fit Lasso model to the resampled data
        lasso = Lasso(alpha=alpha, random_state=None, max_iter=10000)
        lasso.fit(X_resampled, y_resampled)
        
        # Check which features were selected (non-zero coefficients)
        selected = np.abs(lasso.coef_) > 1e-5
        selected_counts += selected
    
    # Features selected in more than threshold proportion of bootstraps
    stable_features = selected_counts / n_bootstrap > threshold
    return stable_features

# Perform stability selection
stable_features = stability_selection(X_scaled, y, n_bootstrap=100, threshold=0.5, alpha=0.01)

# Display selected features
stable_selected_features = X.columns[stable_features]
print("Selected Features from Stability Selection:\n", stable_selected_features)
