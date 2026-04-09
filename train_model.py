# ======================================
# Loan Prediction - Final Fixed Version
# ======================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load dataset
data = pd.read_excel("loan-predictionUC.csv (3) (2).xlsx")

# 2. Drop unnecessary column
data.drop("Loan_ID", axis=1, inplace=True)

# 3. FIX: Clean mixed data types

# Fix Dependents column
data['Dependents'] = data['Dependents'].replace('3+', '3')

# Convert ALL categorical columns to string
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].astype(str)

# 4. Separate features & target
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"].map({'Y': 1, 'N': 0})

# 5. Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# 6. Preprocessing pipelines

# Numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline (FIXED)
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Combine pipelines
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# 9. Full pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', model)
])

# 10. Train model
pipeline.fit(X_train, y_train)

# 11. Evaluate
y_pred = pipeline.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 12. Save model (IMPORTANT)
joblib.dump(pipeline, "loan_model.pkl")

print("\n✅ Model saved as loan_model.pkl")