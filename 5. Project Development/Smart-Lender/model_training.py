import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('loan_prediction.csv')
print("Data loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Create plots directory
import os
os.makedirs('plots', exist_ok=True)

# Exploratory Data Analysis and Visualizations

# 1. Gender Distribution
plt.figure(figsize=(8, 5))
gender_counts = df['Gender'].value_counts()
gender_counts.plot(kind='bar', color=['#3498db', '#e74c3c'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/gender_distribution.png')
plt.close()

# 2. Education Distribution
plt.figure(figsize=(8, 5))
edu_counts = df['Education'].value_counts()
edu_counts.plot(kind='bar', color=['#2ecc71', '#e67e22'])
plt.title('Education Distribution')
plt.xlabel('Education')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/education_distribution.png')
plt.close()

# 3. Loan Status Distribution
plt.figure(figsize=(8, 5))
status_counts = df['Loan_Status'].value_counts()
colors = ['#2ecc71' if x == 'Y' else '#e74c3c' for x in status_counts.index]
status_counts.plot(kind='bar', color=colors)
plt.title('Loan Status Distribution')
plt.xlabel('Loan Status (Y=Approved, N=Rejected)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/loan_status_distribution.png')
plt.close()

# 4. Income Distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
df['ApplicantIncome'].plot(kind='hist', bins=30, color='#3498db', edgecolor='black')
plt.title('Applicant Income Distribution')
plt.xlabel('Applicant Income')

plt.subplot(1, 2, 2)
df['CoapplicantIncome'].plot(kind='hist', bins=30, color='#e67e22', edgecolor='black')
plt.title('Co-applicant Income Distribution')
plt.xlabel('Co-applicant Income')
plt.tight_layout()
plt.savefig('plots/income_distribution.png')
plt.close()

# 5. Credit History vs Loan Status
plt.figure(figsize=(8, 5))
credit_status = pd.crosstab(df['Credit_History'], df['Loan_Status'])
credit_status.plot(kind='bar', color=['#e74c3c', '#2ecc71'])
plt.title('Credit History vs Loan Status')
plt.xlabel('Credit History (1=Good, 0=Bad)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(['Rejected', 'Approved'])
plt.tight_layout()
plt.savefig('plots/credit_history_analysis.png')
plt.close()

print("Visualizations saved in 'plots' directory!")

# Data Preprocessing
df_clean = df.copy()

# Handle missing values
# Gender
df_clean['Gender'].fillna(df_clean['Gender'].mode()[0], inplace=True)

# Married
df_clean['Married'].fillna(df_clean['Married'].mode()[0], inplace=True)

# Dependents
df_clean['Dependents'].fillna(df_clean['Dependents'].mode()[0], inplace=True)

# Self_Employed
df_clean['Self_Employed'].fillna(df_clean['Self_Employed'].mode()[0], inplace=True)

# LoanAmount
df_clean['LoanAmount'].fillna(df_clean['LoanAmount'].median(), inplace=True)

# Loan_Amount_Term
df_clean['Loan_Amount_Term'].fillna(df_clean['Loan_Amount_Term'].mode()[0], inplace=True)

# Credit_History
df_clean['Credit_History'].fillna(df_clean['Credit_History'].mode()[0], inplace=True)

# Education - already has no missing values

# Remove Loan_ID as it's not useful for prediction
df_clean = df_clean.drop('Loan_ID', axis=1)

# Convert categorical variables to numerical
le_dict = {}
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    le_dict[col] = le

# Target variable
df_clean['Loan_Status'] = df_clean['Loan_Status'].map({'Y': 1, 'N': 0})

# Feature Engineering - Total Income
df_clean['TotalIncome'] = df_clean['ApplicantIncome'] + df_clean['CoapplicantIncome']

# Drop individual income columns (optional, but helps with multicollinearity)
# df_clean = df_clean.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1)

# Prepare features and target
X = df_clean.drop('Loan_Status', axis=1)
y = df_clean['Loan_Status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'TotalIncome']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = rf_model.predict(X_train_scaled)
y_pred_test = rf_model.predict(X_test_scaled)

# Evaluation
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
print(f"Training Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")

print("\n" + "-"*50)
print("Classification Report (Test Set):")
print("-"*50)
print(classification_report(y_test, y_pred_test, target_names=['Rejected', 'Approved']))

print("\n" + "-"*50)
print("Confusion Matrix:")
print("-"*50)
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "-"*50)
print("Feature Importance:")
print("-"*50)
print(feature_importance)

# Save model, scaler, and label encoders
joblib.dump(rf_model, 'loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_dict, 'label_encoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

print("\n" + "="*50)
print("Model saved successfully!")
print("Files saved: loan_model.pkl, scaler.pkl, label_encoders.pkl, feature_columns.pkl")
print("="*50)

# Feature Importance Plot
plt.figure(figsize=(10, 6))
feature_importance.plot(x='Feature', y='Importance', kind='bar', legend=False, color='#3498db')
plt.title('Feature Importance for Loan Prediction')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()