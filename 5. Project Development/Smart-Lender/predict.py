import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class LoanPredictor:
    def __init__(self):
        self.model = joblib.load('loan_model.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.label_encoders = joblib.load('label_encoders.pkl')
        self.feature_columns = joblib.load('feature_columns.pkl')
    
    def preprocess_input(self, input_data):
        """
        Preprocess user input for prediction
        input_data: dict with keys - Gender, Married, Dependents, Education, 
                    Self_Employed, ApplicantIncome, CoapplicantIncome, 
                    LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
        """
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                # Handle unseen labels
                try:
                    df[col] = self.label_encoders[col].transform(df[col])
                except ValueError:
                    # If unseen label, use most frequent class (0)
                    df[col] = 0
        
        # Calculate Total Income
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training data
        df = df[self.feature_columns]
        
        # Scale numerical features
        numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'TotalIncome']
        df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def predict(self, input_data):
        """Make prediction"""
        processed_data = self.preprocess_input(input_data)
        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)
        
        return {
            'prediction': int(prediction[0]),
            'status': 'Approved' if prediction[0] == 1 else 'Rejected',
            'probability_approved': float(probability[0][1]),
            'probability_rejected': float(probability[0][0])
        }

# Test the predictor
if __name__ == "__main__":
    predictor = LoanPredictor()
    
    # Sample test input
    test_input = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1,
        'Property_Area': 'Urban'
    }
    
    result = predictor.predict(test_input)
    print("Prediction Result:")
    print(f"Status: {result['status']}")
    print(f"Probability of Approval: {result['probability_approved']:.2%}")
    print(f"Probability of Rejection: {result['probability_rejected']:.2%}")