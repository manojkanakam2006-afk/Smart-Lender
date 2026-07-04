from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from predict import LoanPredictor
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
predictor = LoanPredictor()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'POST':
        try:
            # Get form data
            input_data = {
                'Gender': request.form.get('gender'),
                'Married': request.form.get('married'),
                'Dependents': request.form.get('dependents'),
                'Education': request.form.get('education'),
                'Self_Employed': request.form.get('self_employed'),
                'ApplicantIncome': float(request.form.get('applicant_income', 0)),
                'CoapplicantIncome': float(request.form.get('coapplicant_income', 0)),
                'LoanAmount': float(request.form.get('loan_amount', 0)),
                'Loan_Amount_Term': float(request.form.get('loan_term', 360)),
                'Credit_History': int(request.form.get('credit_history', 1)),
                'Property_Area': request.form.get('property_area')
            }
            
            # Make prediction
            result = predictor.predict(input_data)
            
            # Prepare input summary for display
            input_summary = {
                'Gender': request.form.get('gender'),
                'Married': request.form.get('married'),
                'Dependents': request.form.get('dependents'),
                'Education': request.form.get('education'),
                'Self_Employed': request.form.get('self_employed'),
                'ApplicantIncome': f"${float(request.form.get('applicant_income', 0)):,.2f}",
                'CoapplicantIncome': f"${float(request.form.get('coapplicant_income', 0)):,.2f}",
                'LoanAmount': f"${float(request.form.get('loan_amount', 0)):,.2f}",
                'LoanTerm': f"{float(request.form.get('loan_term', 360))} months",
                'CreditHistory': 'Good' if int(request.form.get('credit_history', 1)) == 1 else 'Bad',
                'PropertyArea': request.form.get('property_area')
            }
            
            return render_template('result.html', 
                                 result=result,
                                 input_summary=input_summary)
        
        except Exception as e:
            return render_template('predict.html', error=f"Error: {str(e)}")
    
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['Gender', 'Married', 'Dependents', 'Education', 
                          'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
                          'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Make prediction
        result = predictor.predict(data)
        
        return jsonify({
            'status': 'success',
            'prediction': result['status'],
            'probability_approved': result['probability_approved'],
            'probability_rejected': result['probability_rejected']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)