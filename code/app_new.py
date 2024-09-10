from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get form data
            data = {
                "seniorcitizen": int(request.form['seniorcitizen']),
                "partner": request.form['partner'],
                "dependents": request.form['dependents'],
                "tenure": int(request.form['tenure']),
                "multiplelines": request.form['multiplelines'],
                "internetservice": request.form['internetservice'],
                "onlinesecurity": request.form['onlinesecurity'],
                "onlinebackup": request.form['onlinebackup'],
                "deviceprotection": request.form['deviceprotection'],
                "techsupport": request.form['techsupport'],
                "streamingtv": request.form['streamingtv'],
                "streamingmovies": request.form['streamingmovies'],
                "contract": request.form['contract'],
                "paperlessbilling": request.form['paperlessbilling'],
                "paymentmethod": request.form['paymentmethod'],
                "monthlycharges": float(request.form['monthlycharges']),
                "totalcharges": float(request.form['totalcharges'])
            }
            
            # Debugging: Print received data
            print("Received data:", data)
            
            # Create DataFrame for prediction
            df = pd.DataFrame(data, index=[0])
            
            # Make prediction
            prediction = model.predict(df)
            prediction_proba = model.predict_proba(df)[:, 1]  # Probability of the positive class
            
            # Debugging: Print prediction and probability
            print("Prediction:", prediction, "Probability:", prediction_proba)
            
            result = {
                'prediction': 'Yes' if prediction[0] == 1 else 'No',
                'probability': float(prediction_proba[0])  # Ensure it's a float
            }
            
            return jsonify(result)  # Return JSON response
        except Exception as e:
            print("Error:", str(e))  # Print any errors
            return jsonify({'error': str(e)}), 400
    return render_template('index_new.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)