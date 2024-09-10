from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
   
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the JSON data from the request
            data = request.json

            # Ensure all required features are present
            required_features = ['seniorcitizen', 'partner', 'dependents', 'tenure', 'multiplelines', 
                             'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection', 
                             'techsupport', 'streamingtv', 'streamingmovies', 'contract', 
                             'paperlessbilling', 'paymentmethod', 'monthlycharges', 'totalcharges']
        
            for feature in required_features:
                if feature not in data:
                    return jsonify({'error': f'Missing feature: {feature}'}), 400

            # Convert JSON to DataFrame
            df = pd.DataFrame(data, index=[0])

            # Make prediction
            prediction = model.predict(df)
            prediction_proba = model.predict_proba(df)[:, 1]

            # Prepare the response
            response = {
                'prediction': 'Yes' if prediction[0] == 1 else 'No',
                'probability': float(prediction_proba[0])
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 400
    else:
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 
    