from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model and preprocessor
model_path = os.path.join('model', 'model.pkl')
preprocessor_path = os.path.join('model', 'preprocessor.pkl')

model = pickle.load(open(model_path, 'rb'))
preprocessor = pickle.load(open(preprocessor_path, 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'area_type': request.form['area_type'],
        'availability': request.form['availability'],
        'location': request.form['location'],
        'size(bhk)': int(request.form['size']),
        'total_sqft': float(request.form['total_sqft']),
        'bath': int(request.form['bath']),
        'balcony': int(request.form['balcony'])
    }

    input_df = pd.DataFrame([input_data])
    processed_input = preprocessor.transform(input_df)
    prediction = model.predict(processed_input)

    print("Prediction Output:", prediction)  # Debug print
    predicted_price = round(prediction[0], 2)
    return render_template('result.html', price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
