## utils/predictor.py
def make_prediction(cleaned_data, model, preprocessor):
    processed = preprocessor.transform(cleaned_data)
    prediction = model.predict(processed)
    return round(prediction[0], 2)
