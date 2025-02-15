from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model
ensemble_pipeline = joblib.load("1.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs from the front-end
        data = request.json
        bedroom = float(data["bedroom"])
        layout_type = int(data["layout_type"])
        property_type = int(data["property_type"])
        area = float(data["area"])
        furnish_type = int(data["furnish_type"])
        bathroom = float(data["bathroom"])
        city = int(data["city"])

        # Map numeric inputs to their corresponding values
        layout_options = {1: 'BHK', 2: 'RK'}
        property_options = {1: 'Apartment', 2: 'Independent House'}
        furnish_options = {1: 'Furnished', 2: 'Semi-Furnished', 3: 'Unfurnished'}
        cities = {1: 'Ahmedabad', 2: 'Mumbai', 3: 'Delhi', 4: 'Bangalore', 5: 'Chennai'}

        layout_type = layout_options.get(layout_type, 'BHK')
        property_type = property_options.get(property_type, 'Apartment')
        furnish_type = furnish_options.get(furnish_type, 'Furnished')
        city = cities.get(city, 'Ahmedabad')

        # Create input data for prediction
        price_per_sqft = area  # Placeholder
        input_data = {
            'bedroom': [bedroom],
            'layout_type': [layout_type],
            'property_type': [property_type],
            'area': [area],
            'furnish_type': [furnish_type],
            'bathroom': [bathroom],
            'city': [city],
            'price_per_sqft': [price_per_sqft],
            'log_area': [np.log(area)]
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data)

        # Preprocess and predict
        preprocessor = ensemble_pipeline.named_steps['preprocessor']
        ensemble_model = ensemble_pipeline.named_steps['ensemble']

        input_preprocessed = preprocessor.transform(input_df)
        predicted_log_price = ensemble_model.predict(input_preprocessed)[0]
        predicted_price = np.exp(predicted_log_price)  # Reverse log transformation

        return jsonify({"predicted_price": f"₹{predicted_price:.2f}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)