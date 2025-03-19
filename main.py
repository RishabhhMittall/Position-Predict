from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model and label encoder
with open("position_model.pkl", "rb") as model_file:
    position_model = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Initialize Flask app
app = Flask(__name__)

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the JSON request
        input_data = request.get_json()

        # Check if all required features are present
        features = [
            "overall", "potential", "value_eur", "wage_eur", "pace",
            "shooting", "passing", "dribbling", "defending", "physic", "skill_moves", "BMI"
        ]
        if not all(feature in input_data for feature in features):
            return jsonify({"error": "Missing required features in input data"}), 400

        # Convert input data into a numpy array
        input_values = [float(input_data[feature]) for feature in features]
        input_array = np.array([input_values])

        # Make prediction
        prediction_encoded = position_model.predict(input_array)
        predicted_position = label_encoder.inverse_transform(prediction_encoded)[0]

        # Return the result as JSON
        return jsonify({"predicted_position": predicted_position})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)