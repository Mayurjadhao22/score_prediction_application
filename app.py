import pickle
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load the pretrained SVR model from disk
MODEL_PATH = "model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model from {MODEL_PATH}: {e}")

# List of expected feature names based on the SVR model metadata
FEATURE_NAMES = [
    "age",
    "gender",
    "course",
    "study_hours",
    "class_attendance",
    "internet_access",
    "sleep_hours",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty",
]


@app.route("/")
def home():
    """Renders basic API status or front-end landing page."""
    return jsonify(
        {
            "status": "online",
            "message": "Student Performance Prediction API using SVR",
            "required_features": FEATURE_NAMES,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Accepts feature inputs via JSON payload and returns the SVR prediction."""
    if model is None:
        return (
            jsonify({"error": "Model file not loaded. Check model.pkl path."}),
            500,
        )

    try:
        data = request.get_json(force=True)

        # Extract feature values in exact required order
        input_features = []
        for feature in FEATURE_NAMES:
            if feature not in data:
                return (
                    jsonify(
                        {"error": f"Missing required feature field: {feature}"}
                    ),
                    400,
                )
            input_features.append(float(data[feature]))

        # Reshape input for model prediction: shape (1, 11)
        features_array = np.array(input_features).reshape(1, -1)

        # Run prediction
        prediction = model.predict(features_array)[0]

        return jsonify(
            {"status": "success", "prediction": float(prediction)}
        )

    except ValueError as ve:
        return (
            jsonify(
                {
                    "error": "Invalid input values. Ensure all inputs are numeric.",
                    "details": str(ve),
                }
            ),
            400,
        )
    except Exception as e:
        return (
            jsonify({"error": "An error occurred during prediction.", "details": str(e)}),
            500,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
