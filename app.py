import pickle
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load the trained SVR model saved in your pickle file
MODEL_PATH = "model.pkl"  # Rename your uploaded model file to model.pkl

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(
        f"Error loading model: {e}. Make sure '{MODEL_PATH}' is in the same directory."
    )

# List of features expected by the model in exact order
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
    """Renders the HTML form for user input."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handles predictions from web form inputs or JSON requests."""
    try:
        # Check if incoming request is JSON or HTML form submit
        if request.is_json:
            data = request.get_json()
            features = [float(data[feature]) for feature in FEATURE_NAMES]
        else:
            features = [
                float(request.form[feature]) for feature in FEATURE_NAMES
            ]

        # Convert feature array to 2D numpy array for prediction
        input_data = np.array([features])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Return JSON if requested, otherwise display result
        if request.is_json:
            return jsonify(
                {"status": "success", "prediction": round(float(prediction), 2)}
            )

        return render_template(
            "index.html", prediction_text=f"Predicted Score: {prediction:.2f}"
        )

    except Exception as e:
        if request.is_json:
            return jsonify({"status": "error", "message": str(e)}), 400
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
