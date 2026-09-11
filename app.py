import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained SVR model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "svm.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Embedded HTML + CSS Layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --input-bg: #0f172a;
            --border: #334155;
            --shadow-soft: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
            --shadow-glow: 0 0 25px rgba(99, 102, 241, 0.35);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.15) 0px, transparent 50%);
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: var(--shadow-soft);
            transition: all 0.3s ease;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-sub);
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-sub);
            text-transform: capitalize;
        }

        .input-group input {
            background: var(--input-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        .input-group input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2), inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 0.9rem;
            background: linear-gradient(135deg, var(--primary), #818cf8);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-glow);
            background: linear-gradient(135deg, var(--primary-hover), var(--primary));
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 16px;
            text-align: center;
            box-shadow: var(--shadow-glow);
            animation: fadeIn 0.4s ease-in-out;
        }

        .result-card h2 {
            font-size: 1.1rem;
            color: var(--text-sub);
            margin-bottom: 0.5rem;
        }

        .result-card .score {
            font-size: 2.5rem;
            font-weight: 700;
            color: #38bdf8;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>SVR Predictor</h1>
        <p>Enter the student metrics below to evaluate the model response</p>
    </div>

    <form method="POST" action="/predict" class="grid-form">
        {% for feature in features %}
        <div class="input-group">
            <label for="{{ feature }}">{{ feature.replace('_', ' ') }}</label>
            <input type="number" step="any" id="{{ feature }}" name="{{ feature }}" required placeholder="0.0">
        </div>
        {% endfor %}
        <button type="submit" class="submit-btn">Run Prediction</button>
    </form>

    {% if prediction is not none %}
    <div class="result-card">
        <h2>Predicted Output</h2>
        <div class="score">{{ prediction }}</div>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

FEATURES = [
    "age", "gender", "course", "study_hours", "class_attendance",
    "internet_access", "sleep_hours", "sleep_quality", "study_method",
    "facility_rating", "exam_difficulty"
]

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, features=FEATURES, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = [float(request.form.get(feat, 0)) for feat in FEATURES]
        prediction_val = model.predict(np.array([input_data]))[0]
        formatted_prediction = f"{prediction_val:.2f}"
    except Exception as e:
        formatted_prediction = f"Error: {str(e)}"
        
    return render_template_string(HTML_TEMPLATE, features=FEATURES, prediction=formatted_prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
