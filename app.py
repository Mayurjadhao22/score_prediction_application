import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load trained SVR model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "svm.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Input Configuration defining Form UI Types & Categorical Option Mappings
FIELDS = [
    {"name": "age", "label": "Age", "type": "number", "step": "1", "min": "10", "max": "100"},
    {
        "name": "gender",
        "label": "Gender",
        "type": "select",
        "options": [("Male", 0), ("Female", 1), ("Other", 2)]
    },
    {
        "name": "course",
        "label": "Course",
        "type": "select",
        "options": [
            ("Arts", 0),
            ("Business", 1),
            ("Engineering", 2),
            ("Medicine", 3),
            ("Science", 4)
        ]
    },
    {"name": "study_hours", "label": "Study Hours (per week)", "type": "number", "step": "0.1", "min": "0"},
    {"name": "class_attendance", "label": "Class Attendance (%)", "type": "number", "step": "0.1", "min": "0", "max": "100"},
    {
        "name": "internet_access",
        "label": "Internet Access",
        "type": "select",
        "options": [("No", 0), ("Yes", 1)]
    },
    {"name": "sleep_hours", "label": "Sleep Hours (per day)", "type": "number", "step": "0.1", "min": "0"},
    {
        "name": "sleep_quality",
        "label": "Sleep Quality",
        "type": "select",
        "options": [("Poor (1)", 1), ("Fair (2)", 2), ("Good (3)", 3), ("Very Good (4)", 4), ("Excellent (5)", 5)]
    },
    {
        "name": "study_method",
        "label": "Study Method",
        "type": "select",
        "options": [
            ("Group Study", 0),
            ("Individual Study", 1),
            ("Online Lectures", 2),
            ("Self-Study", 3)
        ]
    },
    {
        "name": "facility_rating",
        "label": "Facility Rating",
        "type": "select",
        "options": [("1 Star", 1), ("2 Stars", 2), ("3 Stars", 3), ("4 Stars", 4), ("5 Stars", 5)]
    },
    {
        "name": "exam_difficulty",
        "label": "Exam Difficulty",
        "type": "select",
        "options": [("Easy (1)", 1), ("Moderate (2)", 2), ("Hard (3)", 3), ("Very Hard (4)", 4)]
    }
]

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
        }

        .input-group input, .input-group select {
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

        .input-group select option {
            background: var(--bg-color);
            color: var(--text-main);
        }

        .input-group input:focus, .input-group select:focus {
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
        <h1>SVR Performance Predictor</h1>
        <p>Select numerical and categorical features below to run evaluation</p>
    </div>

    <form method="POST" action="/predict" class="grid-form">
        {% for field in fields %}
        <div class="input-group">
            <label for="{{ field.name }}">{{ field.label }}</label>
            {% if field.type == 'select' %}
                <select id="{{ field.name }}" name="{{ field.name }}" required>
                    {% for option_label, option_val in field.options %}
                    <option value="{{ option_val }}">{{ option_label }}</option>
                    {% endfor %}
                </select>
            {% else %}
                <input type="number" 
                       step="{{ field.step }}" 
                       min="{{ field.get('min', '') }}" 
                       max="{{ field.get('max', '') }}" 
                       id="{{ field.name }}" 
                       name="{{ field.name }}" 
                       required 
                       placeholder="Enter value">
            {% endif %}
        </div>
        {% endfor %}
        <button type="submit" class="submit-btn">Predict Score</button>
    </form>

    {% if prediction is not none %}
    <div class="result-card">
        <h2>Predicted Output Result</h2>
        <div class="score">{{ prediction }}</div>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, fields=FIELDS, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features in exact expected numerical array structure
        features_vector = [float(request.form.get(field["name"], 0)) for field in FIELDS]
        prediction_val = model.predict(np.array([features_vector]))[0]
        formatted_prediction = f"{prediction_val:.2f}"
    except Exception as e:
        formatted_prediction = f"Error: {str(e)}"
        
    return render_template_string(HTML_TEMPLATE, fields=FIELDS, prediction=formatted_prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
