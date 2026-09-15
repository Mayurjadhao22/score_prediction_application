import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Score & Customer Response Predictor",
    page_icon="🔮",
    layout="centered",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


@st.cache_resource
def load_model_from_file(file_path):
    """Load pickle model from a file path."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model_from_bytes(uploaded_file):
    """Load pickle model from Streamlit file uploader bytes."""
    return pickle.load(uploaded_file)


def main():
    st.title("🔮 Score Prediction Application")
    st.write(
        "Enter customer demographic and socioeconomic details below to predict their response."
    )

    model = None

    # Resolve model file path or request file upload
    if os.path.exists(DEFAULT_MODEL_PATH):
        model = load_model_from_file(DEFAULT_MODEL_PATH)
    else:
        st.warning(f"`model.pkl` was not found at `{DEFAULT_MODEL_PATH}`.")
        uploaded_model = st.file_uploader(
            "Please upload your `model.pkl` file to continue:",
            type=["pkl", "pickle"],
        )
        if uploaded_model is not None:
            model = load_model_from_bytes(uploaded_model)
            st.success("Model loaded successfully from upload!")
        else:
            st.info("Upload a model file above to enable predictions.")
            st.stop()

    st.subheader("Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        marital_status = st.selectbox(
            "Marital Status", options=["Single", "Married", "Divorced", "Widowed"]
        )
        occupation = st.selectbox(
            "Occupation",
            options=["Student", "Employee", "Self Employed", "Unemployed", "Retired"],
        )

    with col2:
        monthly_income = st.selectbox(
            "Monthly Income",
            options=[
                "No Income",
                "Below 10,000",
                "10,001 to 25,000",
                "25,001 to 50,000",
                "More than 50,000",
            ],
        )
        educational_qualifications = st.selectbox(
            "Educational Qualifications",
            options=["School", "Under Graduate", "Post Graduate", "Ph.D"],
        )
        family_size = st.number_input(
            "Family Size", min_value=1, max_value=20, value=3, step=1
        )
        customer_type = st.selectbox(
            "Customer Type", options=["New", "Existing", "Frequent"]
        )

    # Encode categorical features
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    marital_map = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}
    occupation_map = {
        "Student": 0,
        "Employee": 1,
        "Self Employed": 2,
        "Unemployed": 3,
        "Retired": 4,
    }
    income_map = {
        "No Income": 0,
        "Below 10,000": 1,
        "10,001 to 25,000": 2,
        "25,001 to 50,000": 3,
        "More than 50,000": 4,
    }
    edu_map = {"School": 0, "Under Graduate": 1, "Post Graduate": 2, "Ph.D": 3}
    cust_type_map = {"New": 0, "Existing": 1, "Frequent": 2}

    input_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender_map[gender],
                "Marital Status": marital_map[marital_status],
                "Occupation": occupation_map[occupation],
                "Monthly Income": income_map[monthly_income],
                "Educational Qualifications": edu_map[educational_qualifications],
                "Family size": family_size,
                "Customer Type": cust_type_map[customer_type],
            }
        ]
    )

    st.markdown("---")

    if st.button("Predict Score", type="primary"):
        try:
            prediction = model.predict(input_data)[0]

            st.subheader("Prediction Result")
            if prediction == "Yes" or prediction == 1:
                st.success(f"**Result:** Positive Response ({prediction})")
            else:
                st.warning(f"**Result:** Negative Response ({prediction})")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                if hasattr(model, "classes_"):
                    st.write("**Prediction Probabilities:**")
                    prob_df = pd.DataFrame(
                        [probabilities], columns=[str(c) for c in model.classes_]
                    )
                    st.dataframe(prob_df.style.format("{:.2%}"))

        except Exception as e:
            st.error(f"Error making prediction: {e}")


if __name__ == "__main__":
    main()
