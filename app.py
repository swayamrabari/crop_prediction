from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path(__file__).with_name("Crop_recommendation.csv")
FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]
TARGET = "label"

@st.cache_resource
def train_model():
    data = pd.read_csv(DATA_FILE)
    data.columns = data.columns.str.strip()

    missing_columns = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0),
            ),
        ]
    )
    model.fit(data[FEATURES], data[TARGET])
    return model


st.set_page_config(
    page_title="Crop Recommendation",
    layout="centered",
)

st.title("Crop Recommendation")
st.write("Enter the soil and weather measurements to get a crop recommendation.")

demo_examples = {
    "Rice": [90, 42, 43, 20.88, 82.00, 6.50, 202.94],
    "Banana": [100, 82, 50, 27.38, 80.36, 5.98, 104.63],
    "Apple": [21, 134, 200, 22.63, 92.33, 5.93, 112.65],
}

for feature, default in zip(FEATURES, [50, 50, 40, 25.6, 71.5, 6.5, 103.5]):
    st.session_state.setdefault(feature, str(default))

with st.expander("Demo Examples", expanded=False):
    st.write("Select an example to fill the input fields automatically.")
    demo_buttons = st.columns(len(demo_examples))
    for column, (crop, values) in zip(demo_buttons, demo_examples.items()):
        with column:
            if st.button(crop, use_container_width=True):
                for feature, value in zip(FEATURES, values):
                    st.session_state[feature] = str(value)
                st.rerun()

with st.form("crop_form"):
    st.subheader("Soil Measurements")
    soil_left, soil_middle, soil_right = st.columns(3)
    with soil_left:
        nitrogen = st.text_input("Nitrogen (N)", key="N")
    with soil_middle:
        phosphorus = st.text_input("Phosphorus (P)", key="P")
    with soil_right:
        potassium = st.text_input("Potassium (K)", key="K")

    st.subheader("Weather and Soil Conditions")
    condition_left, condition_middle, condition_right = st.columns(3)
    with condition_left:
        temperature = st.text_input("Temperature (°C)", key="temperature")
    with condition_middle:
        humidity = st.text_input("Humidity (%)", key="humidity")
    with condition_right:
        ph = st.text_input("pH", key="ph")
    rainfall = st.text_input("Rainfall (mm)", key="rainfall")

    submitted = st.form_submit_button("Recommend Crop", type="primary", use_container_width=True)

if submitted:
    try:
        input_values = [
            float(value)
            for value in [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        ]
        model = train_model()
        input_data = pd.DataFrame(
            [input_values],
            columns=FEATURES,
        )
        probabilities = model.predict_proba(input_data)[0]
        classes = model.named_steps["classifier"].classes_
        top_crops = [classes[index] for index in probabilities.argsort()[::-1][:3]]
        best_crop = top_crops[0]

        st.success(f"Recommended crop: {best_crop.title()}")
        st.subheader("Top 3 Suitable Crops")
        for rank, crop in enumerate(top_crops, start=1):
            st.write(f"{rank}. {crop.title()}")
    except (TypeError, ValueError):
        st.error("Please enter valid numeric values for every field.")
