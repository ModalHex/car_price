"""
Section 8: Streamlit deployment.
Run with:  py -m streamlit run streamlit_app.py
(Streamlit apps are run as a script from the terminal -- this file is not
meant to be executed inside a Jupyter cell.)
"""
from pathlib import Path

import joblib
import streamlit as st

from inference import predict_price  # the plain-Python function, tested separately in inference.py


def _resolve_data_file(filename: str) -> Path:
    """Resolve a model/data file relative to this script, with a safe fallback."""
    project_dir = Path(__file__).resolve().parent
    candidate = project_dir / filename
    if candidate.exists():
        return candidate

    cwd_candidate = Path.cwd() / filename
    if cwd_candidate.exists():
        return cwd_candidate

    raise FileNotFoundError(
        f"Could not find '{filename}' in the project directory ({project_dir}) "
        f"or the current working directory ({Path.cwd()})."
    )


ui_meta = joblib.load(_resolve_data_file("ui_meta.pkl"))

st.title("Car Price Predictor")
st.caption("Predicts a fair selling price from the trained model (Section 6-7).")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", ui_meta["brand_options"])
    # cascading dropdown: only show models that actually belong to the chosen brand
    model_name = st.selectbox("Model", ui_meta["brand_to_models"][brand])
    seller_type = st.selectbox("Seller type", ui_meta["seller_type_options"])
    fuel_type = st.selectbox("Fuel type", ui_meta["fuel_type_options"])
    transmission_type = st.selectbox("Transmission", ui_meta["transmission_type_options"])

with col2:
    age_lo, age_hi = ui_meta["ranges"]["vehicle_age"]
    vehicle_age = st.slider("Vehicle age (years)", int(age_lo), int(age_hi), int((age_lo + age_hi) // 2))

    km_lo, km_hi = ui_meta["ranges"]["km_driven"]
    km_driven = st.number_input("Kilometers driven", min_value=float(km_lo), max_value=float(km_hi) * 1.2,
                                 value=float((km_lo + km_hi) / 2), step=1000.0)

    mil_lo, mil_hi = ui_meta["ranges"]["mileage"]
    mileage = st.slider("Mileage (km/l or km/kg)", float(mil_lo), float(mil_hi), float((mil_lo + mil_hi) / 2))

    eng_lo, eng_hi = ui_meta["ranges"]["engine"]
    engine = st.number_input("Engine (cc)", min_value=float(eng_lo), max_value=float(eng_hi),
                              value=float((eng_lo + eng_hi) / 2), step=50.0)

    pow_lo, pow_hi = ui_meta["ranges"]["max_power"]
    max_power = st.number_input("Max power (bhp)", min_value=float(pow_lo), max_value=float(pow_hi),
                                 value=float((pow_lo + pow_hi) / 2), step=1.0)

    seats_lo, seats_hi = ui_meta["ranges"]["seats"]
    seats = st.slider("Seats", int(seats_lo), int(seats_hi), 5)

if st.button("Predict price", type="primary"):
    raw_input = {
        "vehicle_age": vehicle_age, "km_driven": km_driven, "mileage": mileage,
        "engine": engine, "max_power": max_power, "seats": seats,
        "brand": brand, "model": model_name,
        "seller_type": seller_type, "fuel_type": fuel_type, "transmission_type": transmission_type,
    }
    price = predict_price(raw_input)
    st.success(f"Predicted selling price: \u20b9{price:,.0f}")
