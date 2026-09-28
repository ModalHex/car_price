"""
Pure inference logic for the car price predictor.
Deliberately has NO Streamlit dependency, so it can be tested on its own
before any UI code is written (Section 8 planning, step 3).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.special import boxcox as boxcox_transform  # forward Box-Cox given a fixed lambda


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


PIPE = joblib.load(_resolve_data_file("best_model_pipeline.pkl"))
META = joblib.load(_resolve_data_file("deployment_meta.pkl"))

KM_LAMBDA = META["km_lambda"]
LUXURY_BRANDS = set(META["luxury_brands"])
RAW_COLS_NEEDED = META["raw_cols_needed"]


def predict_price(raw_input: dict) -> float:
    """
    raw_input: a dict of the RAW, human-entered values, e.g.
        {
            "vehicle_age": 5, "km_driven": 45000, "mileage": 18.5,
            "engine": 1197, "max_power": 82.0, "seats": 5,
            "brand": "Maruti", "model": "Swift",
            "seller_type": "Individual", "fuel_type": "Petrol",
            "transmission_type": "Manual",
        }
    Returns the predicted selling price as a float, in the same units
    (Rupees) as the original target.
    """
    row = {}

    # -- engineered numeric features, mirroring Section 4.2 EXACTLY --
    # log1p features: fixed formula, no fitted state needed
    row["vehicle_age_log"] = np.log1p(raw_input["vehicle_age"])
    row["engine_log"] = np.log1p(raw_input["engine"])
    row["max_power_log"] = np.log1p(raw_input["max_power"])

    # Box-Cox: MUST reuse the lambda fitted on the training data (km_lambda),
    # never re-fit boxcox() on a single new row -- a lambda fitted on n=1
    # is meaningless, and would silently produce a different transform than
    # the one the model was trained on.
    row["km_driven_boxcox"] = boxcox_transform(raw_input["km_driven"], KM_LAMBDA)

    # mileage is passed through untransformed (Section 4.2: already near-symmetric)
    row["mileage"] = raw_input["mileage"]

    # -- engineered categorical feature, mirroring Section 3/6.2's train-only fix --
    # NOT recomputed from any mean here -- membership was decided once, from
    # TRAINING data only, back in Section 6.2's leakage fix. A brand not seen
    # in training (i.e. not in this fixed list) is simply treated as non-luxury,
    # which is the same behaviour training-time code would have shown it.
    row["is_luxury_brand"] = int(raw_input["brand"] in LUXURY_BRANDS)

    # -- passthrough categorical/raw columns the pipeline's ColumnTransformer
    #    encodes itself (OneHotEncoder for nominal_cols, FrequencyEncoder for
    #    highcard_cols) -- pass the RAW values, not anything pre-encoded --
    row["seats"] = raw_input["seats"]
    row["brand"] = raw_input["brand"]
    row["model"] = raw_input["model"]
    row["seller_type"] = raw_input["seller_type"]
    row["fuel_type"] = raw_input["fuel_type"]
    row["transmission_type"] = raw_input["transmission_type"]

    # Assemble a single-row DataFrame with EXACTLY the columns the pipeline
    # expects (raw_cols_needed), in any order -- ColumnTransformer selects by
    # name, not position, so column order here doesn't matter, but presence does.
    X_new = pd.DataFrame([row])[RAW_COLS_NEEDED]

    pred = PIPE.predict(X_new)[0]

    # Only invert a log-target if the model was actually trained on the log
    # target -- current notebook trains on raw selling_price, so this is a
    # no-op today, but keeps the function correct if that ever changes.
    target_is_log = False  # mirrors notebook's `target == "selling_price_log"` check
    return float(np.expm1(pred)) if target_is_log else float(pred)


if __name__ == "__main__":
    # quick manual sanity checks -- run this file directly before wiring up any UI
    examples = [
        {"vehicle_age": 5, "km_driven": 45000, "mileage": 18.5, "engine": 1197,
         "max_power": 82.0, "seats": 5, "brand": "Maruti", "model": "Swift",
         "seller_type": "Individual", "fuel_type": "Petrol", "transmission_type": "Manual"},
        {"vehicle_age": 2, "km_driven": 8000, "mileage": 12.0, "engine": 2996,
         "max_power": 335.0, "seats": 5, "brand": "Bmw", "model": "X5",
         "seller_type": "Dealer", "fuel_type": "Diesel", "transmission_type": "Automatic"},
        {"vehicle_age": 12, "km_driven": 180000, "mileage": 22.0, "engine": 796,
         "max_power": 47.0, "seats": 4, "brand": "Tata", "model": "Nano",
         "seller_type": "Individual", "fuel_type": "Petrol", "transmission_type": "Manual"},
        # a brand NOT in the training set at all -- should not crash
        {"vehicle_age": 3, "km_driven": 20000, "mileage": 15.0, "engine": 1998,
         "max_power": 150.0, "seats": 5, "brand": "Totally Fictional Motors", "model": "Ghost",
         "seller_type": "Dealer", "fuel_type": "Petrol", "transmission_type": "Manual"},
    ]
    for ex in examples:
        price = predict_price(ex)
        print(f"{ex['brand']:28s} {ex['model']:8s} | predicted price: {price:,.0f}")
