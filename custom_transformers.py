"""
Custom sklearn transformers used by the model pipeline.

This MUST live in its own importable module (not inline in the notebook, not
inline in the Streamlit app). joblib/pickle only saves a *reference* to where
a custom class is defined, not its code.  so whatever process loads the
pickle later needs to be able to `import` this exact class from this exact
module path. Defining it inline in the notebook works fine for the notebook
itself, but breaks the moment anything else (like a Streamlit app) tries to
load the saved pipeline.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Frequency-encodes categorical columns; frequencies computed on the
    training fold only, so no test-fold information leaks in."""
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        self.freq_maps_ = {c: X[c].value_counts(normalize=True) for c in self.cols}
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c] = X[c].map(self.freq_maps_[c]).fillna(0)  # unseen category at test time -> 0
        return X[self.cols]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols)
