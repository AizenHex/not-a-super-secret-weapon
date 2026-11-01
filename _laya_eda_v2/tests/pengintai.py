"""Model pengintai untuk uji E12: mencatat baris yang dilihat saat fit dan predict (bukan kode hasil)."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

CATATAN: dict = {"fit": [], "predict": []}


class Pengintai(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        CATATAN["fit"].append(np.asarray(X, dtype=float).copy())
        self.classes_ = np.unique(y)
        self.prior_ = np.bincount(np.asarray(y, dtype=int), minlength=len(self.classes_)) / len(y)
        return self

    def predict_proba(self, X):
        CATATAN["predict"].append(np.asarray(X, dtype=float).copy())
        return np.tile(self.prior_[self.classes_.astype(int)], (len(X), 1))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
