"""Kepler DR25 tabular model — the quick, strong win.

Pulls the Kepler DR25 KOI catalog (~9.5k vetted objects with measured transit +
stellar parameters) from the NASA Exoplanet Archive and trains a classifier.
Labels: CONFIRMED/CANDIDATE -> planet, FALSE POSITIVE -> not.
We deliberately EXCLUDE the koi_fpflag_* and koi_score columns (those already
encode the answer — using them would be cheating).

Usage:  python backend/train/kepler_koi.py
"""
import os
from collections import Counter
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
MODEL_OUT = os.path.join(BACKEND, "app", "models", "kepler_koi.joblib")

FEATURES = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_model_snr",
            "koi_impact", "koi_num_transits", "koi_steff", "koi_slogg", "koi_srad",
            "koi_teq", "koi_insol"]


def main():
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    print("Fetching Kepler DR25 KOI catalog ...")
    cols = "koi_disposition," + ",".join(FEATURES)
    df = NasaExoplanetArchive.query_criteria(table="q1_q17_dr25_koi", select=cols).to_pandas()
    df = df[df["koi_disposition"].isin(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])].copy()
    y = np.where(df["koi_disposition"] == "FALSE POSITIVE", "not_planet", "planet")
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    print(f"Samples: {len(df)}   classes: {dict(Counter(y))}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    pred = cross_val_predict(rf, X, y, cv=cv)
    print(f"\n== Kepler DR25 RandomForest: 5-fold CV accuracy = {accuracy_score(y, pred):.3f} ==")
    print(classification_report(y, pred, zero_division=0))
    print("confusion matrix (rows=true):", ["not_planet", "planet"])
    print(confusion_matrix(y, pred, labels=["not_planet", "planet"]))

    rf.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump({"model": rf, "features": FEATURES, "classes": list(rf.classes_), "kind": "rf"}, MODEL_OUT)
    print(f"\nSaved -> {MODEL_OUT}")
    top = sorted(zip(FEATURES, rf.feature_importances_), key=lambda t: -t[1])[:6]
    print("Top features:", [(f, round(float(i), 3)) for f, i in top])


if __name__ == "__main__":
    main()
