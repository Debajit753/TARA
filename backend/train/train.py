"""Phase 3 (LEGACY) — train the classifier on the labelled feature table.

SUPERSEDED: the serving model is now the 4-class RandomForest built by
notebooks/tara_colab_rf4_build.ipynb (2,055 ExoMiner-labeled stars, 73.6% grouped).
This script trained the original 3-class model on the small 278-star TOI dataset.
It now saves to model_legacy.joblib so a rerun can NEVER overwrite the serving
model (backend/app/models/tabular/model.joblib) by accident.

Usage:  python backend/train/train.py
"""
import os
from collections import Counter
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False  # xgboost needs libomp (brew install libomp) — optional

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
DATA = os.path.join(BACKEND, "data", "processed", "dataset.csv")
MODEL_OUT = os.path.join(BACKEND, "app", "models", "tabular", "model_legacy.joblib")

BASE_FEATURES = ["period", "depth", "duration", "duration_frac", "snr",
                 "odd_even_diff", "secondary_ratio", "v_shape", "n_transits"]


def load():
    df = pd.read_csv(DATA)
    feats = BASE_FEATURES + (["centroid"] if "centroid" in df.columns else [])
    X = df[feats].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X, df["label"].astype(str).values, df, feats


def main():
    X, y, df, FEATURES = load()
    classes = sorted(set(y))
    counts = Counter(y)
    print(f"Samples: {len(df)}   classes: {dict(counts)}")

    k = min(5, min(counts.values()))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    candidates = [("RandomForest", rf, y)]
    if HAS_XGB:
        xgbc = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.08,
                                 subsample=0.9, colsample_bytree=0.9,
                                 eval_metric="mlogloss", random_state=42, n_jobs=-1)
        candidates.append(("XGBoost", xgbc, y_enc))
    else:
        print("(XGBoost skipped — libomp not installed; using RandomForest only)")

    results = {}
    for name, clf, target in candidates:
        pred = cross_val_predict(clf, X, target, cv=cv)
        if name == "XGBoost":
            pred = le.inverse_transform(pred)
        acc = accuracy_score(y, pred)
        results[name] = (acc, clf, pred)
        print(f"\n== {name}: {k}-fold CV accuracy = {acc:.3f} ==")
        print(classification_report(y, pred, zero_division=0))

    best = max(results, key=lambda n: results[n][0])
    acc, clf, pred = results[best]
    print(f"\n>>> Best model: {best}  (accuracy {acc:.3f}) <<<")
    print("confusion matrix (rows=true, cols=pred):", classes)
    print(confusion_matrix(y, pred, labels=classes))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    if best == "XGBoost":
        clf.fit(X, y_enc)
        joblib.dump({"model": clf, "features": FEATURES, "classes": list(le.classes_),
                     "kind": "xgboost", "label_encoder": le}, MODEL_OUT)
        imp = clf.feature_importances_
    else:
        clf.fit(X, y)
        joblib.dump({"model": clf, "features": FEATURES, "classes": list(clf.classes_),
                     "kind": "rf"}, MODEL_OUT)
        imp = clf.feature_importances_
    print(f"Saved {best} -> {MODEL_OUT}")
    top = sorted(zip(FEATURES, imp), key=lambda t: -t[1])[:5]
    print("Top features:", [(f, round(float(i), 3)) for f, i in top])


if __name__ == "__main__":
    main()
