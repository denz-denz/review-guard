import argparse, os, joblib, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_dataframe

LABELS = ["label_ad","label_irrel","label_rant"]

def main(args):
    df = pd.read_csv(args.data)
    df = preprocess_dataframe(df)
    y = df[LABELS].astype(int).values
    X = df["text"].values

    # For small datasets, use a smaller test size or no stratification
    if len(X) < 20:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=df[LABELS])

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    print(classification_report(yte, pred, target_names=LABELS, zero_division=0))

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(args.out_dir, "baseline.joblib"))
    print(f"Saved baseline model to {os.path.join(args.out_dir, 'baseline.joblib')}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="models/baseline")
    args = p.parse_args()
    main(args)
