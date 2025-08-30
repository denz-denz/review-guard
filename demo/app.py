import streamlit as st
import json, os, joblib
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import PlaceProfile
from src.features import EmbeddingService
from src.rules import rule_advertisement, rule_rant_without_visit, rule_irrelevant
from src.policy_engine import Thresholds, decide

st.set_page_config(page_title="ReviewGuard Demo", layout="centered")

st.title("🛡️ ReviewGuard — Review Moderation Demo")

with st.sidebar:
    st.header("Place Profile")
    place_name = st.text_input("Name", "Sunrise Cafe")
    category = st.text_input("Category", "Cafe")
    city = st.text_input("City", "Singapore")
    model_dir = st.text_input("Model dir", "models/baseline")
    thresholds_path = st.text_input("Thresholds", "configs/thresholds.yaml")

text = st.text_area("Paste a review", "Use code SAVE20 at www.bestcoffee-deals.com for discounts!!", height=160)
run = st.button("Score Review")

def load_baseline(path: str):
    return joblib.load(path)

if run:
    with st.spinner("Scoring..."):
        place = PlaceProfile(place_name, category, city)
        embed = EmbeddingService()
        relevancy = embed.relevancy(text, place)

        # load model (baseline)
        model_path = os.path.join(model_dir, "baseline.joblib")
        if not os.path.exists(model_path):
            st.error("Baseline model not found. Train it with: python src/train_baseline.py --data data/sample_reviews.csv --out_dir models/baseline")
            st.stop()
        pipe = load_baseline(model_path)

        # predict probs
        import numpy as np
        probs_raw = pipe.predict_proba([text])
        labels = ["advertisement","irrelevant","rant_without_visit"]
        probs = {}
        for i,lbl in enumerate(labels):
            # handle estimators without predict_proba
            try:
                p = float(probs_raw[i][0][1])
            except Exception:
                # fall back: decision_function sigmoid
                from scipy.special import expit
                # Transform text through TF-IDF first, then get decision function
                X_transformed = pipe.named_steps['tfidf'].transform([text])
                p = float(expit(pipe.named_steps['clf'].estimators_[i].decision_function(X_transformed)))
            probs[lbl] = p

        ad_rule, ad_spans = rule_advertisement(text)
        rant_rule, rant_spans = rule_rant_without_visit(text)
        irrel_rule, irrel_spans = rule_irrelevant(text, place.as_text())

        if ad_rule: probs["advertisement"] = max(probs["advertisement"], 0.95)
        if rant_rule: probs["rant_without_visit"] = max(probs["rant_without_visit"], 0.80)
        if irrel_rule: probs["irrelevant"] = max(probs["irrelevant"], 0.80)

        th = Thresholds.from_file(thresholds_path)
        decision = decide(th, probs, relevancy)

    st.subheader("Results")
    st.metric("Relevancy", f"{relevancy:.2f}")
    c1,c2,c3 = st.columns(3)
    c1.metric("Advertisement", f"{probs['advertisement']:.2f}")
    c2.metric("Irrelevant", f"{probs['irrelevant']:.2f}")
    c3.metric("Rant (No Visit)", f"{probs['rant_without_visit']:.2f}")
    st.info(f"Decision: **{decision['action'].upper()}**  \nReasons: {', '.join(decision['reasons']) or '—'}")

    # Evidence
    st.subheader("Evidence Spans")
    st.write({"advertisement": ad_spans, "irrelevant": irrel_spans, "rant_without_visit": rant_spans})

    # Simple highlighting
    def highlight(text, spans):
        out = text
        for s in set(spans):
            out = out.replace(s, f"**__{s}__**")
        return out
    combined_spans = (ad_spans or []) + (irrel_rule and irrel_spans or []) + (rant_spans or [])
    st.markdown(highlight(text, combined_spans))

st.caption("Tip: Train baseline first, then paste a review. Thresholds configurable in configs/thresholds.yaml.")
