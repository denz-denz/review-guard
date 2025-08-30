import argparse, os, json, joblib
from typing import Dict, Any
from .utils import get_logger, PlaceProfile
from .features import EmbeddingService
from .rules import rule_advertisement, rule_rant_without_visit, rule_irrelevant
from .policy_engine import Thresholds, decide

LABELS = ["advertisement","irrelevant","rant_without_visit"]

def load_baseline(path: str):
    return joblib.load(path)

def predict_baseline(pipe, text: str) -> Dict[str, float]:
    import numpy as np
    proba = pipe.predict_proba([text])
    # OneVsRest -> list of arrays
    probs = {}
    for i, lbl in enumerate(["advertisement","irrelevant","rant_without_visit"]):
        # handle estimators without predict_proba
        try:
            p = float(proba[i][0][1])
        except Exception:
            # fall back: decision_function sigmoid
            from scipy.special import expit
            p = float(expit(pipe.estimators_[i].decision_function([text])))
        probs[lbl] = p
    return probs

def main(args):
    log = get_logger()

    # Model
    if args.model_type == "baseline":
        model_path = os.path.join(args.model_dir, "baseline.joblib")
        pipe = load_baseline(model_path)
        clf_fn = lambda t: predict_baseline(pipe, t)
    else:
        # lightweight: use baseline if transformer missing
        model_path = os.path.join(args.model_dir, "baseline.joblib")
        pipe = load_baseline(model_path)
        clf_fn = lambda t: predict_baseline(pipe, t)

    # Embeddings for relevancy
    embed = EmbeddingService()

    place = PlaceProfile(args.place_name, args.category, args.city)
    rel = embed.relevancy(args.text, place)

    # Rules
    ad_rule, ad_spans = rule_advertisement(args.text)
    rant_rule, rant_spans = rule_rant_without_visit(args.text)
    irrel_rule, irrel_spans = rule_irrelevant(args.text, place.as_text())

    probs = clf_fn(args.text)
    # Boost probabilities if hard rules fire
    if ad_rule:
        probs["advertisement"] = max(probs["advertisement"], 0.95)
    if rant_rule:
        probs["rant_without_visit"] = max(probs["rant_without_visit"], 0.80)
    if irrel_rule:
        probs["irrelevant"] = max(probs["irrelevant"], 0.80)

    th = Thresholds.from_file(args.thresholds)
    decision = decide(th, probs, rel)

    out = {
        "relevancy_score": round(rel, 4),
        "probs": {k: round(float(v), 4) for k, v in probs.items()},
        "flags": {k: (probs[k] >= 0.5) for k in probs},
        "decision": decision,
        "evidence": {
            "advertisement": ad_spans,
            "rant_without_visit": rant_spans,
            "irrelevant": irrel_spans
        }
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--place_name", default="Unknown Place")
    p.add_argument("--category", default="Unknown Category")
    p.add_argument("--city", default="Unknown City")
    p.add_argument("--model_dir", default="models/baseline")
    p.add_argument("--model_type", choices=["baseline","transformer"], default="baseline")
    p.add_argument("--thresholds", default="configs/thresholds.yaml")
    args = p.parse_args()
    main(args)
