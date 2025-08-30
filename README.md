# ReviewGuard: ML Moderation for Location Reviews

Hybrid rules + ML/NLP system to score **relevancy**, flag **policy violations** (advertisement, irrelevant, rant without visit),
and output **actionable explanations** for Google/Local-style reviews.

## Features
- Fast **regex/rules** for obvious promos and non-visit rants
- **Multi-label classifier** (baseline TF‑IDF + Logistic Regression; optional DistilBERT fine‑tune)
- **Semantic relevancy** via Sentence-Transformers between review and *place profile*
- **Policy engine** with calibrated thresholds → Pass / Hold / Remove
- **Streamlit demo** UI with evidence spans

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start (Demo on sample data)
```bash
# 1) Train fast baseline on sample data
python src/train_baseline.py --data data/sample_reviews.csv --out_dir models/baseline

# 2) Run Streamlit demo
streamlit run demo/app.py
```

## Train Transformer (optional)
```bash
python src/train_transformer.py --data data/sample_reviews.csv --out_dir models/transformer --epochs 1
```

## Inference (CLI)
```bash
python src/inference.py --text "Loved the pasta! Use code SAVE20 at www.pizzapromo.com" \
    --place_name "Mario's Trattoria" --category "Italian Restaurant" --city "Singapore"
```

## Repo Layout
```
reviewguard/
├── configs/thresholds.yaml
├── data/sample_reviews.csv
├── demo/app.py
├── docs/MODEL_CARD.md
├── notebooks/EDA.md
├── src/
│   ├── preprocess.py  ├── features.py  ├── rules.py
│   ├── policy_engine.py  ├── inference.py
│   ├── train_baseline.py ├── train_transformer.py ├── utils.py
├── tests/test_rules.py
├── requirements.txt
└── README.md
```

## Notes
- Sample data is tiny and for smoke tests only.
- DistilBERT training is configured for CPU by default; enable GPU with CUDA if available.
- Thresholds live in `configs/thresholds.yaml`.
- This code is MIT-licensed for hackathon use.
