# Devpost Text (Drop-in)

**Problem & Impact.** Online reviews often contain advertisements, irrelevant content, and hearsay rants. **ReviewGuard** combines deterministic rules, a multi‑label classifier, and semantic relevancy scoring to improve trust for users, fairness for businesses, and reduce moderation costs.

**Approach.** (1) Rules for obvious cases; (2) DistilBERT/TF‑IDF model for policy flags; (3) Sentence‑Transformers for review↔place relevancy; (4) Policy engine with calibrated thresholds and explanations.

**Tech.** Python, PyTorch, Hugging Face Transformers, scikit‑learn, pandas, Sentence‑Transformers, Streamlit.

**APIs/Models.** `sentence-transformers/all-MiniLM-L6-v2` for embeddings. Transformer head for multi‑label classification (3 labels).

**Data.** Google Local Reviews (public), small manual set for validation; weak labels from rules; evaluated on a held‑out venue split.

**Results.** High-precision ad detection; solid macro‑F1 after threshold tuning. Evidence spans and JSON audit trail.

**Limitations.** Multilingual nuance, adversarial promos, sarcasm. **Next:** multilingual models, active learning, per‑category thresholds.
