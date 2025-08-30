import re, pandas as pd
from langdetect import detect, LangDetectException

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # keep case for proper nouns; lightly normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s

def detect_language_safe(s: str, default="en"):
    try:
        return detect(s) if s and s.strip() else default
    except LangDetectException:
        return default

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["review_text"].apply(clean_text)
    df["lang"] = df["text"].apply(detect_language_safe)
    return df
