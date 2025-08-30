import re
from typing import Dict, List, Tuple

URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}")
PROMO_RE = re.compile(r"\b(promo|discount|coupon|code|deal|sale|% ?off|dm|inbox me|contact me|whatsapp|telegram)\b", re.IGNORECASE)
HEARSAY_RE = re.compile(r"\b(never been|haven'?t (been|visited)|didn'?t go|heard (that|it)|my friend (said|told me))\b", re.IGNORECASE)

def rule_advertisement(text: str) -> Tuple[bool, List[str]]:
    spans = []
    for pat in (URL_RE, PHONE_RE, PROMO_RE):
        for m in pat.finditer(text or ""):
            spans.append(m.group(0))
    return (len(spans) > 0 and bool(PROMO_RE.search(text or "") or URL_RE.search(text or ""))), spans

def rule_rant_without_visit(text: str) -> Tuple[bool, List[str]]:
    spans = [m.group(0) for m in HEARSAY_RE.finditer(text or "")]
    # optionally check for negativity keywords (lightweight)
    negative = re.search(r"\b(awful|terrible|bad|dirty|worst|horrible|avoid)\b", (text or "").lower())
    return (len(spans) > 0 or bool(negative and HEARSAY_RE.search(text or ""))), spans

def rule_irrelevant(text: str, place_tokens: str) -> Tuple[bool, List[str]]:
    # heuristic: if text mentions unrelated products strongly; final decision uses semantic relevancy elsewhere
    off_topic = re.search(r"\b(iphone|android phone|bitcoin|crypto|politics|election)\b", (text or "").lower())
    return (bool(off_topic)), [off_topic.group(0)] if off_topic else []
