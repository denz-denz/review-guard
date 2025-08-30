import os, re, json, logging
from dataclasses import dataclass
from typing import Dict, Any

def get_logger(name="reviewguard", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger

@dataclass
class PlaceProfile:
    name: str
    category: str
    city: str

    def as_text(self) -> str:
        parts = [self.name or "", self.category or "", self.city or ""]
        return " | ".join([p for p in parts if p])
