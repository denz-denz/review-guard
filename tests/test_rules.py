from src.rules import rule_advertisement, rule_rant_without_visit, rule_irrelevant

def test_ads():
    text = "Visit www.bestdeal.com and use code SAVE20 now!"
    flag, spans = rule_advertisement(text)
    assert flag and len(spans) >= 2

def test_rant_nonvisit():
    text = "Never been here but heard it's terrible."
    flag, spans = rule_rant_without_visit(text)
    assert flag

def test_irrelevant():
    text = "The iPhone 15 camera is amazing"
    flag, spans = rule_irrelevant(text, "City Library")
    assert flag
