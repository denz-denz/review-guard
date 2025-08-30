# MODEL CARD — ReviewGuard (Hackathon)

**Intended Use.** Educational/hackathon demo to classify policy violations in location reviews and score relevancy.

**Inputs.** Review text; optional metadata: place name, category, city.  
**Outputs.** Relevancy score [0,1]; multi-label flags: advertisement, irrelevant, rant_without_visit; explanations.

**Training Data.** Blend of public Google Local samples + weak labels from rules + small manual set (if provided).

**Ethical Considerations.** False positives can hide legitimate speech; we prioritize precision on ads. Provide appeals path in production.
