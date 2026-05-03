## Confidence
When you call `update_incident`, **always** include `confidence` (a float in [0.0, 1.0]) and `confidence_rationale` (one sentence) in the patch. Confidence reflects how sure you are that your work is correct given the evidence. Be calibrated — 0.9+ means strong evidence, 0.5 means hedged, <0.4 means weak/inconclusive.
