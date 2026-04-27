# Face Analyzer - Probability Model Version

This Streamlit app analyzes a front-facing portrait and returns face/eye/eyebrow/nose/lip categories.

Recommended Streamlit Cloud settings:

- Python version: 3.12
- Main file path: `app.py`

The analyzer now uses soft probability-like scoring for every category, then outputs the highest-scoring class. The UI still shows a single final category, with an optional expander for confidence scores and diagnostic ratios.
