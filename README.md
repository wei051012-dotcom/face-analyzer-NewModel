# Face Analyzer - Tuning Panel Version

This Streamlit app analyzes a front-facing portrait and returns face/eye/eyebrow/nose/lip categories.

## Recommended Streamlit Cloud settings

- Python version: 3.12
- Main file path: `app.py`
- Keep `face_landmarker.task` in the same repo folder as `app.py`

## How tuning works

The app still outputs one final class per feature, but internally every class receives a soft confidence score.

In the left sidebar:

- `吸附力 / Bias`: higher means that category becomes easier to select.
- `門檻 / Center`: the geometric cutoff point. Higher or lower changes how strict the category is.
- Use `只預覽，不加入紀錄` to test many parameter changes without polluting the history.
- Use `查看信心分數與診斷值` to inspect why a label was chosen.
