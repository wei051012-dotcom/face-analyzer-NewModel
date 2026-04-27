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


## 這版新增

鼻型與唇形已移到左側主控制面板，不需要打開進階參數也能調整吸附力。

### 調整方式
- 某一類太常出現：把該類「吸附力」往下拉。
- 某一類幾乎不出現：把該類「吸附力」往上拉。
- 打開「顯示進階參數」後，可以調整鼻寬、鼻長、唇高、上下唇比例等細部門檻。
