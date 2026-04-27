import streamlit as st
import cv2
import numpy as np
from PIL import Image
from analyzer import FaceAnalyzer

st.set_page_config(page_title="Face Analyzer", page_icon="👤", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 78vw !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

DEFAULT_TUNING = {
    "face_square_bias": 0.75,
    "face_round_bias": 1.10,
    "face_long_bias": 1.15,
    "face_heart_bias": 1.25,
    "face_oval_bias": 0.95,
    "face_square_hw_center": 1.32,
    "face_square_jc_center": 1.02,

    "eye_up_bias": 0.85,
    "eye_down_bias": 1.10,
    "eye_round_bias": 1.05,
    "eye_slender_bias": 1.00,
    "eye_almond_bias": 0.90,
    "eye_up_angle_center": 6.5,
    "eye_down_angle_center": -4.5,
    "eye_round_ratio_center": 2.10,

    "brow_arch_bias": 0.75,
    "brow_up_bias": 1.05,
    "brow_down_bias": 1.05,
    "brow_straight_bias": 1.05,
    "brow_arch_center": 0.20,
    "brow_up_center": 0.075,
    "brow_down_center": -0.055,

    "nose_wide_bias": 1.10,
    "nose_narrow_bias": 1.15,
    "nose_high_bias": 1.15,
    "nose_low_bias": 1.05,
    "nose_medium_bias": 0.85,
    "nose_narrow_center": 0.235,
    "nose_high_center": 0.335,

    "lip_upper_bias": 1.05,
    "lip_lower_bias": 1.05,
    "lip_thick_bias": 1.10,
    "lip_thin_bias": 1.05,
    "lip_medium_bias": 0.90,
    "lip_thick_center": 0.108,
    "lip_thin_center": 0.073,
}

if "history" not in st.session_state:
    st.session_state.history = []

if "tuning" not in st.session_state:
    st.session_state.tuning = DEFAULT_TUNING.copy()

def reset_tuning():
    st.session_state.tuning = DEFAULT_TUNING.copy()

def slider(label, key, min_value, max_value, step, help_text=None):
    value = st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=float(st.session_state.tuning.get(key, DEFAULT_TUNING[key])),
        step=step,
        key=f"slider_{key}",
        help=help_text,
    )
    st.session_state.tuning[key] = value
    return value

with st.sidebar:
    st.header("🎛️ 調參控制面板")
    st.caption("不用改程式碼，直接拖拉滑桿。調完後重新按「開始分析」。")

    if st.button("重設所有參數"):
        reset_tuning()
        st.rerun()

    show_advanced = st.toggle("顯示進階參數", value=False)

    st.subheader("臉型")
    st.caption("Bias 越高越容易被判定；越低越不容易。")
    slider("方臉吸附力", "face_square_bias", 0.20, 2.00, 0.05)
    slider("圓臉吸附力", "face_round_bias", 0.20, 2.00, 0.05)
    slider("長臉吸附力", "face_long_bias", 0.20, 2.00, 0.05)
    slider("心型臉吸附力", "face_heart_bias", 0.20, 2.00, 0.05)
    slider("橢圓臉吸附力", "face_oval_bias", 0.20, 2.00, 0.05)
    if show_advanced:
        slider("方臉：臉高/臉寬門檻", "face_square_hw_center", 1.15, 1.55, 0.01)
        slider("方臉：下顎/顴骨門檻", "face_square_jc_center", 0.80, 1.20, 0.01)

    st.subheader("眼型")
    slider("上揚眼吸附力", "eye_up_bias", 0.20, 2.00, 0.05)
    slider("下垂眼吸附力", "eye_down_bias", 0.20, 2.00, 0.05)
    slider("圓眼吸附力", "eye_round_bias", 0.20, 2.00, 0.05)
    slider("細長眼吸附力", "eye_slender_bias", 0.20, 2.00, 0.05)
    slider("杏仁眼吸附力", "eye_almond_bias", 0.20, 2.00, 0.05)
    if show_advanced:
        slider("上揚眼角度門檻", "eye_up_angle_center", 2.0, 12.0, 0.5)
        slider("下垂眼角度門檻", "eye_down_angle_center", -12.0, -2.0, 0.5)
        slider("圓眼：寬高比門檻", "eye_round_ratio_center", 1.60, 2.60, 0.05)

    st.subheader("眉型")
    slider("拱眉吸附力", "brow_arch_bias", 0.20, 2.00, 0.05)
    slider("上揚眉吸附力", "brow_up_bias", 0.20, 2.00, 0.05)
    slider("下垂眉吸附力", "brow_down_bias", 0.20, 2.00, 0.05)
    slider("平眉吸附力", "brow_straight_bias", 0.20, 2.00, 0.05)
    if show_advanced:
        slider("拱眉：眉峰高度門檻", "brow_arch_center", 0.08, 0.32, 0.005)
        slider("上揚眉：眉尾上揚門檻", "brow_up_center", 0.02, 0.16, 0.005)
        slider("下垂眉：眉尾下垂門檻", "brow_down_center", -0.16, -0.02, 0.005)

    if show_advanced:
        st.subheader("鼻型")
        slider("寬鼻吸附力", "nose_wide_bias", 0.20, 2.00, 0.05)
        slider("窄鼻吸附力", "nose_narrow_bias", 0.20, 2.00, 0.05)
        slider("高鼻樑吸附力", "nose_high_bias", 0.20, 2.00, 0.05)
        slider("低鼻樑吸附力", "nose_low_bias", 0.20, 2.00, 0.05)
        slider("中等鼻吸附力", "nose_medium_bias", 0.20, 2.00, 0.05)
        slider("窄鼻：鼻寬/臉寬門檻", "nose_narrow_center", 0.18, 0.30, 0.005)
        slider("高鼻樑：鼻長/臉高門檻", "nose_high_center", 0.25, 0.42, 0.005)

        st.subheader("嘴唇")
        slider("上唇較厚吸附力", "lip_upper_bias", 0.20, 2.00, 0.05)
        slider("下唇較厚吸附力", "lip_lower_bias", 0.20, 2.00, 0.05)
        slider("厚唇吸附力", "lip_thick_bias", 0.20, 2.00, 0.05)
        slider("薄唇吸附力", "lip_thin_bias", 0.20, 2.00, 0.05)
        slider("中等唇吸附力", "lip_medium_bias", 0.20, 2.00, 0.05)
        slider("厚唇：唇高/臉高門檻", "lip_thick_center", 0.07, 0.15, 0.002)
        slider("薄唇：唇高/臉高門檻", "lip_thin_center", 0.04, 0.11, 0.002)

    with st.expander("目前參數 JSON"):
        st.json(st.session_state.tuning)

st.title("👤 Face Analyzer")
st.write("上傳一張正面人像照片，系統會分析臉型、眼型、眉型、鼻型與嘴唇。左側可以直接調整分類參數。")

uploaded_file = st.file_uploader("選擇一張圖片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    if len(image_np.shape) == 2:
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    else:
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="目前上傳的圖片", width=350)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        analyze_clicked = st.button("開始分析並加入紀錄", type="primary")
    with col_b:
        preview_clicked = st.button("只預覽，不加入紀錄")

    if analyze_clicked or preview_clicked:
        with st.spinner("分析中..."):
            analyzer = FaceAnalyzer()
            result = analyzer.analyze(image_cv2, tuning=st.session_state.tuning)

        if result[0] is None:
            st.error("❌ 找不到臉部！請確認照片中包含清晰的正面人臉，光線充足且沒有被遮擋。")
        else:
            analysis, annotated_image = result
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            if analyze_clicked:
                st.success("✅ 分析完成！已加入下方紀錄中。")
                st.session_state.history.append({
                    "file_name": uploaded_file.name,
                    "annotated_image": annotated_image_rgb,
                    "analysis": analysis,
                    "tuning": st.session_state.tuning.copy(),
                })
            else:
                st.info("✅ 預覽完成。這筆沒有加入歷史紀錄。")

            st.subheader("即時結果")
            c1, c2 = st.columns([1, 1.2])
            with c1:
                st.image(annotated_image_rgb, caption="標註後的圖片", use_container_width=True)
            with c2:
                st.table({
                    "特徵": ["臉型", "眼型", "眉型", "鼻型", "嘴唇"],
                    "分類結果": [
                        analysis.get("face_shape", "未知"),
                        analysis.get("eye_shape", "未知"),
                        analysis.get("eyebrow_shape", "未知"),
                        analysis.get("nose_shape", "未知"),
                        analysis.get("lips", "未知"),
                    ],
                })
                with st.expander("查看信心分數與診斷值"):
                    for label, key in {
                        "臉型": "face_shape_probs",
                        "眼型": "eye_shape_probs",
                        "眉型": "eyebrow_shape_probs",
                        "鼻型": "nose_shape_probs",
                        "嘴唇": "lips_probs",
                    }.items():
                        probs = analysis.get(key, {})
                        if probs:
                            st.markdown(f"**{label}**")
                            st.table({
                                "類別": list(probs.keys()),
                                "信心分數": [f"{v * 100:.1f}%" for v in probs.values()],
                            })
                    st.markdown("**量測診斷值**")
                    st.json(analysis.get("debug_metrics", {}))

if st.session_state.history:
    st.markdown("---")
    st.header("📚 分析紀錄")

    if st.button("清除所有紀錄"):
        st.session_state.history = []
        st.rerun()

    if len(st.session_state.history) > 0:
        st.subheader("📈 長度統計趨勢")
        st.write("橫軸為樣本編號（依上傳順序），縱軸為長度（單位：眼距）。")

        import pandas as pd
        import altair as alt

        chart_data = {
            "眼睛寬度": [r["analysis"]["eye_width_eye_dist_ratio"] for r in st.session_state.history],
            "鼻翼寬度": [r["analysis"]["nose_eye_ratio"] for r in st.session_state.history],
            "嘴唇寬度": [r["analysis"]["mouth_eye_ratio"] for r in st.session_state.history],
        }
        df = pd.DataFrame(chart_data)
        df.index = range(1, len(df) + 1)

        chart_col, stat_col = st.columns([2, 1])
        with chart_col:
            df_long = df.reset_index().melt("index", var_name="測量項目", value_name="比例 (倍眼距)")
            chart = alt.Chart(df_long).mark_line(point=True).encode(
                x=alt.X("index:O", title="樣本編號", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("比例 (倍眼距):Q", scale=alt.Scale(zero=False)),
                color="測量項目:N",
                tooltip=["index", "測量項目", "比例 (倍眼距)"],
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        with stat_col:
            st.markdown("**📊 統計數據**")
            for key in chart_data.keys():
                mean_val = df[key].mean()
                std_val = df[key].std() if len(df) > 1 else 0.0
                st.write(f"**{key}**")
                st.write(f"- 平均: {mean_val:.2f}")
                st.write(f"- 標準差: {std_val:.2f}")

        if st.toggle("顯示五官類型分布圖表 (長條圖)"):
            st.subheader("📊 五官類型分布")
            features = ["face_shape", "eye_shape", "eyebrow_shape", "nose_shape", "lips"]
            titles = ["臉型分布", "眼型分布", "眉型分布", "鼻型分布", "嘴唇分布"]
            tabs = st.tabs(titles)
            possible_categories = {
                "face_shape": ["長臉 (Long)", "心形臉 (Heart)", "方臉 (Square)", "圓臉 (Round)", "橢圓臉 (Oval)"],
                "eye_shape": ["上揚眼 (Upturned)", "下垂眼 (Downturned)", "圓眼 (Round)", "細長眼 (Slender)", "杏仁眼 (Almond)"],
                "eyebrow_shape": ["拱眉 (Arched)", "上揚眉 (Upturned)", "下垂眉 (Downturned)", "平眉 (Straight)"],
                "nose_shape": ["寬鼻 (Wide)", "窄鼻 (Narrow)", "高鼻樑 (High bridge)", "低鼻樑 (Low bridge)", "中等鼻 (Medium)"],
                "lips": ["上唇較厚 (Thicker upper lip)", "下唇較厚 (Thicker lower lip)", "厚唇 (Thick)", "薄唇 (Thin)", "中等唇 (Medium)"],
            }

            for tab, feature, title in zip(tabs, features, titles):
                with tab:
                    counts = {cat: 0 for cat in possible_categories[feature]}
                    for r in st.session_state.history:
                        val = r["analysis"].get(feature, "未知")
                        counts[val] = counts.get(val, 0) + 1
                    df_counts = pd.DataFrame(list(counts.items()), columns=["類型", "數量"])
                    bar_chart = alt.Chart(df_counts).mark_bar().encode(
                        x=alt.X("類型:N", title="類型", axis=alt.Axis(labelAngle=0, labelLimit=0)),
                        y=alt.Y("數量:Q", title="數量", axis=alt.Axis(tickMinStep=1)),
                        color=alt.Color("類型:N", legend=None),
                        tooltip=["類型", "數量"],
                    ).properties(height=300)
                    st.altair_chart(bar_chart, use_container_width=True)

    st.markdown("---")

    for i, record in enumerate(reversed(st.session_state.history)):
        real_idx = len(st.session_state.history) - 1 - i
        record_idx = real_idx + 1

        header_col, btn_col = st.columns([5, 1])
        with header_col:
            st.subheader(f"紀錄 #{record_idx}: {record['file_name']}")
        with btn_col:
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ 刪除", key=f"del_{real_idx}"):
                st.session_state.history.pop(real_idx)
                st.rerun()

        col_img, col_data = st.columns([1, 1.2])

        with col_img:
            st.image(record["annotated_image"], caption="標註後的圖片", use_container_width=True)

        with col_data:
            analysis = record["analysis"]
            st.markdown("**🔹 五官類型**")
            st.table({
                "特徵": ["臉型", "眼型", "眉型", "鼻型", "嘴唇"],
                "分類結果": [
                    analysis.get("face_shape", "未知"),
                    analysis.get("eye_shape", "未知"),
                    analysis.get("eyebrow_shape", "未知"),
                    analysis.get("nose_shape", "未知"),
                    analysis.get("lips", "未知"),
                ],
            })

            st.markdown("**🔹 比例量測 (以眼距為單位)**")
            st.table({
                "量測項目": ["眼睛寬度", "鼻翼寬度", "嘴唇寬度"],
                "比例 (倍眼距)": [
                    analysis.get("eye_width_eye_dist_ratio", "未知"),
                    analysis.get("nose_eye_ratio", "未知"),
                    analysis.get("mouth_eye_ratio", "未知"),
                ],
            })

            with st.expander("查看這筆的信心分數 / 參數"):
                for label, key in {
                    "臉型": "face_shape_probs",
                    "眼型": "eye_shape_probs",
                    "眉型": "eyebrow_shape_probs",
                    "鼻型": "nose_shape_probs",
                    "嘴唇": "lips_probs",
                }.items():
                    probs = analysis.get(key, {})
                    if probs:
                        st.markdown(f"**{label}**")
                        st.table({
                            "類別": list(probs.keys()),
                            "信心分數": [f"{v * 100:.1f}%" for v in probs.values()],
                        })
                st.markdown("**量測診斷值**")
                st.json(analysis.get("debug_metrics", {}))
                st.markdown("**當時使用的參數**")
                st.json(record.get("tuning", {}))

        st.markdown("---")
