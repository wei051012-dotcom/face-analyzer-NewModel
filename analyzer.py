from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceAnalyzer:
    """Face feature analyzer using soft probability-like scoring.

    Notes:
    - This is still a rule-based anthropometric model, not a trained ML model.
    - Instead of hard thresholds, each category receives a smooth score.
    - The final user-facing output remains a single label: the category with the highest score.
    """

    def __init__(self, model_path="face_landmarker.task"):
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parent / model_path

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    # ---------- Utility functions ----------
    def get_point(self, landmarks, idx, width, height):
        return np.array([landmarks[idx].x * width, landmarks[idx].y * height], dtype=float)

    @staticmethod
    def safe_div(a, b, default=0.0):
        return float(a / b) if abs(b) > 1e-8 else default

    @staticmethod
    def dist(a, b):
        return float(np.linalg.norm(a - b))

    @staticmethod
    def gaussian_score(value, center, width):
        """Score is highest near center and decays smoothly."""
        width = max(width, 1e-6)
        return float(np.exp(-0.5 * ((value - center) / width) ** 2))

    @staticmethod
    def sigmoid_score(value, center, width, direction="high"):
        """Smooth threshold. direction='high' means larger values score higher."""
        width = max(width, 1e-6)
        z = (value - center) / width
        s = 1.0 / (1.0 + np.exp(-z))
        if direction == "low":
            s = 1.0 - s
        return float(s)

    @staticmethod
    def normalize_scores(raw_scores):
        cleaned = {k: max(float(v), 1e-8) for k, v in raw_scores.items()}
        total = sum(cleaned.values())
        return {k: round(v / total, 3) for k, v in cleaned.items()}

    @staticmethod
    def label_from_probs(probs):
        return max(probs.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def calc_angle(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return float(np.degrees(np.arctan2(dy, dx)))

    def analyze_image(self, image_path):
        image = mp.Image.create_from_file(image_path)
        result = self.detector.detect(image)
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        points = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]
        return {
            "landmarks": points,
            "blendshapes": result.face_blendshapes,
            "matrixes": result.facial_transformation_matrixes,
        }

    def analyze(self, image, tuning=None):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            return None, image

        landmarks = results.face_landmarks[0]
        h, w, _ = image.shape

        # ---------- Landmarks ----------
        top = self.get_point(landmarks, 10, w, h)
        bottom = self.get_point(landmarks, 152, w, h)
        left = self.get_point(landmarks, 234, w, h)
        right = self.get_point(landmarks, 454, w, h)
        jaw_left = self.get_point(landmarks, 132, w, h)
        jaw_right = self.get_point(landmarks, 361, w, h)
        forehead_left = self.get_point(landmarks, 71, w, h)
        forehead_right = self.get_point(landmarks, 301, w, h)

        # Approximate cheekbone region. These points are less extreme than 234/454.
        cheek_left = self.get_point(landmarks, 50, w, h)
        cheek_right = self.get_point(landmarks, 280, w, h)

        left_eye_inner = self.get_point(landmarks, 133, w, h)
        left_eye_outer = self.get_point(landmarks, 33, w, h)
        left_eye_top = self.get_point(landmarks, 159, w, h)
        left_eye_bottom = self.get_point(landmarks, 145, w, h)

        right_eye_inner = self.get_point(landmarks, 362, w, h)
        right_eye_outer = self.get_point(landmarks, 263, w, h)
        right_eye_top = self.get_point(landmarks, 386, w, h)
        right_eye_bottom = self.get_point(landmarks, 374, w, h)

        left_brow_inner = self.get_point(landmarks, 46, w, h)
        left_brow_peak = self.get_point(landmarks, 105, w, h)
        left_brow_outer = self.get_point(landmarks, 107, w, h)
        right_brow_inner = self.get_point(landmarks, 276, w, h)
        right_brow_peak = self.get_point(landmarks, 334, w, h)
        right_brow_outer = self.get_point(landmarks, 336, w, h)

        nose_bridge = self.get_point(landmarks, 168, w, h)
        nose_tip = self.get_point(landmarks, 1, w, h)
        nose_left = self.get_point(landmarks, 129, w, h)
        nose_right = self.get_point(landmarks, 358, w, h)

        mouth_left = self.get_point(landmarks, 61, w, h)
        mouth_right = self.get_point(landmarks, 291, w, h)
        upper_lip_top = self.get_point(landmarks, 0, w, h)
        upper_lip_bottom = self.get_point(landmarks, 13, w, h)
        lower_lip_top = self.get_point(landmarks, 14, w, h)
        lower_lip_bottom = self.get_point(landmarks, 17, w, h)

        # ---------- Measurements ----------
        face_height = self.dist(bottom, top)
        face_width = self.dist(right, left)
        jaw_width = self.dist(jaw_right, jaw_left)
        forehead_width = self.dist(forehead_right, forehead_left)
        cheek_width = self.dist(cheek_right, cheek_left)

        ratio_hw = self.safe_div(face_height, face_width)
        ratio_jc = self.safe_div(jaw_width, cheek_width)
        ratio_fc = self.safe_div(forehead_width, cheek_width)
        ratio_jf = self.safe_div(jaw_width, forehead_width)

        left_eye_width = self.dist(left_eye_outer, left_eye_inner)
        left_eye_height = self.dist(left_eye_bottom, left_eye_top)
        right_eye_width = self.dist(right_eye_outer, right_eye_inner)
        right_eye_height = self.dist(right_eye_bottom, right_eye_top)
        left_eye_ratio = self.safe_div(left_eye_width, left_eye_height)
        right_eye_ratio = self.safe_div(right_eye_width, right_eye_height)
        eye_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # angle convention: positive means outer corner is visually higher after normalization below
        left_angle_raw = self.calc_angle(left_eye_inner, left_eye_outer)
        right_angle_raw = self.calc_angle(right_eye_inner, right_eye_outer)
        # Mirror right eye angle so both eyes share the same sign convention.
        left_up_angle = -left_angle_raw
        right_up_angle = right_angle_raw
        avg_eye_up_angle = (left_up_angle + right_up_angle) / 2

        left_brow_width = self.dist(left_brow_outer, left_brow_inner)
        right_brow_width = self.dist(right_brow_outer, right_brow_inner)
        avg_brow_width = (left_brow_width + right_brow_width) / 2

        # Positive = brow tail higher than brow head, normalized by brow width.
        left_brow_tilt = self.safe_div(left_brow_inner[1] - left_brow_outer[1], left_brow_width)
        right_brow_tilt = self.safe_div(right_brow_inner[1] - right_brow_outer[1], right_brow_width)
        avg_brow_tilt = (left_brow_tilt + right_brow_tilt) / 2

        left_arch_ratio = self.safe_div(left_brow_inner[1] - left_brow_peak[1], left_brow_width)
        right_arch_ratio = self.safe_div(right_brow_inner[1] - right_brow_peak[1], right_brow_width)
        avg_arch_ratio = (left_arch_ratio + right_arch_ratio) / 2

        nose_width = self.dist(nose_right, nose_left)
        nose_length = self.dist(nose_tip, nose_bridge)
        inter_eye_dist = self.dist(right_eye_inner, left_eye_inner)
        nose_face_ratio = self.safe_div(nose_width, face_width)
        nose_eye_ratio = self.safe_div(nose_width, inter_eye_dist)
        nose_len_face_ratio = self.safe_div(nose_length, face_height)
        bridge_ratio = self.safe_div(nose_length, inter_eye_dist)

        upper_lip = self.dist(upper_lip_bottom, upper_lip_top)
        lower_lip = self.dist(lower_lip_bottom, lower_lip_top)
        total_lip = upper_lip + lower_lip
        mouth_width = self.dist(mouth_right, mouth_left)
        lip_face_ratio = self.safe_div(total_lip, face_height)
        upper_lip_share = self.safe_div(upper_lip, total_lip, default=0.5)
        mouth_eye_ratio = self.safe_div(mouth_width, inter_eye_dist)
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        eye_width_eye_dist_ratio = self.safe_div(avg_eye_width, inter_eye_dist)

        analysis = {}

        # ---------- Tuning parameters from Streamlit UI ----------
        # Bias: lower = harder to be selected; higher = easier to be selected.
        # Center: the decision midpoint of a smooth threshold.
        tuning = tuning or {}
        face_square_bias = tuning.get("face_square_bias", 0.75)
        face_round_bias = tuning.get("face_round_bias", 1.10)
        face_long_bias = tuning.get("face_long_bias", 1.15)
        face_heart_bias = tuning.get("face_heart_bias", 1.25)
        face_oval_bias = tuning.get("face_oval_bias", 0.95)
        face_square_hw_center = tuning.get("face_square_hw_center", 1.32)
        face_square_jc_center = tuning.get("face_square_jc_center", 1.02)

        eye_up_bias = tuning.get("eye_up_bias", 0.85)
        eye_down_bias = tuning.get("eye_down_bias", 1.10)
        eye_round_bias = tuning.get("eye_round_bias", 1.05)
        eye_slender_bias = tuning.get("eye_slender_bias", 1.00)
        eye_almond_bias = tuning.get("eye_almond_bias", 0.90)
        eye_up_angle_center = tuning.get("eye_up_angle_center", 6.5)
        eye_down_angle_center = tuning.get("eye_down_angle_center", -4.5)
        eye_round_ratio_center = tuning.get("eye_round_ratio_center", 2.10)

        brow_arch_bias = tuning.get("brow_arch_bias", 0.75)
        brow_up_bias = tuning.get("brow_up_bias", 1.05)
        brow_down_bias = tuning.get("brow_down_bias", 1.05)
        brow_straight_bias = tuning.get("brow_straight_bias", 1.05)
        brow_arch_center = tuning.get("brow_arch_center", 0.20)
        brow_up_center = tuning.get("brow_up_center", 0.075)
        brow_down_center = tuning.get("brow_down_center", -0.055)

        nose_wide_bias = tuning.get("nose_wide_bias", 1.10)
        nose_narrow_bias = tuning.get("nose_narrow_bias", 1.15)
        nose_high_bias = tuning.get("nose_high_bias", 1.15)
        nose_low_bias = tuning.get("nose_low_bias", 1.05)
        nose_medium_bias = tuning.get("nose_medium_bias", 0.85)
        nose_wide_center = tuning.get("nose_wide_center", 0.285)
        nose_narrow_center = tuning.get("nose_narrow_center", 0.235)
        nose_high_center = tuning.get("nose_high_center", 0.335)
        nose_low_center = tuning.get("nose_low_center", 0.285)
        nose_bridge_high_center = tuning.get("nose_bridge_high_center", 1.02)
        nose_bridge_low_center = tuning.get("nose_bridge_low_center", 0.88)
        nose_medium_width_center = tuning.get("nose_medium_width_center", 0.26)
        nose_medium_length_center = tuning.get("nose_medium_length_center", 0.31)

        lip_upper_bias = tuning.get("lip_upper_bias", 1.05)
        lip_lower_bias = tuning.get("lip_lower_bias", 1.05)
        lip_thick_bias = tuning.get("lip_thick_bias", 1.10)
        lip_thin_bias = tuning.get("lip_thin_bias", 1.05)
        lip_medium_bias = tuning.get("lip_medium_bias", 0.90)
        lip_upper_share_center = tuning.get("lip_upper_share_center", 0.54)
        lip_lower_share_center = tuning.get("lip_lower_share_center", 0.43)
        lip_thick_center = tuning.get("lip_thick_center", 0.108)
        lip_thin_center = tuning.get("lip_thin_center", 0.073)
        lip_medium_height_center = tuning.get("lip_medium_height_center", 0.09)
        lip_medium_upper_share_center = tuning.get("lip_medium_upper_share_center", 0.47)

        # ---------- Soft classification ----------
        # Face shape: designed to reduce over-selection of square/oval while allowing rare labels to appear.
        face_raw = {
            "長臉 (Long)": face_long_bias * self.sigmoid_score(ratio_hw, 1.44, 0.055, "high"),
            "圓臉 (Round)": face_round_bias * self.sigmoid_score(ratio_hw, 1.34, 0.06, "low") * self.gaussian_score(ratio_jc, 0.90, 0.12),
            "方臉 (Square)": face_square_bias * self.sigmoid_score(ratio_hw, face_square_hw_center, 0.06, "low") * self.sigmoid_score(ratio_jc, face_square_jc_center, 0.055, "high"),
            "心形臉 (Heart)": face_heart_bias * self.sigmoid_score(ratio_fc - ratio_jc, 0.08, 0.04, "high") * self.sigmoid_score(ratio_jf, 0.90, 0.06, "low"),
            "橢圓臉 (Oval)": face_oval_bias * self.gaussian_score(ratio_hw, 1.42, 0.11) * self.gaussian_score(ratio_jc, 0.90, 0.14),
        }
        face_probs = self.normalize_scores(face_raw)
        analysis["face_shape"] = self.label_from_probs(face_probs)
        analysis["face_shape_probs"] = face_probs

        # Eye shape: tilt categories can win even if eye_ratio is almond-like.
        eye_raw = {
            "上揚眼 (Upturned)": eye_up_bias * self.sigmoid_score(avg_eye_up_angle, eye_up_angle_center, 1.8, "high"),
            "下垂眼 (Downturned)": eye_down_bias * self.sigmoid_score(avg_eye_up_angle, eye_down_angle_center, 1.8, "low"),
            "圓眼 (Round)": eye_round_bias * self.sigmoid_score(eye_ratio, eye_round_ratio_center, 0.18, "low"),
            "細長眼 (Slender)": eye_slender_bias * self.sigmoid_score(eye_ratio, 2.95, 0.22, "high"),
            "杏仁眼 (Almond)": eye_almond_bias * self.gaussian_score(eye_ratio, 2.55, 0.35) * self.gaussian_score(avg_eye_up_angle, 0.0, 5.5),
        }
        eye_probs = self.normalize_scores(eye_raw)
        analysis["eye_shape"] = self.label_from_probs(eye_probs)
        analysis["eye_shape_probs"] = eye_probs

        # Eyebrow shape: arch is based on peak height; tilt based on tail/head slope.
        brow_raw = {
            "拱眉 (Arched)": brow_arch_bias * self.sigmoid_score(avg_arch_ratio, brow_arch_center, 0.035, "high"),
            "上揚眉 (Upturned)": brow_up_bias * self.sigmoid_score(avg_brow_tilt, brow_up_center, 0.025, "high"),
            "下垂眉 (Downturned)": brow_down_bias * self.sigmoid_score(avg_brow_tilt, brow_down_center, 0.025, "low"),
            "平眉 (Straight)": brow_straight_bias * self.gaussian_score(avg_brow_tilt, 0.0, 0.07) * self.sigmoid_score(avg_arch_ratio, 0.18, 0.04, "low"),
        }
        brow_probs = self.normalize_scores(brow_raw)
        analysis["eyebrow_shape"] = self.label_from_probs(brow_probs)
        analysis["eyebrow_shape_probs"] = brow_probs

        # Nose shape: high/low bridge from bridge length proxy; wide/narrow from alar width.
        nose_raw = {
            "寬鼻 (Wide)": nose_wide_bias * self.sigmoid_score(nose_face_ratio, nose_wide_center, 0.025, "high"),
            "窄鼻 (Narrow)": nose_narrow_bias * self.sigmoid_score(nose_face_ratio, nose_narrow_center, 0.025, "low"),
            "高鼻樑 (High bridge)": nose_high_bias * self.sigmoid_score(nose_len_face_ratio, nose_high_center, 0.025, "high") * self.sigmoid_score(bridge_ratio, nose_bridge_high_center, 0.10, "high"),
            "低鼻樑 (Low bridge)": nose_low_bias * self.sigmoid_score(nose_len_face_ratio, nose_low_center, 0.025, "low") * self.sigmoid_score(bridge_ratio, nose_bridge_low_center, 0.10, "low"),
            "中等鼻 (Medium)": nose_medium_bias * self.gaussian_score(nose_face_ratio, nose_medium_width_center, 0.04) * self.gaussian_score(nose_len_face_ratio, nose_medium_length_center, 0.045),
        }
        nose_probs = self.normalize_scores(nose_raw)
        analysis["nose_shape"] = self.label_from_probs(nose_probs)
        analysis["nose_shape_probs"] = nose_probs

        lip_raw = {
            "上唇較厚 (Thicker upper lip)": lip_upper_bias * self.sigmoid_score(upper_lip_share, lip_upper_share_center, 0.035, "high"),
            "下唇較厚 (Thicker lower lip)": lip_lower_bias * self.sigmoid_score(upper_lip_share, lip_lower_share_center, 0.035, "low"),
            "厚唇 (Thick)": lip_thick_bias * self.sigmoid_score(lip_face_ratio, lip_thick_center, 0.012, "high"),
            "薄唇 (Thin)": lip_thin_bias * self.sigmoid_score(lip_face_ratio, lip_thin_center, 0.010, "low"),
            "中等唇 (Medium)": lip_medium_bias * self.gaussian_score(lip_face_ratio, lip_medium_height_center, 0.018) * self.gaussian_score(upper_lip_share, lip_medium_upper_share_center, 0.08),
        }
        lip_probs = self.normalize_scores(lip_raw)
        analysis["lips"] = self.label_from_probs(lip_probs)
        analysis["lips_probs"] = lip_probs

        # Measurements shown in Streamlit.
        analysis["eye_width_eye_dist_ratio"] = round(eye_width_eye_dist_ratio, 3)
        analysis["nose_eye_ratio"] = round(nose_eye_ratio, 3)
        analysis["mouth_eye_ratio"] = round(mouth_eye_ratio, 3)

        # Extra diagnostic values. Useful if you want to tune thresholds later.
        analysis["debug_metrics"] = {
            "face_height_width": round(ratio_hw, 3),
            "jaw_cheek_ratio": round(ratio_jc, 3),
            "forehead_cheek_ratio": round(ratio_fc, 3),
            "eye_width_height": round(eye_ratio, 3),
            "eye_up_angle_deg": round(avg_eye_up_angle, 2),
            "brow_tilt_ratio": round(avg_brow_tilt, 3),
            "brow_arch_ratio": round(avg_arch_ratio, 3),
            "nose_face_ratio": round(nose_face_ratio, 3),
            "nose_len_face_ratio": round(nose_len_face_ratio, 3),
            "upper_lip_share": round(upper_lip_share, 3),
            "lip_face_ratio": round(lip_face_ratio, 3),
        }

        # ---------- Draw annotations ----------
        annotated_image = image.copy()

        def draw_pts(pts, color, closed=False):
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_image, [pts], closed, color, 2)

        draw_pts([top, left, bottom, right], (0, 255, 0), True)
        draw_pts([left_eye_inner, left_eye_top, left_eye_outer, left_eye_bottom], (255, 0, 0), True)
        draw_pts([right_eye_inner, right_eye_top, right_eye_outer, right_eye_bottom], (255, 0, 0), True)
        draw_pts([left_brow_inner, left_brow_peak, left_brow_outer], (0, 0, 255))
        draw_pts([right_brow_inner, right_brow_peak, right_brow_outer], (0, 0, 255))
        draw_pts([nose_bridge, nose_tip, nose_left, nose_right, nose_tip], (0, 255, 255))
        draw_pts([mouth_left, upper_lip_top, mouth_right, lower_lip_bottom], (255, 0, 255), True)

        return analysis, annotated_image
