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

    def analyze(self, image):
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

        # ---------- Soft classification ----------
        # Face shape: tuned to reduce false Square predictions.
        # Square should require a clearly short/wide face AND a relatively broad jaw.
        face_raw = {
            "長臉 (Long)": 1.20 * self.sigmoid_score(ratio_hw, 1.43, 0.060, "high"),
            "圓臉 (Round)": 1.18 * self.sigmoid_score(ratio_hw, 1.34, 0.065, "low") * self.gaussian_score(ratio_jc, 0.88, 0.13),
            "方臉 (Square)": 0.58 * self.sigmoid_score(ratio_hw, 1.30, 0.045, "low") * self.sigmoid_score(ratio_jc, 1.03, 0.045, "high"),
            "心形臉 (Heart)": 1.28 * self.sigmoid_score(ratio_fc - ratio_jc, 0.075, 0.045, "high") * self.sigmoid_score(ratio_jf, 0.92, 0.065, "low"),
            "橢圓臉 (Oval)": 1.10 * self.gaussian_score(ratio_hw, 1.43, 0.13) * self.gaussian_score(ratio_jc, 0.90, 0.16),
        }
        face_probs = self.normalize_scores(face_raw)
        analysis["face_shape"] = self.label_from_probs(face_probs)
        analysis["face_shape_probs"] = face_probs

        # Eye shape: tuned to reduce false Upturned predictions.
        # Upturned now needs a stronger positive tilt; mild tilt is absorbed by Almond.
        eye_raw = {
            "上揚眼 (Upturned)": 0.72 * self.sigmoid_score(avg_eye_up_angle, 7.0, 1.5, "high"),
            "下垂眼 (Downturned)": 1.00 * self.sigmoid_score(avg_eye_up_angle, -6.0, 1.7, "low"),
            "圓眼 (Round)": 1.10 * self.sigmoid_score(eye_ratio, 2.05, 0.20, "low"),
            "細長眼 (Slender)": 1.02 * self.sigmoid_score(eye_ratio, 2.95, 0.24, "high"),
            "杏仁眼 (Almond)": 1.25 * self.gaussian_score(eye_ratio, 2.55, 0.42) * self.gaussian_score(avg_eye_up_angle, 0.0, 7.5),
        }
        eye_probs = self.normalize_scores(eye_raw)
        analysis["eye_shape"] = self.label_from_probs(eye_probs)
        analysis["eye_shape_probs"] = eye_probs

        # Eyebrow shape: tuned to reduce false Arched predictions.
        # Arched now needs a clearer peak; mild curvature is treated as Straight/tilted brow.
        brow_raw = {
            "拱眉 (Arched)": 0.68 * self.sigmoid_score(avg_arch_ratio, 0.23, 0.035, "high"),
            "上揚眉 (Upturned)": 1.00 * self.sigmoid_score(avg_brow_tilt, 0.085, 0.028, "high") * self.sigmoid_score(avg_arch_ratio, 0.26, 0.05, "low"),
            "下垂眉 (Downturned)": 1.00 * self.sigmoid_score(avg_brow_tilt, -0.070, 0.028, "low") * self.sigmoid_score(avg_arch_ratio, 0.26, 0.05, "low"),
            "平眉 (Straight)": 1.35 * self.gaussian_score(avg_brow_tilt, 0.0, 0.085) * self.sigmoid_score(avg_arch_ratio, 0.22, 0.045, "low"),
        }
        brow_probs = self.normalize_scores(brow_raw)
        analysis["eyebrow_shape"] = self.label_from_probs(brow_probs)
        analysis["eyebrow_shape_probs"] = brow_probs

        # Nose shape: high/low bridge from bridge length proxy; wide/narrow from alar width.
        nose_raw = {
            "寬鼻 (Wide)": 1.10 * self.sigmoid_score(nose_face_ratio, 0.285, 0.025, "high"),
            "窄鼻 (Narrow)": 1.15 * self.sigmoid_score(nose_face_ratio, 0.235, 0.025, "low"),
            "高鼻樑 (High bridge)": 1.15 * self.sigmoid_score(nose_len_face_ratio, 0.335, 0.025, "high") * self.sigmoid_score(bridge_ratio, 1.02, 0.10, "high"),
            "低鼻樑 (Low bridge)": 1.05 * self.sigmoid_score(nose_len_face_ratio, 0.285, 0.025, "low") * self.sigmoid_score(bridge_ratio, 0.88, 0.10, "low"),
            "中等鼻 (Medium)": 0.85 * self.gaussian_score(nose_face_ratio, 0.26, 0.04) * self.gaussian_score(nose_len_face_ratio, 0.31, 0.045),
        }
        nose_probs = self.normalize_scores(nose_raw)
        analysis["nose_shape"] = self.label_from_probs(nose_probs)
        analysis["nose_shape_probs"] = nose_probs

        lip_raw = {
            "上唇較厚 (Thicker upper lip)": 1.05 * self.sigmoid_score(upper_lip_share, 0.54, 0.035, "high"),
            "下唇較厚 (Thicker lower lip)": 1.05 * self.sigmoid_score(upper_lip_share, 0.43, 0.035, "low"),
            "厚唇 (Thick)": 1.10 * self.sigmoid_score(lip_face_ratio, 0.108, 0.012, "high"),
            "薄唇 (Thin)": 1.05 * self.sigmoid_score(lip_face_ratio, 0.073, 0.010, "low"),
            "中等唇 (Medium)": 0.90 * self.gaussian_score(lip_face_ratio, 0.09, 0.018) * self.gaussian_score(upper_lip_share, 0.47, 0.08),
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
