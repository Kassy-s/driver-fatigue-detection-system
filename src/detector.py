import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.metrics import FatigueMetrics

class DrowsinessDetector:
    """
    Production-grade facial landmark tracker using MediaPipe Tasks Face Landmarker.
    Tracks 478 3D facial landmarks to calculate true EAR and MAR metrics in real time.
    """
    # Canonical landmark index mappings
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    MOUTH_INDICES = [13, 14, 78, 308]  # Upper lip, lower lip, left corner, right corner

    def __init__(self, model_path="models/face_landmarker.task", ear_threshold=0.21, consecutive_frames=12, mar_threshold=0.60):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.mar_threshold = mar_threshold

        self.closed_frame_count = 0
        self.drowsy_event_count = 0
        self.yawn_event_count = 0

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Download it first.")

        # Initialize MediaPipe Tasks FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)

        status = {
            "face_detected": False,
            "drowsy_alert": False,
            "yawn_alert": False,
            "avg_ear": 0.0,
            "mar": 0.0,
            "landmarks_px": []
        }

        if not detection_result.face_landmarks:
            self.closed_frame_count = 0
            return status

        status["face_detected"] = True
        landmarks = detection_result.face_landmarks[0]

        def get_coords(indices):
            return [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]

        left_pts = get_coords(self.LEFT_EYE_INDICES)
        right_pts = get_coords(self.RIGHT_EYE_INDICES)
        mouth_pts = get_coords(self.MOUTH_INDICES)

        left_ear = FatigueMetrics.calculate_ear(left_pts)
        right_ear = FatigueMetrics.calculate_ear(right_pts)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = FatigueMetrics.calculate_mar(mouth_pts)

        status["avg_ear"] = round(avg_ear, 3)
        status["mar"] = round(mar, 3)
        status["landmarks_px"] = left_pts + right_pts + mouth_pts

        # Drowsiness logic: Eyes closed below threshold
        if avg_ear < self.ear_threshold:
            self.closed_frame_count += 1
            if self.closed_frame_count >= self.consecutive_frames:
                status["drowsy_alert"] = True
                self.drowsy_event_count += 1
        else:
            self.closed_frame_count = 0

        # Yawn logic: Mouth open wide
        if mar > self.mar_threshold:
            status["yawn_alert"] = True
            self.yawn_event_count += 1

        return status

    def release(self):
        self.detector.close()