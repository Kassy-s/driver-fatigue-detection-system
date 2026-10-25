import numpy as np

class FatigueMetrics:
    """
    Computes facial geometric ratios: Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
    """
    @staticmethod
    def euclidean_distance(pt1, pt2):
        """Calculates the Euclidean distance between two 2D/3D points."""
        return np.linalg.norm(np.array(pt1) - np.array(pt2))

    @classmethod
    def calculate_ear(cls, eye_landmarks):
        """
        Calculates Eye Aspect Ratio (EAR) given 6 standard ocular landmark coordinates:
        P1, P2, P3, P4, P5, P6.
        EAR = (|P2 - P6| + |P3 - P5|) / (2 * |P1 - P4|)
        """
        if len(eye_landmarks) != 6:
            return 0.0

        p1, p2, p3, p4, p5, p6 = eye_landmarks
        
        vertical_1 = cls.euclidean_distance(p2, p6)
        vertical_2 = cls.euclidean_distance(p3, p5)
        horizontal = cls.euclidean_distance(p1, p4)

        if horizontal == 0:
            return 0.0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return float(ear)

    @classmethod
    def calculate_mar(cls, mouth_landmarks):
        """
        Calculates Mouth Aspect Ratio (MAR) to detect yawning events.
        Takes 4 key points: Top lip, Bottom lip, Left corner, Right corner.
        MAR = |Top - Bottom| / |Left - Right|
        """
        if len(mouth_landmarks) != 4:
            return 0.0

        top, bottom, left, right = mouth_landmarks
        vertical = cls.euclidean_distance(top, bottom)
        horizontal = cls.euclidean_distance(left, right)

        if horizontal == 0:
            return 0.0

        return float(vertical / horizontal)