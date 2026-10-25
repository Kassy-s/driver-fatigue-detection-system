import os
import cv2
import json
import numpy as np
from datetime import datetime
from src.detector import DrowsinessDetector

def main():
    print("=" * 65)
    print("   REAL-TIME DRIVER FATIGUE & DISTRACTION MONITORING SYSTEM   ")
    print("=" * 65)

    detector = DrowsinessDetector(
        model_path="models/face_landmarker.task",
        ear_threshold=0.21,
        consecutive_frames=12,
        mar_threshold=0.60
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Unable to open webcam.")
        return

    print("[✓] Camera initialized. Live telemetry feed active.")
    print("[i] Press 'q' to end the monitoring session.\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            status = detector.process_frame(frame)

            # Draw HUD telemetry
            cv2.rectangle(frame, (10, 10), (320, 110), (20, 20, 20), -1)
            ear_color = (0, 0, 255) if status["drowsy_alert"] else (0, 255, 0)
            mar_color = (0, 165, 255) if status["yawn_alert"] else (255, 255, 0)

            cv2.putText(frame, f"EAR: {status['avg_ear']:.3f} (Lim: {detector.ear_threshold})", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, ear_color, 2)
            cv2.putText(frame, f"MAR: {status['mar']:.3f} (Lim: {detector.mar_threshold})", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, mar_color, 2)

            # Draw key facial points
            for (px, py) in status["landmarks_px"]:
                cv2.circle(frame, (int(px), int(py)), 2, (0, 255, 255), -1)

            # Alerts
            if status["drowsy_alert"]:
                cv2.putText(frame, "CRITICAL: DROWSINESS DETECTED!", (50, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            if status["yawn_alert"]:
                cv2.putText(frame, "WARNING: YAWNING DETECTED!", (50, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            cv2.imshow("Driver Fatigue Monitoring System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()

        os.makedirs("reports", exist_ok=True)
        summary_file = os.path.join("reports", "session_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "session_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_drowsy_alerts": detector.drowsy_event_count,
                "total_yawn_alerts": detector.yawn_event_count
            }, f, indent=4)
        print(f"\n[✓] Telemetry report saved to: {summary_file}")
        print("=" * 65)

if __name__ == "__main__":
    main()