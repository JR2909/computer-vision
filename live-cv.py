import os
import cv2
from ultralytics import YOLO

os.environ["DISPLAY"] = ":0"
os.environ["QT_QPA_PLATFORM"] = "xcb"

def run_live_detection():
  model = YOLO("yolo11n.pt")
  cap = cv2.VideoCapture(0)

  if not cap.isOpened():
    print("Fehler: Konnte die Webcam nicht öffnen.")
    return

  print("Starte Live-Erkennung. Drücke 'q' zum Beenden.")

  try:
    while cap.isOpened():
      success, frame = cap.read()
      if not success:
        print("Fehler beim Empfangen des Video-Frames.")
        break

      results = model(frame, stream=True)
      for r in results:
        annotated_frame = r.plot()

      cv2.imshow("Live Object Detection", annotated_frame)

      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
  finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam geschlossen.")


if __name__ == "__main__":
  run_live_detection()