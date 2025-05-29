import cv2
import time
import logging
from ultralytics import YOLO
from datetime import datetime
from picamera2 import Picamera2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load YOLO model using Ultralytics API
model = YOLO('best.pt')

# Set label target (misalnya 'burung', 'belalang', dll)
TARGET_CLASSES = ['burung pipit', 'tikus', 'wereng']

# Initialize Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (320, 240)}))
picam2.start()

logging.info("Sistem Deteksi Hama Dimulai...")

try:
    while True:
        # Capture frame from Pi Camera
        frame = picam2.capture_array()

        # Detect objects with YOLO
        results = model(frame)
        boxes = results[0].boxes
        names = results[0].names

        # Display results on frame
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
            if cls_name in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                logging.info(f"Deteksi: {cls_name} dengan kepercayaan {conf:.2f}")

        # Display frame
        cv2.imshow("Deteksi Hama", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Add a small delay to reduce CPU load
        time.sleep(0.1)

except KeyboardInterrupt:
    logging.info("Dihentikan oleh pengguna.")
except Exception as e:
    logging.error(f"An error occurred: {e}")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
