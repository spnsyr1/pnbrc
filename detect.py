import cv2
import time
import torch
from picamera2 import Picamera2
from datetime import datetime

# Load YOLO model (ganti dengan path model YOLOv11 kamu)
model = torch.hub.load('ultralytics/yolov8n', 'custom', path='best.pt', force_reload=True)

# Set label target (misalnya 'burung', 'belalang', dll)
TARGET_CLASSES = ['burung pipit', 'tikus', 'wereng']

# Inisialisasi kamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

print("[INFO] Sistem Deteksi Hama Dimulai...")

try:
    while True:
        # Ambil frame dari Pi Camera
        frame = picam2.capture_array()

        # Deteksi objek dengan YOLO
        results = model(frame)

        # Konversi hasil ke pandas dataframe
        df = results.pandas().xyxy[0]

        # Filter deteksi hanya target yang diinginkan
        detected = df[df['name'].isin(TARGET_CLASSES)]

        # Tampilkan hasil di frame
        for _, row in detected.iterrows():
            x1, y1, x2, y2, conf, cls_name = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Log deteksi
            print(f"[{datetime.now()}] Deteksi: {cls_name} dengan kepercayaan {conf:.2f}")

            # TODO: Kirim notifikasi WhatsApp/Twilio jika perlu

        # Tampilkan frame (jika pakai monitor)
        cv2.imshow("Deteksi Hama", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Dihentikan oleh pengguna.")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
