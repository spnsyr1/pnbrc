import cv2 as cv
import os
from ultralytics import YOLO
import supervision as sv

def predict_from_webcam(save_video=False, filename=None, cam_id=0):
    try:
        # Inisialisasi kamera
        cap = cv.VideoCapture(cam_id)
        if not cap.isOpened():
            raise Exception("Webcam tidak bisa dibuka!")

        # Ambil ukuran dan fps dari webcam
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS)) or 30  # Default ke 30 fps jika tidak tersedia

        # Set up annotators
        thickness = sv.calculate_optimal_line_thickness((w, h))
        text_scale = sv.calculate_optimal_text_scale((w, h))
        box_annotator = sv.BoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
        label_annotator = sv.LabelAnnotator(
            text_scale=text_scale, text_thickness=thickness,
            text_position=sv.Position.TOP_LEFT, color_lookup=sv.ColorLookup.TRACK
        )

        model = YOLO("best.pt")
        tracker = sv.ByteTrack(frame_rate=fps)
        class_dict = model.names

        if save_video and filename:
            os.makedirs("detected_videos", exist_ok=True)
            save_path = os.path.join("detected_videos", filename)
            out = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*"XVID"), fps, (w, h))
        else:
            out = None

        # Proses frame-frame dari webcam
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            if detections.tracker_id is not None:
                detections = detections[detections.confidence > 0.6]

                labels = [
                    f"{class_dict[cls]} {conf*100:.1f}%"
                    for cls, conf in zip(detections.class_id, detections.confidence)
                ]

                box_annotator.annotate(frame, detections=detections)
                label_annotator.annotate(frame, detections=detections, labels=labels)

            if save_video and out:
                out.write(frame)

            cv.imshow("YOLO11 Webcam Detection", frame)

            if cv.waitKey(1) & 0xFF == ord('p'):
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        cap.release()
        if save_video and out:
            out.release()
        cv.destroyAllWindows()
        print("Webcam processing selesai, semua resource dilepas.")

predict_from_webcam(save_video=True, filename="hasil_webcam.avi")