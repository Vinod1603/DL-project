from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort
from util import get_car, read_license_plate, write_csv

def detect_vehicles(model, frame, vehicle_classes):
    """Detect vehicles in the frame."""
    detections = model(frame)[0]
    return [
        [x1, y1, x2, y2, score] 
        for x1, y1, x2, y2, score, class_id in detections.boxes.data.tolist() 
        if int(class_id) in vehicle_classes
    ]

def detect_license_plates(model, frame):
    """Detect license plates in the frame."""
    detections = model(frame)[0]
    return [
        [x1, y1, x2, y2, score] 
        for x1, y1, x2, y2, score, class_id in detections.boxes.data.tolist()
    ]

def process_license_plate(frame, bbox):
    """Extract and process the license plate from the frame."""
    x1, y1, x2, y2 = map(int, bbox)
    plate_crop = frame[y1:y2, x1:x2]
    plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, plate_thresh = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
    return plate_thresh

def track_and_assign(mot_tracker, vehicle_detections, license_plate_detections, frame, results, frame_nmr):
    """Track vehicles and assign license plates to cars."""
    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(vehicle_detections))

    for license_plate in license_plate_detections:
        if len(license_plate) == 5:
            x1, y1, x2, y2, score = license_plate
            class_id = None  # Default if class_id is missing
        elif len(license_plate) == 6:
            x1, y1, x2, y2, score, class_id = license_plate
        else:
            continue  # Skip invalid detections

        # Assign license plate to a car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car((x1, y1, x2, y2, score, class_id), track_ids)
        if car_id != -1:
            # Process license plate
            plate_thresh = process_license_plate(frame, (x1, y1, x2, y2))
            plate_text, plate_text_score = read_license_plate(plate_thresh)

            if plate_text is not None:
                # Save results
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': plate_text,
                        'bbox_score': score,
                        'text_score': plate_text_score,
                    }
                }

# Main Functionality
def main(video_path, output_csv, coco_model_path, plate_model_path):
    # Initialize models and tracker
    coco_model = YOLO(coco_model_path)
    license_plate_model = YOLO(plate_model_path)
    mot_tracker = Sort()

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    vehicles = [2, 3, 5, 7]  # Vehicle classes
    results = {}
    frame_nmr = -1
    ret = True

    while ret:
        ret, frame = cap.read()
        if ret:
            frame_nmr += 1
            results[frame_nmr] = {}

            # Detect vehicles and license plates
            vehicle_detections = detect_vehicles(coco_model, frame, vehicles)
            license_plate_detections = detect_license_plates(license_plate_model, frame)

            # Track and assign license plates
            track_and_assign(mot_tracker, vehicle_detections, license_plate_detections, frame, results, frame_nmr)

    cap.release()

    # Write results to CSV
    write_csv(results, output_csv)

# Run the script
if __name__ == "__main__":
    main(
        video_path='./sample2.mp4',
        output_csv='./test.csv',
        coco_model_path='yolov8n.pt',
        plate_model_path='license_plate_detector.pt',
    )
