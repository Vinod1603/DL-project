import ast
import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length=50):
    """Draw a decorative border around a rectangle."""
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Top-left
    cv2.line(img, (x1, y1), (x1 + line_length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_length), color, thickness)

    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length), color, thickness)

    # Top-right
    cv2.line(img, (x2, y1), (x2 - line_length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length), color, thickness)

    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length), color, thickness)


def parse_bounding_box(bbox_string):
    """Parse bounding box string to integer coordinates."""
    bbox_string = bbox_string.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
    return list(map(int, ast.literal_eval(bbox_string)))


def prepare_license_plate_data(results, video_path, width, height):
    """Prepare license plate data with crops for overlaying on frames."""
    license_plate_data = {}
    cap = cv2.VideoCapture(video_path)

    for car_id in results['car_id'].unique():
        car_data = results[results['car_id'] == car_id]
        max_score_row = car_data.loc[car_data['license_number_score'].idxmax()]

        frame_nmr = int(max_score_row['frame_nmr'])
        license_number = max_score_row['license_number']
        bbox = parse_bounding_box(max_score_row['license_plate_bbox'])

        x1, y1, x2, y2 = bbox

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
        ret, frame = cap.read()

        if ret and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height:
            license_crop = frame[y1:y2, x1:x2]
            if license_crop.size > 0:
                new_height = 400
                new_width = int((x2 - x1) * new_height / (y2 - y1))
                license_crop = cv2.resize(license_crop, (new_width, new_height))
                license_plate_data[car_id] = {
                    'license_crop': license_crop,
                    'license_plate_number': license_number
                }

    cap.release()
    return license_plate_data


def process_frame(frame, df, license_plate_data, frame_shape):
    """Process a single frame by overlaying bounding boxes and license plates."""
    for _, row in df.iterrows():
        car_bbox = parse_bounding_box(row['car_bbox'])
        license_bbox = parse_bounding_box(row['license_plate_bbox'])

        car_x1, car_y1, car_x2, car_y2 = car_bbox
        x1, y1, x2, y2 = license_bbox

        # Draw car bounding box
        draw_border(frame, (car_x1, car_y1), (car_x2, car_y2), color=(0, 255, 0), thickness=25)

        # Draw license plate bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 12)

        # Overlay license plate image and text
        if row['car_id'] in license_plate_data:
            license_crop = license_plate_data[row['car_id']]['license_crop']
            license_number = license_plate_data[row['car_id']]['license_plate_number']

            H, W, _ = license_crop.shape
            top = max(0, car_y1 - H - 100)
            left = max(0, (car_x1 + car_x2 - W) // 2)

            if top + H <= frame_shape[0] and left + W <= frame_shape[1]:
                frame[top:top + H, left:left + W] = license_crop

                (text_width, text_height), _ = cv2.getTextSize(license_number, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)
                text_x = max(0, (car_x1 + car_x2 - text_width) // 2)
                text_y = max(0, top - 10)
                cv2.putText(frame, license_number, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 5)

    return frame


def main():
    # Load results CSV
    results = pd.read_csv('./test_interpolated.csv')

    # Load video
    video_path = 'sample2.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('./out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Prepare license plate data
    license_plate_data = prepare_license_plate_data(results, video_path, width, height)

    # Process frames
    frame_nmr = -1
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if not ret:
            break

        df = results[results['frame_nmr'] == frame_nmr]
        frame = process_frame(frame, df, license_plate_data, frame.shape)

        # Write processed frame to output video
        out.write(frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
