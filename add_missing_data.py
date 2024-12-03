import csv
import numpy as np
from scipy.interpolate import interp1d


def parse_csv_data(data):
    """Parse and structure CSV data for processing."""
    return {
        'frame_numbers': np.array([int(row['frame_nmr']) for row in data]),
        'car_ids': np.array([int(float(row['car_id'])) for row in data]),
        'car_bboxes': np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data]),
        'license_plate_bboxes': np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data]),
    }


def interpolate_values(x, y_start, y_end, frames_gap):
    """Perform linear interpolation for missing values."""
    x_new = np.linspace(x[0], x[1], num=frames_gap, endpoint=False)
    interp_func = interp1d(x, np.vstack((y_start, y_end)), axis=0, kind='linear')
    return interp_func(x_new)[1:]  # Exclude the starting frame value


def interpolate_car_data(car_frame_numbers, car_bboxes, license_plate_bboxes):
    """Interpolate bounding boxes for a specific car."""
    car_bboxes_interpolated = []
    license_plate_bboxes_interpolated = []

    for i in range(len(car_frame_numbers)):
        frame_number = car_frame_numbers[i]
        car_bbox = car_bboxes[i]
        license_plate_bbox = license_plate_bboxes[i]

        if i > 0:
            prev_frame_number = car_frame_numbers[i - 1]
            prev_car_bbox = car_bboxes_interpolated[-1]
            prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

            if frame_number - prev_frame_number > 1:
                # Interpolate missing frames
                frames_gap = frame_number - prev_frame_number
                car_bboxes_interpolated.extend(
                    interpolate_values([prev_frame_number, frame_number], prev_car_bbox, car_bbox, frames_gap)
                )
                license_plate_bboxes_interpolated.extend(
                    interpolate_values([prev_frame_number, frame_number], prev_license_plate_bbox, license_plate_bbox, frames_gap)
                )

        car_bboxes_interpolated.append(car_bbox)
        license_plate_bboxes_interpolated.append(license_plate_bbox)

    return car_bboxes_interpolated, license_plate_bboxes_interpolated


def build_interpolated_data(data, parsed_data):
    """Construct the interpolated dataset."""
    frame_numbers = parsed_data['frame_numbers']
    car_ids = parsed_data['car_ids']
    car_bboxes = parsed_data['car_bboxes']
    license_plate_bboxes = parsed_data['license_plate_bboxes']

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)

    for car_id in unique_car_ids:
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_selected = car_bboxes[car_mask]
        license_plate_bboxes_selected = license_plate_bboxes[car_mask]

        car_bboxes_interpolated, license_plate_bboxes_interpolated = interpolate_car_data(
            car_frame_numbers, car_bboxes_selected, license_plate_bboxes_selected
        )

        for i, frame_number in enumerate(range(car_frame_numbers[0], car_frame_numbers[0] + len(car_bboxes_interpolated))):
            row = {
                'frame_nmr': str(frame_number),
                'car_id': str(car_id),
                'car_bbox': ' '.join(map(str, car_bboxes_interpolated[i])),
                'license_plate_bbox': ' '.join(map(str, license_plate_bboxes_interpolated[i])),
            }

            if str(frame_number) not in map(str, car_frame_numbers):
                row.update({
                    'license_plate_bbox_score': '0',
                    'license_number': '0',
                    'license_number_score': '0',
                })
            else:
                original_row = next(
                    row for row in data
                    if int(row['frame_nmr']) == frame_number and int(float(row['car_id'])) == car_id
                )
                row.update({
                    'license_plate_bbox_score': original_row.get('license_plate_bbox_score', '0'),
                    'license_number': original_row.get('license_number', '0'),
                    'license_number_score': original_row.get('license_number_score', '0'),
                })

            interpolated_data.append(row)

    return interpolated_data


# Main Execution
if __name__ == "__main__":
    # Load CSV data
    with open('test.csv', 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    # Parse and interpolate data
    parsed_data = parse_csv_data(data)
    interpolated_data = build_interpolated_data(data, parsed_data)

    # Write updated data to a new CSV file
    header = [
        'frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox',
        'license_plate_bbox_score', 'license_number', 'license_number_score'
    ]
    with open('test_interpolated.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(interpolated_data)
