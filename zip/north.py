import cv2
import numpy as np
import csv
from ultralytics import YOLO

# Function to split frame into quadrants
def split_frame(frame, quadrant):
    height, width, _ = frame.shape
    print(frame.shape)
    mid_height = height // 2
    mid_width = width // 2
    # Define the cropping coordinates for eqach quadrant
    crop_coords = [
        (500, mid_height-100, 1000, mid_width-300),  # West quadrant
        (300, mid_height-300, mid_width, width-1000),  # North quadran
        (mid_height+300, height, 300, mid_width+1000),  # South quadrant
        (mid_height, height-50, mid_width+200, width)  # East quadrant
    ]
    # Return the selected quadrant based on the user's choice
    if quadrant == "N":
        return frame[crop_coords[1][0]:crop_coords[1][1], crop_coords[1][2]:crop_coords[1][3]]
    elif quadrant == "E":
        return frame[crop_coords[3][0]:crop_coords[3][1], crop_coords[3][2]:crop_coords[3][3]]
    elif quadrant == "S":
        return frame[crop_coords[2][0]:crop_coords[2][1], crop_coords[2][2]:crop_coords[2][3]]
    elif quadrant == "W":
        return frame[crop_coords[0][0]:crop_coords[0][1], crop_coords[0][2]:crop_coords[0][3]]
    else:
        print("Invalid input. Defaulting to the entire frame.")
        return frame  # Return the entire frame if the input is invalid

def apply_transformation(frame, dx, dy, width, height):
    transformation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, transformation_matrix, (width, height))

transformations = []
with open(r'C:\Users\saver\3010project\transpo\zip\transformations.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header
    for row in csv_reader:
        transformations.append((int(row[0]), float(row[1]), float(row[2])))

model = YOLO('yolov8x.pt')

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Cursor Position: x={x}, y={y}")

cv2.namedWindow("Processed Frame")
cv2.setMouseCallback("Processed Frame", mouse_callback)

video_path = r"C:\Users\saver\3010project\transpo\zip\videoplayback.webm"
cap = cv2.VideoCapture(video_path)
width, height = int(cap.get(3)), int(cap.get(4))
if not cap.isOpened():
    print("Error opening video file")
    exit()

lines = {
    "N": [((62, 436), (610, 102)), 
          ((0, 525), (649, 115)), 
          ((0, 574), (690, 128)), 
          ((70, 580), (685, 177)), 
          ((153, 580), (710, 200))],
    "E": [((298-300, 155-55), (1520-300, 831-55)), 
          ((342-300, 121-55), (1550-300, 770-55)), 
          ((386-300, 101-55), (1600-300, 737-55)), 
          ((425-300, 70-55), (1645-300, 680-55)), 
          ((480-300, 44-55), (1700-300, 641-55)),
          ((525-300, 13-55), (1740-300, 600-55))],
    "S": [((1270, 190), (560, 755)), 
          ((1200, 153), (465, 715)), 
          ((1133, 140), (400, 660)), 
          ((1070, 107), (335, 620)), 
          ((1010, 70), (275, 575)),
          ((960, 42), (235, 513))],
    "W": [((364, 470), (0, 251)), 
          ((415, 445), (0, 205)), 
          ((470, 420), (0, 160)), 
          ((520, 396), (0, 115)),
          ((563, 371), (0, 70)),
          ((606, 347), (0, 25))]
}
nys=60
nye=35
new_lines =     {"N": [((62, 436+nys), (610, 102+nye)), 
          ((0, 525+nys), (649, 115+nye)), 
          ((0, 574+nys), (690, 128+nye)), 
          ((70, 580+nys), (685, 177+nye)), 
          ((153, 580+nys), (710, 200+nye))],
    "E": [((298-300, 155-55+nys), (1520-300, 831-55+nye)), 
          ((342-300, 121-55+nys), (1550-300, 770-55+nye)), 
          ((386-300, 101-55+nys), (1600-300, 737-55+nye)), 
          ((425-300, 70-55+nys), (1645-300, 680-55+nye)), 
          ((480-300, 44-55+nys), (1700-300, 641-55+nye)),
          ((525-300, 13-55+nys), (1740-300, 600-55+nye))],
    "S": [((1270, 190+2*nys), (560, 755+2*nye)), 
          ((1200, 153+2*nys), (465, 715+2*nye)), 
          ((1133, 140+2*nys), (400, 660+2*nye)), 
          ((1070, 107+2*nys), (335, 620+2*nye)), 
          ((1010, 70+2*nys), (275, 575+2*nye)),
          ((960, 42+2*nys), (235, 513+2*nye))],
    "W": [((364-70, 470+nye-5), (0, 251+2*nys-15)), 
          ((415-70, 445+nye-5), (0, 205+2*nys-15)), 
          ((470-70, 420+nye-5), (0, 160+2*nys-15)), 
          ((520-70, 396+nye-5), (0, 115+2*nys-15)),
          ((563-70, 371+nye-5), (0, 70+2*nys-15)),
          ((606-70, 347+nye-5), (0, 25+2*nys-15))]
}

def get_all_line_coefficients(lines):
    coefficients = {}
    for quadrant, line_pairs in lines.items():
        coefficients[quadrant] = [line_coefficients(p1, p2) for p1, p2 in line_pairs]
    return coefficients

def line_coefficients(p1, p2):
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = A * p1[0] + B * p1[1]
    return A, B, -C

coefficients = get_all_line_coefficients(lines)
quadrant = input("Which quadrant would you like to track? (N, E, S, W): ").upper()

def draw_lines_on_quadrant(quadrant_frame, quadrant):
    line_color = (0, 255, 0)  
    thickness = 2
    height, width, _ = quadrant_frame.shape
    for line in lines[quadrant]:
        cv2.line(quadrant_frame, line[0], line[1], line_color, thickness)
    return quadrant_frame

# Define lane boundaries using line coefficients for the selected quadrant
lane_boundaries = [(coefficients[quadrant][i], coefficients[quadrant][i + 1]) for i in range(len(coefficients[quadrant]) - 1)]
lane_counts = [0] * len(lane_boundaries)  # Initialize lane counts

def update_lane_counts(centers, lane_boundaries, lane_counts):
    for center in centers:
        lane_index = determine_lane(center, lane_boundaries)
        if lane_index is not None:
            lane_counts[lane_index] += 1
    return lane_counts

def determine_lane(center, lane_boundaries):
    for lane_index, (line1, line2) in enumerate(lane_boundaries):
        if is_between_lines(center, line1, line2):
            return lane_index
    return None

def is_between_lines(point, line1, line2):
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    eval1 = A1 * point[0] + B1 * point[1] + C1
    eval2 = A2 * point[0] + B2 * point[1] + C2
    return eval1 * eval2 <= 0

# Skip the first 500 frames
for _ in range(500):
    ret, _ = cap.read()
    if not ret:
        print("Failed to skip the first 500 frames.")
        cap.release()
        exit()


frame_count = 0
# Initialize a dictionary to keep track of which lane each vehicle ID has entered
vehicle_current_lane = {}

def update_lane_counts(centers, track_ids, lane_boundaries, lane_counts, vehicle_current_lane):
    for center, track_id in zip(centers, track_ids):
        new_lane_index = determine_lane(center, lane_boundaries)
        if new_lane_index is not None:
            current_lane_index = vehicle_current_lane.get(track_id)
            # Check if this vehicle has switched to a new lane
            if current_lane_index is None or current_lane_index != new_lane_index:
                # Update lane count only if it's a new lane for this vehicle
                if current_lane_index is not None:
                    # Decrease the count from the previous lane if necessary
                    lane_counts[current_lane_index] -= 1
                # Increase the count in the new lane
                lane_counts[new_lane_index] += 1
                # Update the vehicle's current lane
                vehicle_current_lane[track_id] = new_lane_index
    return lane_counts

# Inside the main loop where you process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    dx, dy = transformations[frame_count // 2][1:]
    frame = apply_transformation(frame, dx, dy, width, height)
    quadrant_frame = split_frame(frame, quadrant)
    results = model.track(quadrant_frame, persist=True)
    
    # Extract centers and track IDs
    centers = []
    track_ids = []
    # Iterate over the detected objects in the frame if there are any
    if results[0].boxes.id is not None:
        for det in results[0].boxes:
            x1, y1, x2, y2 = det.xyxy[0].int().tolist()
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            track_id = det.id.item()
            centers.append((center_x, center_y))
            track_ids.append(int(track_id))
    
    lane_counts = update_lane_counts(centers, track_ids, lane_boundaries, lane_counts, vehicle_current_lane)
    annotated_frame = draw_lines_on_quadrant(results[0].plot(), quadrant)  # Ensure you are drawing on the correct image
    cv2.imshow("Processed Frame", annotated_frame)
    print("Lane counts:", lane_counts)
    #print the current frame count
    print("Frame count:", frame_count)
    if frame_count == 4900:
        #turn the lines into the new lines
        lines = new_lines
        coefficients = get_all_line_coefficients(lines)
        lane_boundaries = [(coefficients[quadrant][i], coefficients[quadrant][i + 1]) for i in range(len(coefficients[quadrant]) - 1)]

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

print("Final lane counts:", lane_counts)
cap.release()
cv2.destroyAllWindows()
