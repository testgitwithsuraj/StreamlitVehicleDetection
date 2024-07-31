import cv2
import numpy as np
from sort import Sort

def process_video(video_path, resolution, frame_rate, detection_sensitivity, show_bounding_boxes, show_object_ids, show_tracking_paths, show_avg_speed, show_vehicle_density):
    cap = cv2.VideoCapture(video_path)
    
    # Adjust resolution
    if resolution == "480p":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    elif resolution == "720p":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    elif resolution == "1080p":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cap.set(cv2.CAP_PROP_FPS, frame_rate)
    
    min_height_rect = 80
    min_width_rect = 80
    count_line_position = 550

    # Initialize Subtractor
    algo = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    tracker = Sort()

    def center_handle(x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return int(cx), int(cy)

    counter = 0

    while True:
        ret, frame1 = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 5)

        # Apply background subtractor
        img_sub = algo.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((5, 5)), iterations=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

        detections = []

        for (i, c) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(c)
            validate_counter = (w >= min_width_rect) and (h >= min_height_rect)
            if not validate_counter:
                continue

            detections.append([x, y, x + w, y + h])

        tracked_objects = tracker.update(np.array(detections))

        for (x1, y1, x2, y2, object_id) in tracked_objects:
            center = center_handle(x1, y1, x2 - x1, y2 - y1)
            if show_bounding_boxes:
                cv2.rectangle(frame1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            if show_object_ids:
                cv2.putText(frame1, "Vehicle " + str(object_id), (int(x1), int(y1) - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)
            if show_tracking_paths:
                cv2.circle(frame1, center, 4, (0, 0, 255), -1)

            if count_line_position - 6 < center[1] < count_line_position + 6:
                counter += 1
                cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
                print("Vehicle Counter:" + str(counter))

        cv2.putText(frame1, "Vehicle Counter:" + str(counter), (450, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)

        # Mock calculations for average speed and vehicle density
        avg_speed = 30  # Placeholder value
        avg_density = 0.5  # Placeholder value

        yield frame1, counter, avg_speed, avg_density

    cap.release()
