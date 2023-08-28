from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
from utils.video_processing import load_video, setup_yolo, get_bird_eye_view
from detect import detect_people_yolo
from tracker import track_people
from config import VIDEOS_DIR, TRACKED_VIDEOS, HOMOGRAPHY_DIR

FRAME_INTERVAL = 10
WAIT_KEY_INTERVAL = 25

def load_homography_from_txt(filename: Path) -> list:
    """Load homography values from a text file."""
    with filename.open('r') as file:
        return [float(val) for val in file.readline().strip().split(',')]

def main():
    # Load video
    video_capture = load_video(str(VIDEOS_DIR / "001.avi"))

    if not video_capture.isOpened():
        print("Ошибка: не удалось открыть видео.")
        return

    # Define video frame dimensions
    frame_dims = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Set up output video
    out = cv2.VideoWriter(str(TRACKED_VIDEOS / "tracked_video.avi"), cv2.VideoWriter_fourcc(*'XVID'), 20.0, frame_dims)

    trackers, tracked_paths, frame_count = [], defaultdict(list), 0

    # Set up YOLO
    net, output_layers = setup_yolo()

    # Load homography matrix
    homography_values = load_homography_from_txt(HOMOGRAPHY_DIR / "001.txt")
    homography_matrix = np.array(homography_values).reshape(3, 3)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("Ошибка: не удалось прочитать кадр.")
            break

        frame_count += 1
        # Reset trackers at interval
        if frame_count % FRAME_INTERVAL == 0:
            trackers.clear()

        # Detect people in the frame
        people = detect_people_yolo(frame, net, output_layers)

        # Initialize trackers on the first frame with detected people
        if not trackers and people:
            trackers = [cv2.TrackerCSRT_create() for _ in people]
            for tracker, person in zip(trackers, people):
                tracker.init(frame, tuple(person))

        # Update trackers and display paths
        frame, trackers, tracked_paths = track_people(frame, trackers, tracked_paths)

        # Write frame to the output video
        out.write(frame)

        # Convert the frame to bird-eye view
        bird_eye_frame = get_bird_eye_view(frame, homography_matrix)

        # Display tracked video in a window
        cv2.imshow('Tracked Video', frame)

        if cv2.waitKey(WAIT_KEY_INTERVAL) & 0xFF == ord('q'):
            break

    out.release()
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
