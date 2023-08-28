import cv2
from config import  YOLO_DIR 


def load_video(video_path):
    return cv2.VideoCapture(video_path)


def play_video(video_capture):
    """
    Воспроизводит видео.
    
    Args:
    - video_capture (cv2.VideoCapture): Объект для чтения видео.
    """
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def setup_yolo():
    # Настройка модели YOLO
    weights_path = YOLO_DIR / "yolov3-tiny.weights"
    config_path = YOLO_DIR / "yolov3-tiny.cfg"

    net = cv2.dnn.readNet(str(weights_path), str(config_path))
    layer_names = net.getLayerNames()
    output_layer_indexes = net.getUnconnectedOutLayers().flatten()
    output_layers = [layer_names[i - 1] for i in output_layer_indexes]

    return net, output_layers

def get_bird_eye_view(frame, homography_matrix):
    height, width, _ = frame.shape
    bird_eye_view = cv2.warpPerspective(frame, homography_matrix, (width, height))
    return bird_eye_view
