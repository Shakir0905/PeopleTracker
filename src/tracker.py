import cv2

MAX_POINTS = 100


def track_people(frame, trackers, tracked_paths):
    for idx, tracker in enumerate(trackers):
        ok, bbox = tracker.update(frame)
        if ok:
            center_bottom = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))
            cv2.circle(frame, center_bottom, 3, (0, 0, 255), -1)
            
            tracked_paths[idx].append(center_bottom)
            
            while len(tracked_paths[idx]) > MAX_POINTS:
                tracked_paths[idx].pop(0)
            
            for i in range(1, len(tracked_paths[idx])):
                cv2.line(frame, tracked_paths[idx][i-1], tracked_paths[idx][i], (0, 0, 255), 2)
                
            cv2.line(frame, (int(bbox[0]), int(bbox[1])), center_bottom, (0, 255, 255), 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)

    return frame, trackers, tracked_paths

