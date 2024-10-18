import cv2
from cv2 import VideoWriter_fourcc


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frame, output_video_path):
    fourcc = VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24,
                          (output_video_frame[0].shape[1], output_video_frame[0].shape[0]))
    for frame in output_video_frame:
        out.write(frame)
    out.release()
