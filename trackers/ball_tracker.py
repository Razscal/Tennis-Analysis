import cv2
from ultralytics import YOLO
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        # print(df_ball_positions)
        df_ball_positions = df_ball_positions.bfill()
        # print(df_ball_positions)

        # convert back to list
        ball_positions = [{1 : x} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1,
                                                                                     center=False).mean()
        df_ball_positions['ball_hit'] = 0
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        minimum_change_frames_for_hit = 25
        threshold = int(minimum_change_frames_for_hit * 1.2)

        for i in range(len(df_ball_positions) - threshold):
            delta_y = df_ball_positions['delta_y'].iloc[i]
            next_delta_y = df_ball_positions['delta_y'].iloc[i + 1]

            negative_position_change = delta_y > 0 and next_delta_y < 0
            positive_position_change = delta_y < 0 and next_delta_y > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + threshold):
                    following_delta_y = df_ball_positions['delta_y'].iloc[change_frame + 1]

                    negative_position_change_following_frame = delta_y > 0 and following_delta_y < 0
                    positive_position_change_following_frame = delta_y < 0 and following_delta_y > 0

                    if negative_position_change_following_frame or positive_position_change_following_frame:
                        change_count += 1

                if change_count >= minimum_change_frames_for_hit:
                    df_ball_positions.at[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()
        return frame_nums_with_ball_hits

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as pickle_file:
                ball_detections = pickle.load(pickle_file)
                return ball_detections
        else:
            ball_detections = [self.detect_frame(frame) for frame in frames]
            if stub_path is not None:
                with open(stub_path, 'wb') as pickle_file:
                    pickle.dump(ball_detections, pickle_file)
            return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict([frame], conf=0.15)[0]
        ball_dict = {}
        for box in results.boxes:
            # print(box)
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        return ball_dict

    def draw_boxes(self, frames, ball_detections):
        output_frames = []

        for frame, ball_dict in zip(frames, ball_detections):
            for track_id, box in ball_dict.items():
                x1, y1, x2, y2 = map(int, box)
                cv2.putText(frame, f"Ball ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 255, 250), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 255, 250), 2)
            output_frames.append(frame)
        return output_frames