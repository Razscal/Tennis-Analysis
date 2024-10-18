import cv2
from sympy.physics.units import length
from ultralytics import YOLO
import pickle
import warnings
from utils import get_center_of_box, measure_distance

warnings.filterwarnings("ignore")

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def filter_players(self, court_key_points, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(court_key_points, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: box for track_id, box in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        # print(filtered_player_detections)
        return filtered_player_detections

    def choose_players(self, court_key_points, player_dict):
        distances = []
        for track_id , box in player_dict.items():
            player_center = get_center_of_box(box)
            min_distance = float('inf')
            for i in range(0, len(court_key_points), 2):
                court_keypoint = (court_key_points[i], court_key_points[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        # print(distances)
        # sort the distances
        distances.sort(key=lambda x: x[1])

        # choose the first 2 tracks
        choosen_players = [distances[0][0], distances[1][0]]
        # print(choosen_players)
        return choosen_players


    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as pickle_file:
                player_detections = pickle.load(pickle_file)
                return player_detections
        else:
            player_detections = [self.detect_frame(frame) for frame in frames]
            if stub_path is not None:
                with open(stub_path, 'wb') as pickle_file:
                    pickle.dump(player_detections, pickle_file)
            return player_detections

    def detect_frame(self, frame):
        results = self.model.track([frame], persist=True, conf=0.2)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            # print(box)
            try:
                track_id = int(box.id.tolist()[0])
                result = box.xyxy.tolist()[0]

                #Mapping object class name {0 : 'person', 1 : 'ball'} to current detected class in image {cls : ([0, .]), xyxy : ([0, 0, 0, 0])}
                object_cls_id = box.cls.tolist()[0]
                object_cls_name = id_name_dict[object_cls_id]

                if object_cls_name.lower() == 'person':
                    player_dict[track_id] = result
            except Exception as e:
                continue

        return player_dict

    def draw_boxes(self, frames, ball_detections):
        output_frames = []

        for frame, ball_dict in zip(frames, ball_detections):
            for track_id, box in ball_dict.items():
                x1, y1, x2, y2 = map(int, box)
                cv2.putText(frame, f"Tennis Player ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            output_frames.append(frame)
        return output_frames