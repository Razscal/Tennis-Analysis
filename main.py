import cv2
from jedi.debug import speed
from networkx.algorithms.bipartite.basic import color

from utils import read_video, save_video, draw_player_stats
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from speed_tracker import SpeedAnalysis


def main(input_video, read_from_stub = False):
	# Read video
	video_frames = read_video(input_video)
	# print(video_frames[0].shape)

	# Detect players
	player_tracker = PlayerTracker('yolov8x.pt')
	player_detections = player_tracker.detect_frames(video_frames,
	                                                 read_from_stub,
	                                                 stub_path='tracker_stubs/player_detections.pkl')
	# print(player_detections)

	# Detect ball
	ball_detections = ball_tracker.detect_frames(video_frames,
	                                             read_from_stub,
	                                             stub_path='tracker_stubs/ball_detections.pkl')
	print('------------Ball detections---------------')
	print(ball_detections)
	ball_detections = ball_tracker.interpolate_ball_position(ball_detections)
	print('------------Ball detections after interpolation---------------')
	print(ball_detections)

	# Detect court lines.
	court_line_detector = CourtLineDetector(court_model_path)
	court_key_points = court_line_detector.predict(video_frames[0])
	# print(court_key_points)

	# Choose players
	player_detections = player_tracker.filter_players(court_key_points, player_detections)

	# Mini court
	mini_court = MiniCourt(video_frames[0])

	# Detect ball shot frame
	ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
	print('------------Ball shot frames---------------')
	print(ball_shot_frames)
	# print('Ball shot frames' , ball_shot_frames)

	# Convert position to mini court position
	player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
		player_detections, ball_detections, court_key_points)
	print(ball_mini_court_detections)

	# Drawing
	output_frames = player_tracker.draw_boxes(video_frames, player_detections)
	output_frames = ball_tracker.draw_boxes(output_frames, ball_detections)
	output_frames = court_line_detector.draw_keypoints_on_video(output_frames, court_key_points)
	output_frames = mini_court.draw_mini_court(output_frames)
	output_frames = mini_court.draw_points_on_mini_court(output_frames, player_mini_court_detections, color= (0, 0, 255))
	output_frames = mini_court.draw_points_on_mini_court(output_frames, ball_mini_court_detections, color= (70,230, 62))

	# Speed analysis
	speed_analysis = SpeedAnalysis(mini_court, output_frames)
	player_stats_df = speed_analysis.calculate_speed(ball_shot_frames, ball_mini_court_detections, player_mini_court_detections)

	output_frames = draw_player_stats(output_frames, player_stats_df)

	# Draw frames
	for i, frame in enumerate(output_frames):
		cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 0, 250), 2)

	# Save video
	save_video(output_frames, 'outputs_videos/output_video.avi')


if __name__ == "__main__":
	main(input_video='input_video_train.mp4', read_from_stub=False)
