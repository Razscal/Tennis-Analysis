import pandas as pd
from sympy.abc import lamda

from utils import measure_distance, draw_player_stats
from mini_court import MiniCourt
from copy import deepcopy

class SpeedAnalysis:
	def __init__(self, minicourt : MiniCourt, video_frames):
		self.minicourt = minicourt
		self.video_frames = video_frames

	player_stats_data = [
		{
			"frame_num" : 0,
			"player_1_number_of_shots" : 0,
			"player_1_total_shot_speed" : 0,
			"player_1_last_shot_speed" : 0,
			"player_1_total_player_speed" : 0,
			"player_1_last_player_speed" : 0,

			"player_2_number_of_shots" : 0,
			"player_2_total_shot_speed" : 0,
			"player_2_last_shot_speed" : 0,
			"player_2_total_player_speed" : 0,
			"player_2_last_player_speed" : 0
		}
	]

	def calculate_speed(self, ball_shot_frames, ball_minicourt_detections, player_mini_court_detections):
		for ball_shot_idx in range(len(ball_shot_frames) - 1):
			print(ball_shot_idx)
			start_frame = ball_shot_frames[ball_shot_idx]
			end_frame = ball_shot_frames[ball_shot_idx + 1]
			ball_shot_time_in_seconds = (end_frame - start_frame) / 24 # 24 fps

			# Get distance covered by the ball
			distance_covered_by_ball_in_pixels = measure_distance(ball_minicourt_detections[start_frame][1], ball_minicourt_detections[end_frame][1])
			distance_covered_by_ball_in_meters = self.minicourt.convert_pixels_to_meters(distance_covered_by_ball_in_pixels)

			# Speed of the ball
			speed_of_ball = distance_covered_by_ball_in_meters / ball_shot_time_in_seconds * 3.6

			# Player who shot the ball
			player_position = player_mini_court_detections[start_frame]
			player_shot_ball = min(player_position.keys(), key=lambda player_id: measure_distance(player_position[player_id], ball_minicourt_detections[start_frame][1]))
			print(player_shot_ball)

			player_opponent_player_id = 1 if player_shot_ball == 2 else 2
			distance_covered_by_opponent_player_pixels = measure_distance(player_mini_court_detections[start_frame][player_opponent_player_id], player_mini_court_detections[end_frame][player_opponent_player_id])
			distance_covered_by_opponent_player_meters = self.minicourt.convert_pixels_to_meters(distance_covered_by_opponent_player_pixels)

			speed_of_opponents = distance_covered_by_opponent_player_meters / ball_shot_time_in_seconds * 3.6

			current_player_stats = deepcopy(self.player_stats_data[-1])

			current_player_stats['frame_num'] = start_frame
			current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
			current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball
			current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball

			current_player_stats[f'player_{player_opponent_player_id}_total_player_speed'] += speed_of_opponents
			current_player_stats[f'player_{player_opponent_player_id}_last_player_speed'] = speed_of_opponents

			self.player_stats_data.append(current_player_stats)

		# Create a dataframe
		print(self.player_stats_data)
		player_stats_data_df = pd.DataFrame(self.player_stats_data)
		frame_df = pd.DataFrame({'frame_num' : list(range(len(self.video_frames) - 1))})
		player_stats_data_df = pd.merge(frame_df, player_stats_data_df, on='frame_num', how='left')
		player_stats_data_df = player_stats_data_df.ffill()

		player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']
		player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']
		player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_1_number_of_shots']
		player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_2_number_of_shots']

		return player_stats_data_df


