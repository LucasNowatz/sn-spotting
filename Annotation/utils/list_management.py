from utils.event_class import Event
import json
import os

class ListManager:

	def __init__(self):

		self.event_list = list()

	def create_list_from_json(self, path, half):

		self.event_list.clear()
		self.event_list = self.read_json(path, half)
		self.sort_list()

	def create_text_list(self):

		list_text = list()
		for event in self.event_list:
			list_text.append(event.to_text())

		return list_text

	def delete_event(self, index):

		self.event_list.pop(index)
		self.sort_list()

	def add_event(self, event):

		self.event_list.append(event)
		self.sort_list()

	def sort_list(self):

		self.event_list = sorted(self.event_list, key=lambda event: int(event.position))

	def soccerNetToV2(self,label):

		if label == "soccer-ball" or label == "soccer-ball-own":
			return "Goal"
		if label == "r-card":
			return "Red card"
		if label == "y-card":
			return "Yellow card"
		if label == "yr-card":
			return "Yellow->red card"
		if label == "substitution-in":
			return "Substitution"
		return "Other"

	def read_json(self, path, half):

		event_list = list()
		per_clip_ground_truth = os.path.basename(path) == "ground_truth.json"
		with open(path) as file:
			data = json.load(file)["annotations"]

			for event in data:
				tmp_half = int(event["gameTime"][0])
				if not per_clip_ground_truth and tmp_half != half:
					continue
				tmp_time = event["gameTime"][4:]
				tmp_position = 0
				if "position" in event:
					tmp_position = int(event["position"])
				else:
					tmp_position = int((int(tmp_time[0:2])*60 + int(tmp_time[3:]))*1000)
				tmp_label = None
				if os.path.basename(path) == "Labels.json":
					tmp_label = self.soccerNetToV2(event["label"])
				else:
					tmp_label = event["label"]
				tmp_team = event["team"]
				tmp_visibility = "default"
				if "visibility" in event:
					tmp_visibility = event["visibility"]
				tmp_frame_index = None
				if "frame_index" in event:
					tmp_frame_index = int(event["frame_index"])
				event_list.append(
					Event(
						tmp_label,
						tmp_half,
						tmp_time,
						tmp_team,
						tmp_position,
						tmp_visibility,
						tmp_frame_index,
					)
				)
		return event_list

	def _annotation_dict(self, event, original=None):
		tmp_dict = dict(original) if original else {}
		tmp_dict["gameTime"] = str(event.half) + " - " + str(event.time)
		tmp_dict["label"] = str(event.label)
		tmp_dict["team"] = str(event.team)
		tmp_dict["visibility"] = str(event.visibility)
		tmp_dict["position"] = str(event.position)
		return tmp_dict

	def save_file(self, path, half):

		if os.path.basename(path) == "ground_truth.json":
			final_list = self.event_list
		elif half == 1:
			list_other_half = self.read_json(path,2)
			final_list = self.event_list + list_other_half
		else:
			list_other_half = self.read_json(path,1)
			final_list = list_other_half + self.event_list

		with open(path, 'r') as original_file:
			data = json.load(original_file)

		fps = float(data.get("fps_assumed", 25.0))
		original_by_position = {}
		for ann in data.get("annotations", []):
			key = str(ann.get("position", ""))
			if key:
				original_by_position[key] = ann

		annotations_dictionary = list()
		for event in final_list:
			key = str(event.position)
			original = original_by_position.get(key)
			tmp_dict = self._annotation_dict(event, original)
			if "frame_index" not in tmp_dict and fps:
				tmp_dict["frame_index"] = int(round(int(event.position) / 1000.0 * fps))
			if "body_part" not in tmp_dict:
				tmp_dict["body_part"] = "not_applicable"
			annotations_dictionary.append(tmp_dict)

		data["annotations"] = annotations_dictionary

		if os.path.basename(path) == "ground_truth.json":
			path_to_save = path
		else:
			path_to_save = os.path.join(os.path.dirname(path), "Labels-v2.json")
		with open(path_to_save, "w") as save_file:
			json.dump(data, save_file, indent=2, sort_keys=False)
			save_file.write("\n")
