

class Event:

	def __init__(self, label=None, half=None, time=None, team=None, position=None, visibility=None, frame_index=None):

		self.label = label
		self.half = half
		self.time = time
		self.team = team
		self.position = position
		self.visibility = visibility
		self.frame_index = frame_index

	def to_text(self):
		return self.time + " || " + self.label + " - " + self.team  + " - " + str(self.half) + " - " + str(self.visibility)

	def to_display_text(self, list_index: int) -> str:
		frame = self.frame_index if self.frame_index is not None else "?"
		return (
			f"[{list_index}] frame {frame} | pos {self.position} | "
			f"{self.label} - {self.team} - {self.visibility}"
		)

	def __lt__(self, other):
		self.position < other.position

def ms_to_time(position):
	minutes = int(position//1000)//60
	seconds = int(position//1000)%60
	return str(minutes).zfill(2) + ":" + str(seconds).zfill(2)