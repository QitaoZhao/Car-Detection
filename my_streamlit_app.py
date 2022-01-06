# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This demo lets you to explore the Udacity self-driving car image dataset.
# More info: https://github.com/streamlit/demo-self-driving

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_echarts import st_echarts
import os, urllib, cv2, psutil, time, pyowm
from utils import get_loc

video_path = "test.mp4"
frames = []

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
	st.set_page_config(
		 page_title="Ex-stream-ly Cool App",
		 page_icon="🧊",
		 layout="wide",
		 initial_sidebar_state="collapsed", # "auto" or "expanded" or "collapsed"
		 menu_items={
			 'Get Help': 'https://www.extremelycoolapp.com/help',
			 'Report a bug': "https://www.extremelycoolapp.com/bug",
			 'About': "# This is a header. This is an *extremely* cool app!"
		}
	)

	# Render the readme as markdown using st.markdown.
	readme_text = st.markdown(get_file_content_as_string("instructions.md"))

	# Once we have the dependencies, add a selector for the app mode on the sidebar.
	st.sidebar.title("What to do")
	app_mode = st.sidebar.selectbox("Choose the app mode",
		["Show instructions", "Run the app", "Show the source code"])
	if app_mode == "Show instructions":
		st.sidebar.success('To continue select "Run the app".')
	elif app_mode == "Show the source code":
		readme_text.empty()
		st.code(get_file_content_as_string("my_streamlit_app.py"))
	elif app_mode == "Run the app":
		readme_text.empty()
		run_the_app()

# This file downloader demonstrates Streamlit animation.
def download_file(file_path):
	# Don't download the file twice. (If possible, verify the download using the file length.)
	if os.path.exists(file_path):
		if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
			return
		elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
			return

	# These are handles to two visual elements to animate.
	weights_warning, progress_bar = None, None
	try:
		weights_warning = st.warning("Downloading %s..." % file_path)
		progress_bar = st.progress(0)
		with open(file_path, "wb") as output_file:
			with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
				length = int(response.info()["Content-Length"])
				counter = 0.0
				MEGABYTES = 2.0 ** 20.0
				while True:
					data = response.read(8192)
					if not data:
						break
					counter += len(data)
					output_file.write(data)

					# We perform animation by overwriting the elements.
					weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
						(file_path, counter / MEGABYTES, length / MEGABYTES))
					progress_bar.progress(min(counter / length, 1.0))

	# Finally, we remove these visual elements by calling .empty().
	finally:
		if weights_warning is not None:
			weights_warning.empty()
		if progress_bar is not None:
			progress_bar.empty()

# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():

	def show_weather_data():
		place = "Qingdao"
		unit_c = 'celsius'
		weather = st.session_state.weather
		temperature = weather.temperature(unit='celsius')['temp']

		GMT_FORMAT = '%Y-%m-%d %H:%M:%S+00:00'
		st.sidebar.title(f"📍 Weather at {place[0].upper()+place[1:]} currently: ")
		st.sidebar.write(f"### 🌡️ Temperature: {temperature} °C")
		st.sidebar.write(f"### ☁️ Sky: {weather.detailed_status[0].upper()+weather.detailed_status[1:]}")
		st.sidebar.write(f"### 🌪 Wind Speed: {round(weather.wind(unit='km_hour')['speed'])} km/h")
		st.sidebar.write(f"### 🌅 Sunrise Time :     {datetime.strptime(weather.sunrise_time(timeformat='iso'), GMT_FORMAT)+timedelta(hours=8)}")
		st.sidebar.write(f"### 🌇 Sunset Time :      {datetime.strptime(weather.sunset_time(timeformat='iso'), GMT_FORMAT)+timedelta(hours=8)}")

	# @st.experimental_memo # It may accelerate the process	
	def save_frame(frame):
		if len(st.session_state.frames) < 61:
			st.session_state.frames.append(frame)
		else:
			st.session_state.frames.pop(0)
			st.session_state.frames.append(frame)

	def select_frame():
		if st.session_state.slider == 60:
			st.session_state.stop = False
		else:
			st.session_state.stop = True

	# Fetch system infomation
	def get_sys_info(sys_info):
		cpu_percent = psutil.cpu_percent(1)
		sys_info.append(cpu_percent)
		disk = psutil.disk_usage('/')
		disk_usage = disk.used / disk.total 
		sys_info.append(round(100*disk_usage,1))
		mem = psutil.virtual_memory()
		mem_usage = mem.used / mem.total
		sys_info.append(round(100*mem_usage,1))

	def draw_gauges(columns):
		def show_gauge(value, label):
			option = {
				"tooltip": {
					"formatter": '{a} <br/>{b} : {c}%',
				},
				"series": [{
					"name": '进度',
					"type": 'gauge',
					"startAngle": 0,
					"endAngle": 360,
					"progress": {
						"show": "true"
					},
					"radius":'100%', 
					"itemStyle": {
						# "color": '#58D9F9',
						"color": '#D23D3B',
						"shadowColor": 'rgba(206,104,104,0.45)',
						"shadowBlur": 10,
						"shadowOffsetX": 2,
						"shadowOffsetY": 2,
						"radius": '55%',
					},
					"progress": {
						"show": "true",
						"roundCap": "true",
						"width": 15
					},
					"pointer": {
						"length": '60%',
						"width": 8,
						"offsetCenter": [0, '5%']
					},
					"detail": {
						"valueAnimation": "true",
						"formatter": '{value}%',
						"color": '#FAFAFA',
						"backgroundColor": '#262730',
						"fontSize": 20,
						"borderColor": '#999',
						"borderWidth": 4,
						"width": '60%',
						"lineHeight": 10,
						"height": 10,
						"bottom": "0%",
						"borderRadius": 188,
						"offsetCenter": [0, '40%'],
					},
					"data": [{
						"value": value,
						"name": label,
					}]
				}]
			};
			st_echarts(options=option)

		col1, col2, col3 = columns
		with col1:
			show_gauge(st.session_state.sys_info[0], "CPU Usage")
		with col2:
			show_gauge(st.session_state.sys_info[1], "Disk Usage")
		with col3:
			show_gauge(st.session_state.sys_info[2], "Memory Usage")

	def draw_map(road_map, my_loc):
		# lat, lon = my_loc
		img = cv2.imread("./map.jpg")
		road_map.image(img, caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")

	if "first_time" not in st.session_state:
	# set the initial default value of the slider widget
		st.session_state.first_time = True

	if "sys_info" not in st.session_state:
	# Store the system information
		st.session_state.sys_info = []

	if "slider" not in st.session_state:
	# Set the initial default value of the slider widget
		st.session_state.slider = 60

	if "my_loc" not in st.session_state:
	# Set the location
		st.session_state.my_loc = None

	if "weather" not in st.session_state:
	# Store weather conditions
		st.session_state.weather = None

	if "road_map" not in st.session_state:
	# Store road_map placeholder
		st.session_state.road_map = None

	if "road_image" not in st.session_state:
	# Store road_image placeholder
		st.session_state.road_image = None

	if "stop" not in st.session_state:
		st.session_state.stop = False

	if "frames" not in st.session_state:
		st.session_state.frames = frames

	st.subheader("System Information Monitoring 🚗")
	gauges = st.empty()
	col1, col2, col3 = gauges.columns(3)
	st.subheader("Road Map with Real-time Car-detection 🚦")
	col_1, col_2 = st.columns([5,5])

	if st.session_state.first_time:
		with st.spinner('Initialization...'):
			# Initialize placeholders
			st.session_state.road_map = col_1.empty()
			st.session_state.road_image = col_2.empty()

			# Fetch system info at first run
			cpu_percent = psutil.cpu_percent(1)
			disk = psutil.disk_usage('/')
			disk_usage = disk.used / disk.total 
			mem = psutil.virtual_memory()
			mem_usage = mem.used / mem.total
			st.session_state.sys_info.append(cpu_percent)
			st.session_state.sys_info.append(round(100*disk_usage,1))
			st.session_state.sys_info.append(round(100*mem_usage,1))

			# Fetch location
			# st.session_state.my_loc = get_loc()
			st.session_state.my_loc = None

			# Fetch weather info
			owm = pyowm.OWM('13396e2da2b93d0b4b2c526651854212')
			place = "Qingdao"
			mgr = owm.weather_manager()
			obs = mgr.weather_at_place(place)
			st.session_state.weather = obs.weather
		st.balloons()

	draw_gauges([col1,col2,col3])
	draw_map(st.session_state.road_map, st.session_state.my_loc)
	show_weather_data()
	st.sidebar.markdown("# Replay the video")
	side_bar = st.sidebar.empty()

	if st.session_state.first_time:
		my_bar = side_bar.progress(0)
		cap = cv2.VideoCapture(video_path)
		if (cap.isOpened()):
			for i in range(61):
				my_bar.progress(i)
				ret, frame = cap.read() # (720, 1280, 3) mp4文件读进来是bgr
				frame = cv2.resize(frame, dsize=(960, 540))
				save_frame(frame)
				st.session_state.road_image.image(frame, caption=None, width=None, use_column_width=True, clamp=True, channels="BGR", output_format="auto")
		my_bar.empty()

	selected_frame_index = side_bar.slider("Choose a frame (index)",
											0, len(st.session_state.frames)-1, 
											key="slider",
											on_change=select_frame,
											)

	chart_data = pd.DataFrame(
		np.random.randint(0, 10, (60,3)),
		columns=['car', 'person', 'bike'])

	st.subheader("Statistics 📈")
	st.markdown("Something may useful.")
	st.line_chart(chart_data, width=260, height=250)
	st.write("😆😆 Enjoy yourself!!!")
	c = st.empty()

	# for i in range(1, 30):
	#     time.sleep(0.5) #just here so you can see the change
	#     c.text(i)

	# Avoid re-running everything
	st.session_state.first_time = False

	if "cap" not in st.session_state:
	# Store the cap accross reruns
		st.session_state.cap = cap

	if (st.session_state.cap.isOpened()):
		while(True):
			if not st.session_state.stop:
				ret, frame = st.session_state.cap.read() # (720, 1280, 3) mp4文件读进来是bgr
				frame = cv2.resize(frame, dsize=(960, 540))
				save_frame(frame)
				st.session_state.road_image.image(frame, caption=None, width=None, use_column_width=True, clamp=False, channels="BGR", output_format="auto")
			else:
				st.session_state.road_image.image(st.session_state.frames[st.session_state.slider], 
									caption=None, width=None, 
									use_column_width=True, clamp=False, 
									channels="BGR", output_format="auto")	
				st.stop()		

# Select frames based on the selection in the sidebar
@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(summary, label, min_elts, max_elts):
	return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index

# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui():
	st.sidebar.markdown("# Model")
	confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
	overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
	return confidence_threshold, overlap_threshold

# Draws an image with boxes overlayed to indicate the presence of cars, pedestrians etc.
def draw_image_with_boxes(image, boxes, header, description):
	# Superpose the semi-transparent object detection boxes.    # Colors for the boxes
	LABEL_COLORS = {
		"car": [255, 0, 0],
		"pedestrian": [0, 255, 0],
		"truck": [0, 0, 255],
		"trafficLight": [255, 255, 0],
		"biker": [255, 0, 255],
	}
	image_with_boxes = image.astype(np.float64)
	for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
		image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS[label]
		image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2

	# Draw the header and image.
	st.subheader(header)
	st.markdown(description)
	st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
	url = 'https://raw.githubusercontent.com/QitaoZhao/Car-Detection/my-branch/' + path # Use raw content
	response = urllib.request.urlopen(url)
	return response.read().decode("utf-8")

# This function loads an image from Streamlit public repo on S3. We use st.cache on this
# function as well, so we can reuse the images across runs.
@st.cache(show_spinner=False)
def load_image(url):
	with urllib.request.urlopen(url) as response:
		image = np.asarray(bytearray(response.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	image = image[:, :, [2, 1, 0]] # BGR -> RGB
	return image

# Run the YOLO model to detect objects.
def yolo_v3(image, confidence_threshold, overlap_threshold):
	# Load the network. Because this is cached it will only happen once.
	@st.cache(allow_output_mutation=True)
	def load_network(config_path, weights_path):
		net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
		output_layer_names = net.getLayerNames()
		output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
		return net, output_layer_names
	net, output_layer_names = load_network("yolov3.cfg", "yolov3.weights")

	# Run the YOLO neural net.
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layer_outputs = net.forward(output_layer_names)

	# Supress detections in case of too low confidence or too much overlap.
	boxes, confidences, class_IDs = [], [], []
	H, W = image.shape[:2]
	for output in layer_outputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > confidence_threshold:
				box = detection[0:4] * np.array([W, H, W, H])
				centerX, centerY, width, height = box.astype("int")
				x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				class_IDs.append(classID)
	indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlap_threshold)

	# Map from YOLO labels to Udacity labels.
	UDACITY_LABELS = {
		0: 'pedestrian',
		1: 'biker',
		2: 'car',
		3: 'biker',
		5: 'truck',
		7: 'truck',
		9: 'trafficLight'
	}
	xmin, xmax, ymin, ymax, labels = [], [], [], [], []
	if len(indices) > 0:
		# loop over the indexes we are keeping
		for i in indices.flatten():
			label = UDACITY_LABELS.get(class_IDs[i], None)
			if label is None:
				continue

			# extract the bounding box coordinates
			x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

			xmin.append(x)
			ymin.append(y)
			xmax.append(x+w)
			ymax.append(y+h)
			labels.append(label)

	boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})
	return boxes[["xmin", "ymin", "xmax", "ymax", "labels"]]

# Path to the Streamlit public S3 bucket
DATA_URL_ROOT = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"

# External files to download.
EXTERNAL_DEPENDENCIES = {
	"yolov3.weights": {
		"url": "https://pjreddie.com/media/files/yolov3.weights",
		"size": 248007048
	},
	"yolov3.cfg": {
		"url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
		"size": 8342
	}
}

if __name__ == "__main__":
	main()
