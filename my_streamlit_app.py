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
import os, urllib, cv2

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

	# Download external dependencies.
	for filename in EXTERNAL_DEPENDENCIES.keys():
		download_file(filename)

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
	# To make Streamlit fast, st.cache allows us to reuse computation across runs.
	# In this common pattern, we download data from an endpoint only once.
	@st.cache
	def load_metadata(url):
		return pd.read_csv(url)

	# This function uses some Pandas magic to summarize the metadata Dataframe.
	@st.cache
	def create_summary(metadata):
		one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]], columns=["label"])
		summary = one_hot_encoded.groupby(["frame"]).sum().rename(columns={
			"label_biker": "biker",
			"label_car": "car",
			"label_pedestrian": "pedestrian",
			"label_trafficLight": "traffic light",
			"label_truck": "truck"
		})
		return summary

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

	# Update system infomation
	def update_sys_info(metrics):
		metrics.empty()
		col1, col2, col3 = metrics.columns(3)
		col1.metric("Temperature", "70 °F", "1.2 °F")
		col2.metric("Wind", "9 mph", "-8%")
		col3.metric("Humidity", "86%", "4%")

	if "rerun" not in st.session_state:
	# set the initial default value of the slider widget
		st.session_state.rerun = True

	if "slider" not in st.session_state:
	# set the initial default value of the slider widget
		st.session_state.slider = 60

	if "stop" not in st.session_state:
		st.session_state.stop = False

	if "frames" not in st.session_state:
		st.session_state.frames = frames

	# An amazing property of st.cached functions is that you can pipe them into
	# one another to form a computation DAG (directed acyclic graph). Streamlit
	# recomputes only whatever subset is required to get the right answer!
	metadata = load_metadata(os.path.join(DATA_URL_ROOT, "labels.csv.gz"))
	summary = create_summary(metadata)

	cap = cv2.VideoCapture(video_path)

	st.subheader("System Information")
	st.markdown("System info monitoring.")
	metrics = st.empty()
	col1, col2, col3 = metrics.columns(3)
	col1.metric("Temperature", "70 °F", "1.2 °F")
	col2.metric("Wind", "9 mph", "-8%")
	col3.metric("Humidity", "86%", "4%")
	st.subheader("Road Image")
	st.markdown("Real-time car-detection with road conditions under recording.")
	placeholder = st.empty()
	st.sidebar.markdown("# Frame")
	side_bar = st.sidebar.empty()
	selected_frame_index = 0

	if st.session_state.rerun:
		my_bar = side_bar.progress(0)
		if (cap.isOpened()):
			for i in range(61):
				my_bar.progress(i)
				ret, frame = cap.read() # (720, 1280, 3) mp4文件读进来是bgr
				save_frame(frame)
				placeholder.image(frame, caption=None, width=None, use_column_width=None, clamp=False, channels="BGR", output_format="auto")
		my_bar.empty()

	# Avoid rerunning everything
	st.session_state.rerun = False

	selected_frame_index = side_bar.slider("Choose a frame (index)",
											0, len(st.session_state.frames)-1, 
											key="slider",
											on_change=select_frame,
											)

	chart_data = pd.DataFrame(
    	np.random.randint(0, 10, (60,3)),
    	columns=['car', 'person', 'bike'])

	st.subheader("Statistics")
	st.markdown("Something may useful.")
	st.line_chart(chart_data, width=260, height=250)

	if "cap" not in st.session_state:
	# Store the cap accross reruns
		st.session_state.cap = cap

	if (st.session_state.cap.isOpened()):
		while(True):
			if not st.session_state.stop:
				ret, frame = st.session_state.cap.read() # (720, 1280, 3) mp4文件读进来是bgr
				save_frame(frame)
				placeholder.image(frame, caption=None, width=None, use_column_width=None, clamp=False, channels="BGR", output_format="auto")
			else:
				placeholder.image(st.session_state.frames[st.session_state.slider], 
									caption=None, width=None, 
									use_column_width=None, clamp=False, 
									channels="BGR", output_format="auto")	


	# print(stop)
	# while (cap.isOpened()):
	# 	if not stop:
	# 		ret, frame = cap.read() # (720, 1280, 3) mp4文件读进来是bgr
	# 		save_frame(frame)
	# 	if 2 <= len(frames) <= 61 and not is_61:
	# 		print(len(frames))
	# 		selected_frame_index = side_bar.slider("Choose a frame (index)",
	# 										0, len(frames)-1, len(frames)-1, 
	# 										key=len(frames),
	# 										)
	# 		print(selected_frame_index)
	# 		print("rerun here")
	# 		if selected_frame_index != len(frames) - 1:
	# 			print("good")
	# 			stop = 1
	# 			print(stop)
	# 	if len(frames) == 61: # Stop generate more sliders
	# 		is_61 = True
	# 	placeholder.image(frames[selected_frame_index], caption=None, width=None, use_column_width=None, clamp=False, channels="BGR", output_format="auto")

# This sidebar UI is a little search engine to find certain object types.
def frame_selector_ui(summary, side_bar, key):


	# # The user can pick which type of object to search for.
	# object_type = st.sidebar.selectbox("Search for which objects?", summary.columns, 2)

	# # The user can select a range for how many of the selected objecgt should be present.
	# min_elts, max_elts = st.sidebar.slider("How many %ss (select a range)?" % object_type, 0, 25, [10, 20])
	# selected_frames = get_selected_frames(summary, object_type, min_elts, max_elts)
	# if len(selected_frames) < 1:
	# 	return None, None

	# # Choose a frame out of the selected frames.
	# selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(selected_frames) - 1, 0)

	# # Draw an altair chart in the sidebar with information on the frame.
	objects_per_frame = summary.loc[selected_frames, object_type].reset_index(drop=True).reset_index()
	chart = alt.Chart(objects_per_frame, height=120).mark_area().encode(
		alt.X("index:Q", scale=alt.Scale(nice=False)),
		alt.Y("%s:Q" % object_type))
	selected_frame_df = pd.DataFrame({"selected_frame": [selected_frame_index]})
	vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(x = "selected_frame")
	st.sidebar.altair_chart(alt.layer(chart, vline))

	# selected_frame = selected_frames[selected_frame_index]

	selected_frame_index = side_bar.slider("Choose a frame (index)",
											0, max(len(frames)-1, 1), max(len(frames)-1, 1), 
											key=key)

	return selected_frame_index

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
