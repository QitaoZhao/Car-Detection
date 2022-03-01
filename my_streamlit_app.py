# -*- coding: utf-8 -*-
# Author: Qitao Zhao
# More info: https://github.com/streamlit/demo-self-driving

import cv2
import psutil
import pyowm
import urllib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts
from utils import get_loc

video_path = "test_2.mp4"
frames = []
result = {'detection_classes': ['car', 'person'], 'detection_scores': [0.95, 0.98]}


# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
	st.set_page_config(
		page_title="Ex-stream-ly Cool App",
		page_icon="🧊",
		layout="wide",
		initial_sidebar_state="collapsed",  # "auto" or "expanded" or "collapsed"
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


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
	def show_weather_data():
		place = "Qingdao"
		unit_c = 'celsius'
		weather = st.session_state.weather
		temperature = weather.temperature(unit='celsius')['temp']

		GMT_FORMAT = '%Y-%m-%d %H:%M:%S+00:00'
		st.sidebar.title(f"📍 Weather at {place[0].upper() + place[1:]} currently: ")
		st.sidebar.write(f"### 🌡️ Temperature: {temperature} °C")
		st.sidebar.write(f"### ☁️ Sky: {weather.detailed_status[0].upper() + weather.detailed_status[1:]}")
		st.sidebar.write(f"### 🌪 Wind Speed: {round(weather.wind(unit='km_hour')['speed'])} km/h")
		st.sidebar.write(
			f"### 🌅 Sunrise Time :     {datetime.strptime(weather.sunrise_time(timeformat='iso'), GMT_FORMAT) + timedelta(hours=8)}")
		st.sidebar.write(
			f"### 🌇 Sunset Time :      {datetime.strptime(weather.sunset_time(timeformat='iso'), GMT_FORMAT) + timedelta(hours=8)}")

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

	# Fetch system information
	def get_sys_info(sys_info):
		cpu_percent = psutil.cpu_percent(1)
		sys_info.append(cpu_percent)
		disk = psutil.disk_usage('/')
		disk_usage = disk.used / disk.total
		sys_info.append(round(100 * disk_usage, 1))
		mem = psutil.virtual_memory()
		mem_usage = mem.used / mem.total
		sys_info.append(round(100 * mem_usage, 1))

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
					"radius": '100%',
					"itemStyle": {
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
			}
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
		road_map.image(img, caption=None, width=None, use_column_width=True, clamp=False, channels="RGB",
					   output_format="auto")

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

	if "result_return" not in st.session_state:
		# Store returned result placeholder
		st.session_state.result_return = None

	if "stop" not in st.session_state:
		st.session_state.stop = False

	if "frames" not in st.session_state:
		st.session_state.frames = frames

	st.subheader("System Information Monitoring 🚗")
	gauges = st.empty()
	col1, col2, col3 = gauges.columns(3)
	st.subheader("Road Map with Real-time Car-detection 🚦")
	col_1, col_2 = st.columns([5, 5])

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
			st.session_state.sys_info.append(round(100 * disk_usage, 1))
			st.session_state.sys_info.append(round(100 * mem_usage, 1))

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

	draw_gauges([col1, col2, col3])
	draw_map(st.session_state.road_map, st.session_state.my_loc)
	show_weather_data()
	st.sidebar.markdown("# Replay the video")
	side_bar = st.sidebar.empty()
	st.subheader("Statistics 📈")
	st.markdown("Detection result.")

	if st.session_state.first_time:
		st.session_state.result_return = st.sidebar.empty()
		my_bar = side_bar.progress(0)
		cap = cv2.VideoCapture(video_path)
		if (cap.isOpened()):
			for i in range(61):
				my_bar.progress(i)
				ret, frame = cap.read()  # (720, 1280, 3) mp4文件读进来是bgr
				frame = cv2.resize(frame, dsize=(960, 540))
				save_frame(frame)
				st.session_state.road_image.image(frame, caption=None, width=None, use_column_width=True, clamp=True,
												  channels="BGR", output_format="auto")
				st.session_state.result_return.write(result)
		my_bar.empty()

	selected_frame_index = side_bar.slider("Choose a frame (index)",
										   0, len(st.session_state.frames) - 1,
										   key="slider",
										   on_change=select_frame,
										   )

	chart_data = pd.DataFrame(
		np.random.randint(0, 10, (60, 3)),
		columns=['car', 'person', 'bike'])

	# st.subheader("Statistics 📈")
	# st.markdown("Something may useful.")
	st.line_chart(chart_data, width=260, height=250)
	st.write("😆😆 Enjoy yourself!!!")
	c = st.empty()

	# Avoid re-running everything
	st.session_state.first_time = False

	if "cap" not in st.session_state:
		# Store the cap across reruns
		st.session_state.cap = cap

	if (st.session_state.cap.isOpened()):
		while (True):
			if not st.session_state.stop:
				ret, frame = st.session_state.cap.read()  # (720, 1280, 3) mp4文件读进来是bgr
				try:
					frame = cv2.resize(frame, dsize=(960, 540))
				except cv2.error:
					st.stop()
				else:
					pass
				save_frame(frame)
				st.session_state.road_image.image(frame, caption=None, width=None, use_column_width=True, clamp=False,
												  channels="BGR", output_format="auto")
				st.session_state.result_return.write(result)
			else:
				st.session_state.road_image.image(st.session_state.frames[st.session_state.slider],
												  caption=None, width=None,
												  use_column_width=True, clamp=False,
												  channels="BGR", output_format="auto")
				st.session_state.result_return.write(result)
				st.stop()


# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
	url = 'https://raw.githubusercontent.com/QitaoZhao/Car-Detection/my-branch/' + path  # Use raw content
	response = urllib.request.urlopen(url)
	return response.read().decode("utf-8")


if __name__ == "__main__":
	main()