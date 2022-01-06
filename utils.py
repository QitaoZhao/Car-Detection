from urllib.request import urlopen
import requests
import IPy
import geocoder
import cv2
import numpy as np
from PIL import Image

def get_location(ip):
	url =  'https://sp0.baidu.com/8aQDcjqpAAV3otqbppnN2DJv/api.php?co=&resource_id=6006&t=1529895387942&ie=utf8&oe=gbk&cb=op_aladdin_callback&format=json&tn=baidu&cb=jQuery110203920624944751099_1529894588086&_=1529894588088&query=%s'%ip
	r = requests.get(url)
	r.encoding = r.apparent_encoding
	html = r.text
	c1 = html.split('location":"')[1]
	c2 = c1.split('","')[0]
	return c2

def check_ip(ip):
	try:
		IPy.IP(ip)
		return True
	except Exception as e:
		print(e)
		return False

def get_loc():
	my_ip = urlopen('http://ip.42.pl/raw').read()
	ip = str(my_ip).strip('b')
	ip = eval(ip)
	# print(ip)
	# if check_ip(ip):
	# 	print('IP位置为:',get_location(ip))

	# g = geocoder.google("1403 Washington Ave, New Orleans, LA 70130")
	# g = geocoder.arcgis(u"山东省青岛市即墨区滨海公路72号山东大学青岛校区")
	g = geocoder.arcgis(get_location(ip))
	return g.latlng

key = '793002c3bffa08e7b12ad453d71f226f' #高德地图API
#静态地图
def map(location, zoom=10):
	parameters = {
		'key':key,
		'location':location, #经纬度
		'zoom':zoom #缩放级别[1,17]
	}
	r = requests.get('https://restapi.amap.com/v3/staticmap?' + 
		'location=%.6f,%.6f'%(location[1],location[0]) +
		'&zoom=%d&size=960*540'%(zoom) +
		'&markers=mid,,A:%.6f,%.6f'%(location[1],location[0]) +
		'&key=793002c3bffa08e7b12ad453d71f226f')
	image = r.content
	with open("./map.jpg",'wb') as fp:
		fp.write(image)
	return image
# my_ip = urlopen('http://ip.42.pl/raw').read()
# ip = str(my_ip).strip('b')
# ip = eval(ip)
# print(ip)
# g = geocoder.arcgis(get_location(ip))
# print(g.latlng)
# # data = map(g.latlng, 15)
# data = map([36.362695,120.692479], 15)

# img = cv2.imread("./map.jpg")
# print(img.shape)
# cv2.imshow("map_image", img)
# cv2.waitKey(0)  # 无限期等待输入




