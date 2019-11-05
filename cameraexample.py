import picamera
import time
from PIL import Image
import numpy as np

#s = (2073600, 6)
#gauss = np.zeros(s)
#gauss[:,0] = 1
#xpos = np.arange(1080)
#tgauss = np.transpose(gauss)
#one = np.matmul(tgauss, gauss)
#two = np.lin.alg.inv(one)
#almost = np.matmul(two, tgauss)
#def gaussfit(jpgname):
	#jpg = Image.open('home/pi/')
	#np.array(jpg)
	#Yvec = np.reshape(avg.T, -1)
	#Yvec = np.log(Yvec)
	#coeff = np.matmul(almost, Yvec)
	#f = coeff[0]
	#d = coeff[1]
	#e = coeff[2]
	#a = coeff[3]
	#b = coeff[4]
	#c = coeff[5]
	#y0 = (e- ((c * d) / (2  * a))) / ((-2 * b) + (( c ** 2) / (2 * a)))
	#x0 = (d + (c * y0)) / (-2 *a)
	#return x0, y0

camera = picamera.PiCamera()
camera.resolution = (1024, 768)
#camera.CAPTURE_TIMEOUT = 60
camera.vflip = True
#camera.capture('bayerpls.jpg', format='jpeg', bayer=True)
#camera.start_recording('examplevid5.h264')
#time.sleep(5)
#camera.stop_recording
camera.capture_continuous(format='jpeg')
#ver = {
#	'RP_ov5647': 1,
#	'RP_imx219': 2,
#	}[camera.exif_tags['IFD0.Mode1']]
#offset = {
#	1: 6404096,
#	2: 10270208,
#	}[ver]
#data =  stream.getvalue()[-offset:]
#assert data[:4] == 'BRCM'
#data = data[32768:]
#data = np.fromstring(data, dtype=np.uint8)
#reshape, crop = {
#	1: ((1952, 3264), (1944, 3240)),
#	2: ((2480, 4128), (2464, 4100)),
#	}[ver]
#data = data.reshape(reshape)[:crop[0], :crop[1]]
#data = data.astype(np.uint16) << 2
#for byte in range(4):
#	data[:, byte::5] |= ((data[:, 4::5] >> ((4 - byte) * 2)) & 0b11)#data = np.delete(data, np.s_[4::5], 1)
#rgb = np.zeros(data.shape + (3,), dtype=data.dtype)
#rbg[1::2, 0::2, 0] = data[1::2, 0::2] #RED
#rbg[0::2, 0::2, 1] = data[0::2, 0::2] #GREEN
#rbg[1::2, 1::2, 1] = data[1::2, 1::2] #GREEN
#rbg[0::2, 1::2, 2] = data[0::2, 1::2] #BLUE

