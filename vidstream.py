from __future__ import print_function
from picamera import PiCamera
from time import sleep
import serial
import time
import io
import imutils
from analysistools import *
from scipy.signal import *
from scipy.ndimage.filters import *
from imutils.video import FPS
from picamera.array import PiRGBArray
import threading
from multiprocessing import Process
import argparse
import numpy as np
import matplotlib.pyplot as plt
import psutil, os, sys
import Queue
import datetime
import glob
from PIL import Image
import PiVideoStream as PVS
import picamera
from guppy import hpy
import scipy.signal as SIG
import timeit

start = time.time()

num_frames = 1
height = 768
width = 1024

yup = hpy()

w = 1024
h = 768
resolution = (w, h)
s = (w*h, 6)
gauss = np.zeros(s)
gauss[:,0] = 1
xpos = np.arange(w)
for k in range(h):
	gauss[(k * w) : (k + 1) * w, 1] = xpos
	gauss[(k * w) : (k + 1) * w, 2] = k
	gauss[(k * w) : (k + 1) * w, 3] = xpos ** 2
	gauss[(k * w) : (k + 1) * w, 4] = k ** 2
	gauss[(k * w) : (k + 1) * w, 5] = k * xpos
tgauss = np.transpose(gauss)
one = np.matmul(tgauss, gauss)
two = np.linalg.inv(one)
almost = np.matmul(two, tgauss)
center = []
xposarr = []
yposarr = []

#camera = picamera.PiCamera()
#camera.resolution = (1024, 768)

def gaussfit(vstream, fps0, args0):
	current_frame = 0
	index = 0
	#while index < args0["num_frames"]:
	while fps._numFrames < args0["num_frames"]:
		#print(fps0._numFrames)
		frame = vstream.queue.get()
		#print(np.nonzero(frame))
		if frame is not None:
			data = np.fromstring(frame, dtype=np.uint8).reshape((h, w, 3))
			print(data)
			blue = data[:,:,2]
			#avg = np.average(blue, 2)
			#avg = np.average(blue)
			Yvec = np.reshape(blue.T,-1)
			Yvec = np.log(Yvec + 1e-6)
			coeff = np.matmul(almost, Yvec)
			f = coeff[0]
			d = coeff[1]
			e = coeff[2]
			a = coeff[3]
			b = coeff[4]
			c = coeff[5]
			y0 = (e - ((c * d) / (2 * a))) / ((-2 * b) + ((c ** 2) / (2 * a)))
			x0 = (d + (c * y0)) / (-2 * a)
			xposarr.append(x0)
			yposarr.append(y0)
			values = np.array([x0, y0])
			#values = np.zeros(2)
			center.append(values)
			print(current_frame)
			index += 1
			current_frame += 1
			fps0.update()
	fps0.stop()
	vstream.stop()
	##print("{:.2f}".format(fps0.fps()))
	return x0, y0

#print(yup.heap())
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=num_frames,
	help="# of frames to loop over")
ap.add_argument("-d", "--display", type=int, default=-1)
pargs = vars(ap.parse_args())
fr = 2
vs = PVS.PiVideoStream(resolution=resolution, framerate=fr)
fps = FPS().start()
que = Queue.Queue()

#gaussfit('example6.jpg')
threading.Thread(target=lambda q, vs, fps, pargs: q.put(gaussfit(vs, fps, pargs)),
	args=(que, vs, fps, pargs)).start()

vs.start()
x0, y0 = que.get()


end = time.time()


print(end-start)
os.chdir('/home/pi')

freqx, pxxx = SIG.periodogram(xposarr, fs=1, nfft=len(xposarr))
freqy, pxxy = SIG.periodogram(yposarr, fs=1, nfft=len(yposarr))

xspec = plt.loglog(freqx, pxxx)
plt.title('XSpec')
plt.show(xspec)

yspec = plt.loglog(freqy, pxxy)
plt.title('YSpec')
plt.show(yspec)

today = str(datetime.date.today())
files_today = glob.glob('*' + today + '*')
##camera.capture('frame.jpg')
np.savetxt("GaussFits" + str(len(files_today)) + "-" + today + ".dat", np.vstack(center).transpose())


