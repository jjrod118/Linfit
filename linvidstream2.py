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
import PiVideoStream2 as PVS
import picamera
from guppy import hpy
import scipy.signal as SIG
import timeit

start = time.time()

num_frames = 500
h = 2464
w = 3280
resolution = (w, h)
r2 = (h, w)

sx = (3280, 3)
sy = (2464, 3)
bigx = np.zeros(sx)
bigy = np.zeros(sy)
bigx[:,0] = 1
xpos = np.arange(w)
bigx[:,1] = xpos
bigx[:,2] = xpos ** 2
ypos = np.arange(h)
bigy[:,0] = 1
bigy[:,1] = ypos
bigy[:,2] = ypos ** 2

tbigx = np.transpose(bigx)
onex = np.matmul(tbigx, bigx)
twox = np.linalg.inv(onex)
almostx = np.matmul(twox, tbigx)
tbigy = np.transpose(bigy)
oney = np.matmul(tbigy, bigy)
twoy = np.linalg.inv(oney)
almosty = np.matmul(twoy, tbigy)

center = []
cenxa = []
cenya = []

#tenframes = np.zeros(resolution)
#print(tenframes)

#stream = io.BytesIO()

def bayerfit(vstream, fps0, args0):
	current_frame = 0
        index = 0
        tenframes = np.zeros(r2)
        while fps._numFrames < args0["num_frames"]:
		print('pls')
                frame = vstream.queue.get()
		print('here')
                if frame is not None:
                	ver = {
                		'RP_ov5647': 1,
                		'RP_imx219': 2,
                		}[camera.exif_tags['IFD0.Mode1']]
			offset = {
        			1: 6404096,
        			2: 10270208,
        			}[ver]
			data =  vstream.streamIO.getvalue()[-offset:]
			assert data[:4] == 'BRCM'
			data = data[32768:]
			data = np.fromstring(data, dtype=np.uint8)
			reshape, crop = {
        			1: ((1952, 3264), (1944, 3240)),
        			2: ((2480, 4128), (2464, 4100)),
        			}[ver]
			data = data.reshape(reshape)[:crop[0], :crop[1]]
			data = data.astype(np.uint16) << 2
			for byte in range(4):
        			data[:, byte::5] |= ((data[:, 4::5] >> ((4 - byte) * 2)) & 0b11)
			data = np.delete(data, np.s_[4::5], 1)
			rgb = np.zeros(data.shape + (3,), dtype=data.dtype)
			rbg[1::2, 0::2, 0] = data[1::2, 0::2] #RED
			rbg[0::2, 0::2, 1] = data[0::2, 0::2] #GREEN
			rbg[1::2, 1::2, 1] = data[1::2, 1::2] #GREEN
			rbg[0::2, 1::2, 2] = data[0::2, 1::2] #BLUE
                        blue = rbg[0::2, 1::2, 2]
                        tenframes = tenframes + blue
                        index += 1
                        current_frame +=1
                        print(current_frame)
                        fps0.update()
                        if current_frame % 12 == 0:
                                print(current_frame)
				bluex = np.average(blue, 0)
                                bluey = np.average(blue, 1)
                                Yvecx = np.log(bluex + 1e-6)
                                Yvecy = np.log(bluey + 1e-6)
                                coeffx = np.matmul(almostx, Yvecx)
                                coeffy = np.matmul(almosty, Yvecy)
                                a_x = coeffx[0]
                                b_x = coeffx[1]
                                c_x = coeffx[2]
                                a_y = coeffy[0]
                                b_y = coeffy[1]
                                c_y = coeffy[2]
                                cenx = -b_x/(2 * c_x)
                                ceny = -b_y/(2 * c_y)
                                cenxy = np.array([cenx, ceny])
                                cenxa.append(cenx)
                                cenya.append(ceny)
                                center.append(cenxy)
                                index += 1
                                current_frame += 1
                                fps0.update()
        fps0.stop()
        vstream.stop()
        return cenx, ceny, bluex, Yvecx

#x0, y0 = que.get()

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=num_frames,
        help="# of frames to loop over")
ap.add_argument("-d", "--display", type=int, default=-1)
pargs = vars(ap.parse_args())
fr = 12
vs = PVS.PiVideoStream2(resolution=resolution, framerate=fr)
fps = FPS().start()
que = Queue.Queue()

threading.Thread(target=lambda q, vs, fps, pargs: q.put(bayerfit(vs, fps, pargs)),
        args=(que, vs, fps, pargs)).start()

end = time.time()


print(end-start)
os.chdir('/home/pi')

freqx, pxxx = SIG.periodogram(cenxa, fs=1, nfft=len(cenxa))
freqy, pxxy = SIG.periodogram(cenya, fs=1, nfft=len(cenya))

xspec = plt.loglog(freqx, pxxx)
plt.title('XSpec')
plt.show(xspec)

yspec = plt.loglog(freqy, pxxy)
plt.title('YSpec')
plt.show(yspec)

#bluep = plt.plot(bluex, xpos)
#plt.show(bluep)

#yvecp = plt.plot(Yvecx, xpos)
#plt.show(yvecp)


today = str(datetime.date.today())
files_today = glob.glob('*' + today + '*')
np.savetxt("LinFits" + str(len(files_today)) + "-" + today + ".dat", np.vstack(center).transpose())



