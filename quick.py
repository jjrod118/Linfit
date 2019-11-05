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
from PiVideoStream import *
from imutils.video import FPS
from picamera.array import PiRGBArray
from threading import Thread
from multiprocessing import Process
import argparse
import cv2
import numpy as np
import matplotlib as plt
import psutil, os, sys
import Queue
import datetime
import glob

# fixed variables
num_frames = 420
height = 1944
width = 2592

def analyze_vs(vstream, fps0, args0):
    """takes red data from vstream images and prepares 1-d arrays for analysis"""
    current_frame = 0
    rdats = []
    while fps._numFrames < args0["num_frames"]:
        frame = vstream.queue.get() # get frame
        if frame != None:
            data = np.fromstring(frame, dtype=np.uint8).reshape((240,2592,3))
            r = data[:,:,2] # stream outputs bgr format so r (index 2) is red channel
            rav = np.mean(r[133:143,:],axis=0) # averages rows where diffraction pattern prominent
            bkg = (np.mean(r[123:133,:],axis=0)+np.mean(r[143:153,:],axis=0))/2. # background estimate
            rdat = rav - bkg # subtract background
            rdats.append(rdat)
            print(current_frame)
            current_frame += 1
            fps0.update()

    fps0.stop()
    vstream.stop()
    print("{:.2f}".format(fps0.fps()))
    return rdats

# construct the argument parser and parse the args
ap = argparse.ArgumentParser()
ap.add_argument("-n","--num-frames", type=int, default=num_frames,
        help="# of frames to loop over")
ap.add_argument("-d", "--display", type=int, default=-1)
ap.add_argument("-t", "--type", choices=['silica', 'tungsten', 'other'], help="specify the type of fiber for filenames")
pargs = vars(ap.parse_args())


#Initialize the serial port, video stream, fps monitoring, and queue
#ser = serial.Serial('/dev/ttyAMA0')
vs = PiVideoStream()
fps = FPS().start()
que = Queue.Queue()
print('hi')
# start stepper then setup and start video stream and analyze_vs
#ser.write('-3800\n')
print('mid')
Thread(target=lambda q, vs, fps, pargs: q.put(analyze_vs(vs, fps, pargs)),
    args=(que, vs, fps, pargs)).start()
vs.start()
print('cmon')
# get data output from Video Stream and send stage to starting position
rdats = que.get()
#ser.write('3800\n')


# 76-105 calculate diameter for each image
index = 0
start = time.time()
fit = [1e-4,0,0]
ds = []
indices = []

for rdat in rdats:
    # takes psd of data looks for extrema and returns the largest one
    pdat0, pdat1 = psd(rdat, len(rdat), 1)
    max_vals = argrelextrema(pdat1,np.greater,order=3)[0]
    maxloc = np.argmax(pdat1[max_vals])

    # tries to handle if spatial frequency of peak is too low
    if max_vals[maxloc] < 8:
        maxloc = max_vals[maxloc+1]
    else:
        maxloc = max_vals[maxloc]


    fit[1] = pdat0[maxloc]
    fit[2] = pdat1[maxloc]
    out = lmfitter(pdat0[maxloc-1:maxloc+2],pdat1[maxloc-1:maxloc+2],lorentz,fit)
    spacing = 1.0/out[1]
    if spacing > 7.:
        diam = 9.5e-3*660e-9/(spacing*1.12e-6)
        print( diam )
        ds.append(diam)
        indices.append(index)
    index += 1
end = time.time()
print(end-start)

# move to data directory
os.chdir('/home/pi')

# prepare filename and write data to file
today = str(datetime.date.today())
files_today = glob.glob('*' + today + '*')
np.savetxt("qzfiber" + str(len(files_today)) + "-" + today + "_"+ str(pargs.type) + ".dat",np.vstack((indices, ds)).transpose())

