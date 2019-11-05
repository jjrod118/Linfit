
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
from resizeimage import resizeimage

w = 1024
h = 768
resolution = (w, h)
s = (w*h, 6)
#s = (2073600, 6)
np.zeros(s)
gauss = np.zeros(s)
gauss[:,0] = 1
xpos = np.arange(w)
for k in range(h):
	gauss[(k * w) : (k + 1) * w, 1] = xpos
	gauss[(k * w) : (k + 1) * w, 2] = k
	gauss[(k * w) : (k + 1) * w, 3] = xpos ** 2
	gauss[(k * w) : (k + 1) * w, 4] = k ** 2
	gauss[(k * w) : (k + 1) * w, 5] = xpos * k

tgauss = np.transpose(gauss)
one = np.matmul(tgauss, gauss)
two = np.linalg.inv(one)
almost = np.matmul(two, tgauss)
center = []
xposarr = []
yposarr = []

def gaussfit(jpgname):
	jpg = Image.open("/home/pi/Desktop/" + jpgname + ".jpg")
	newjpg = resizeimage.resize_cover(jpg, [1024, 768])
	data = np.array(newjpg)
	#print(data)
	avg = np.average(data, 2)
	#print(avg)
	Yvec = np.reshape(avg.T, -1)
	#print(Yvec)
	Yvec = np.log(Yvec)
	coeff = np.matmul(almost, Yvec)
	f = coeff[0]
	d = coeff[1]
	e = coeff[2]
	a = coeff[3]
	b = coeff[4]
	c = coeff[5]
	y0 = (e - ((c * d) / (2* a))) / ((-2 * b) + ((c ** 2) / (2 * a)))
	x0 = (d + (c * y0)) / (-2 * a)
	xposarr.append(x0)
	yposarr.append(y0)
	values = np.array([x0, y0])
	center.append(values)
	return x0, y0

gaussfit("example1")

today = str(datetime.date.today())
files_today = glob.glob('*' + today + '*')
##camera.capture('frame.jpg')
np.savetxt("GaussFits" + str(len(files_today)) + "-" + today + ".dat", np.vstack(center).transpose())
