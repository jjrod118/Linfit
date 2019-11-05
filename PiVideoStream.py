# import the necessary packages
from picamera.array import PiRGBArray, PiYUVArray
from picamera import PiCamera
from threading import Thread
import cv2
import time
from multiprocessing import Process
import io
from Queue import *
from imutils.video import FPS
import collections

num_frames = 50

class PiVideoStream:
    def __init__(self, resolution=(2592, 240), framerate=8):
        # initialize the camera and stream
        self.queue = Queue()
        self.frame_num = 0
        self.camera = PiCamera(sensor_mode=4)
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.shutter_speed = 100
        self.camera.iso = 800
        self.camera.image_effect = 'none'
        self.camera.awb_gains = (0.9,0.0)
        self.camera.awb_mode = 'off'
        time.sleep(5.0)
        self.camera.exposure_mode = 'off'
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,format="bgr",  use_video_port=True)
        self.frame = None
        self.stopped = False

    def start(self):
        # start reading frames from the video stream
        self.update()
        return self
 
    def update(self):
        # keep looping infinitely until the video stream is stopped
        print "Starting Update"
        for f in self.stream:
            # grab the frame from the stream and put in queue
            self.frame = f.array
            self.queue.put(self.frame)
            self.frame_num += 1
	    #print(self.frame_num)
            # clear the stream buffer
            self.rawCapture.truncate(0)
	    #if self.frame_num == num_frames:
	    	#self.stop()
            # if the thread indicator variable is set, stop the video stream
             # and free up camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return 

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the video stream should be stopped
        self.stopped = True
