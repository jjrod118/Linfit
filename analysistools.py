import numpy as np
import scipy.optimize as opt

lorentz = lambda p, x: p[2]/(np.pi*(p[0]+(x-p[1])**2/p[0]))

def psd(data, num_points, sample_rate, win="Hann"):
    """Calculate power spectral density."""
    # TODO: choice of window function is ignored for now
    win = np.array([0.5 * (1 - np.cos(2*np.pi * i/num_points)) for i in
        range(num_points)])
    winsum1 = sum(win)
    winsum2 = sum(win**2)
    nyquist = sample_rate / 2.
    resolution = nyquist / (num_points/2 +1)
    f = np.arange(0, nyquist, resolution)
    num_ffts = (2 * len(data)/num_points) - 1  # number of FFTs to average
    enbw = sample_rate * winsum2 / (winsum1**2)
    psd_data = np.zeros(num_points/2 + 1)      # result array
    #print num_ffts
    for i in range(num_ffts):
        windowed = data[(i*num_points/2):((i*num_points/2)+num_points)] * win
        tmp = np.fft.rfft(windowed)
        psd_data += 2 * abs(tmp)**2 / (winsum1**2 * enbw)
    psd_data = psd_data / num_ffts
    return f, psd_data

def lmfitter(t,data,fitfunc,guess):
    """Just a wrapper for scipy.optimize.leastsq"""
    errfunc = lambda p, x, y: fitfunc(p, x) - y
    result = opt.leastsq(errfunc,guess,args=(t,data))
    return result[0]

