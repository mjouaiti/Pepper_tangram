import cv2
import numpy as np
import imutils
import sys
import scipy.signal
import filter
import pandas as pd
import matplotlib.pyplot as plt

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

rec = pd.read_csv("/Users/Melanie/Documents/Studies/LORIA/Dominique/angles.csv")
data = rec.angle

plt.figure()
plt.plot(data)

plt.figure()
plt.plot(butter_highpass_filter(data, 0.1, .3))
plt.show()
