from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# we are using a pre trained model
CLASSES = ["person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))





