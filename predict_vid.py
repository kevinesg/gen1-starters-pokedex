import config.C as C
from config.label_frames import label_frames
import numpy as np
import cv2
import tensorflow as tf

tf.keras.backend.clear_session()
model = tf.saved_model.load(C.MODEL)

VIDEO_FILE = 'video.mp4'    # change this to your own video file
USE_CAM = False

if USE_CAM:
    vid = cv2.VideoCapture(0)
else:
    vid = cv2.VideoCapture(VIDEO_FILE)

while True:
    grabbed, frame = vid.read()

    if not grabbed:
        break

    label_frames(frame, model)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()