import numpy as np
import cv2
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import os
import cv2
from numpy import result_type
from signature import match
from virtualPainter import airsigner

# Matching Threshold
THRESHOLD = 50


def checkSimilarity():
    airsign1 = airsigner()
    airsign2 = airsigner()

    # np.arrays werden von 2D in 3D umgewandelt
    airsign1 = np.expand_dims(airsign1, axis=-1)  # Add a single channel (third dimension)
    airsign1 = np.repeat(airsign1, 3, axis=-1)  # Repeat the channel along the last dimension to create 3 channels
    airsign2 = np.expand_dims(airsign2, axis=-1)  # Add a single channel (third dimension)
    airsign2 = np.repeat(airsign2, 3, axis=-1)  # Repeat the channel along the last dimension to create 3 channels

    result = match(airsign1, airsign2)

    if result <= THRESHOLD:
        messagebox.showerror("Failure", "Signatures Do Not Match, they are "+str(result)+f" % similar!!")
        pass
    else:
        messagebox.showinfo("Success", "Signatures Match, they are "+str(result)+f" % similar!!")
    return True


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    checkSimilarity()
    cap = cv2.VideoCapture(0)
    cap.release()
    cv2.destroyAllWindows()