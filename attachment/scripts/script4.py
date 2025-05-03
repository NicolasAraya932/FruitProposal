import cv2
import numpy as np


img = cv2.imread("/workspace/RadianceFields/0000.png")

def inverted_colors(img):
    # Invert the colors of the image
    inverted_img = 255 - img
    return inverted_img
new_img = inverted_colors(img)

folder = "/workspace/RadianceFields/output/"

# Inverting every image from the folder and saving it to other folder

#creating new folder
import os

if not os.path.exists("/workspace/RadianceFields/output_/"):
    os.makedirs("/workspace/RadianceFields/output_/")

for filename in os.listdir(folder):
    if filename.endswith(".png"):
        img = cv2.imread(os.path.join(folder, filename))
        inverted_img = inverted_colors(img)
        cv2.imwrite(os.path.join("/workspace/RadianceFields/output_/", filename), inverted_img)
    else:
        continue
    



