import google_streetview.api
import google_streetview.helpers
from dotenv import load_dotenv
from PIL import Image
from imutils import paths
import numpy as np
import cv2


# Define parameters for street view api
params = {
    'size': '640x640', # max 640x640 pixels
    'location': '33.933260, -83.382731',
    'heading': '0;60;120;180;240;300',
    'pitch': '0',
    'fov': '90',
    'key': 'AIzaSyA1rcyAoZTClEYgNTvkvv8mHKUIsnVMC-U',
    }

# Create a results object
# results = google_streetview.api.results(params)

api_list = google_streetview.helpers.api_list(params)

results = google_streetview.api.results(api_list)

results.preview()

# For each result in results, check if date is in range

results.download_links('images')

results.save_metadata('metadata')


def stitch_images(): 

    imagePaths = sorted(list(paths.list_images("images")))
    images = []

    print(imagePaths)
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        images.append(image)

    stitcher = cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)

    if status == 0:
        cv2.imwrite("output.png", stitched)

        cv2.imshow("Stitched", stitched)
        cv2.waitKey('q')
    else:
        print("error with stitching")

stitch_images()